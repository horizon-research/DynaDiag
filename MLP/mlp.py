import argparse
import math
import os
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
    _TORCHVISION_AVAILABLE = True
except Exception:
    # Allow running without torchvision
    _TORCHVISION_AVAILABLE = False

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

try:
    # Import when used as a package module: `python -m dynaDiag.mlp`
    from .linear_softmax import TempSoftmaxDiagLinear  # type: ignore
except Exception:
    # Import when run as a standalone script: `python mlp.py`
    from linear_softmax import TempSoftmaxDiagLinear  # type: ignore


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: List[int],
        linear_type: Literal["dense", "temp_diag"] = "dense",
        activation: Literal["relu", "gelu", "tanh"] = "relu",
        dropout: float = 0.0,
        tempdiag_sparsity: float = 0.1,
        tempdiag_temperature: float = 1.0,
        tempdiag_chunk_size: int = 64,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device_ = device if device is not None else get_device()

        act_layer: nn.Module
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "gelu":
            act_layer = nn.GELU()
        elif activation == "tanh":
            act_layer = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        def make_linear(in_features: int, out_features: int) -> nn.Module:
            if linear_type == "dense":
                return nn.Linear(in_features, out_features, bias=bias)
            elif linear_type == "temp_diag":
                return TempSoftmaxDiagLinear(
                    in_features,
                    out_features,
                    device=self.device_,
                    sparsity=tempdiag_sparsity,
                    temperature=tempdiag_temperature,
                    chunk_size=tempdiag_chunk_size,
                    bias=bias,
                )
            else:
                raise ValueError(f"Unsupported linear_type: {linear_type}")

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(make_linear(prev_dim, hidden_dim))
            layers.append(act_layer)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        # Classifier head: always dense 
        layers.append(nn.Linear(prev_dim, num_classes, bias=bias))

        self.net = nn.Sequential(*layers)
        self.to(self.device_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def iter_temp_layers(module: nn.Module):
    for m in module.modules():
        if isinstance(m, TempSoftmaxDiagLinear):
            yield m


class TemperatureScheduler:
    def __init__(
        self,
        scheme: str,
        total_epochs: int,
        temp_start: float,
        temp_end: float,
        temp_min: float,
        temp_max: float,
        warmup_epochs: int,
        gamma: float,
        step_size: int,
        cycles: int,
    ) -> None:
        self.scheme = scheme
        self.total_epochs = max(1, int(total_epochs))
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.gamma = float(gamma)
        self.step_size = max(1, int(step_size))
        self.cycles = max(1, int(cycles))

    def get(self, epoch_index_zero_based: int) -> float:
        e = max(0, min(self.total_epochs - 1, int(epoch_index_zero_based)))
        T = self.total_epochs
        if self.scheme == "constant":
            return self.temp_start
        if self.scheme == "linear":
            alpha = e / max(1, T - 1)
            return (1 - alpha) * self.temp_start + alpha * self.temp_end
            import math as _math
            prog = e / max(1, T - 1)
            return self.temp_min + 0.5 * (self.temp_max - self.temp_min) * (1 + _math.cos(_math.pi * prog))
        if self.scheme == "exponential":
            val = self.temp_start * (self.gamma ** e)
            return max(self.temp_end, val)
        if self.scheme == "step":
            k = e // self.step_size
            val = self.temp_start * (self.gamma ** k)
            return max(self.temp_end, val)
        if self.scheme == "warmup_cosine":
            import math as _math
            if e < self.warmup_epochs:
                alpha = e / max(1, self.warmup_epochs)
                return (1 - alpha) * self.temp_start + alpha * self.temp_max
            # remaining
            rem = max(1, T - self.warmup_epochs)
            prog = (e - self.warmup_epochs) / max(1, rem - 1)
            return self.temp_min + 0.5 * (self.temp_max - self.temp_min) * (1 + _math.cos(_math.pi * prog))
        if self.scheme == "cyclic":
            import math as _math
            cycle_len = max(1, T // self.cycles)
            pos_in_cycle = e % cycle_len
            prog = pos_in_cycle / max(1, cycle_len - 1)
            return self.temp_min + 0.5 * (self.temp_max - self.temp_min) * (1 + _math.cos(_math.pi * prog))
        # Fallback
        return self.temp_start


def get_mnist_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
    download: bool = True,
):
    if not _TORCHVISION_AVAILABLE:
        raise RuntimeError("torchvision is required for MNIST. Please install torchvision.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(root=data_dir, train=True, transform=transform, download=download)
    test_ds = datasets.MNIST(root=data_dir, train=False, transform=transform, download=download)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Flatten 28x28 images
        images = images.view(images.size(0), -1)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(images.size(0), -1)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configurable MLP for MNIST with dense or TempSoftmaxDiagLinear layers")

    # Data
    parser.add_argument("--data-dir", type=str, default=os.path.expanduser("./mnist"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)

    # Model
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 256], help="Hidden layer sizes, space-separated")
    parser.add_argument("--activation", type=str, choices=["relu", "gelu", "tanh"], default="relu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--linear-type", type=str, choices=["dense", "temp_diag"], default="dense")
    parser.add_argument("--bias", action="store_true", help="Use bias in linear layers")

    parser.add_argument("--sparsity", type=float, default=0.9, help="Sparsity for TempSoftmaxDiagLinear (fraction of zeros)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Initial temperature for TempSoftmaxDiagLinear")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size for TempSoftmaxDiagLinear computation")

    parser.add_argument(
        "--temp-schedule",
        type=str,
        choices=["constant", "linear", "cosine", "exponential", "step", "warmup_cosine", "cyclic"],
        default="constant",
        help="Temperature scheduling scheme over epochs",
    )
    parser.add_argument("--temp-start", type=float, default=None, help="Start temperature (defaults to --temperature)")
    parser.add_argument("--temp-end", type=float, default=0.001, help="End/min temperature for schedules that decay")
    parser.add_argument("--temp-min", type=float, default=None, help="Min temperature for cosine/cyclic (defaults to --temp-end)")
    parser.add_argument("--temp-max", type=float, default=None, help="Max temperature for cosine/warmup/cyclic (defaults to --temp-start)")
    parser.add_argument("--temp-warmup-epochs", type=int, default=0, help="Warmup epochs for warmup_cosine")
    parser.add_argument("--temp-gamma", type=float, default=0.95, help="Decay factor for step/exponential")
    parser.add_argument("--temp-step-size", type=int, default=10, help="Epochs per step for step schedule")
    parser.add_argument("--temp-cycles", type=int, default=1, help="Number of cycles for cyclic schedule")

    parser.add_argument("--alpha-nnz-threshold", type=float, default=1e-6, help="Threshold to count non-zero alpha weights")
    parser.add_argument(
        "--alpha-freeze-mode",
        type=str,
        choices=["never", "at_start", "at_epoch"],
        default="never",
        help="Control when to freeze alpha selection: never, at_start, or at a specific epoch",
    )
    parser.add_argument("--alpha-freeze-epoch", type=int, default=0, help="Epoch to freeze alpha when mode=at_epoch (1-based)")

    # Train
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="dynaDiag-MNIST", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/user")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name")

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = get_device(force_cpu=args.force_cpu)

    if args.linear_type == "temp_diag" and device.type == "cpu":
        # It's fine to run on CPU but warn about performance
        print("Warning: TempSoftmaxDiagLinear on CPU may be slow.")

    # Data
    train_loader, test_loader = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
    )

    # Model
    input_dim = 28 * 28
    num_classes = 10
    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=list(args.hidden_dims),
        linear_type=args.linear_type,
        activation=args.activation,
        dropout=args.dropout,
        tempdiag_sparsity=args.sparsity,
        tempdiag_temperature=args.temperature,
        tempdiag_chunk_size=args.chunk_size,
        bias=args.bias,
        device=device,
    )

    uses_temp_layers = any(True for _ in iter_temp_layers(model))
    if uses_temp_layers:
        temp_start = args.temp_start if args.temp_start is not None else args.temperature
        temp_min = args.temp_min if args.temp_min is not None else args.temp_end
        temp_max = args.temp_max if args.temp_max is not None else temp_start
        temp_sched = TemperatureScheduler(
            scheme=args.temp_schedule,
            total_epochs=args.epochs,
            temp_start=temp_start,
            temp_end=args.temp_end,
            temp_min=temp_min,
            temp_max=temp_max,
            warmup_epochs=args.temp_warmup_epochs,
            gamma=args.temp_gamma,
            step_size=args.temp_step_size,
            cycles=args.temp_cycles,
        )
    else:
        temp_sched = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_wandb = bool(args.wandb)
    if use_wandb and not _WANDB_AVAILABLE:
        print("wandb not installed; disable --wandb or install wandb.")
        use_wandb = False
    if use_wandb:
        wb_config = {
            "linear_type": args.linear_type,
            "hidden_dims": list(args.hidden_dims),
            "activation": args.activation,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        if uses_temp_layers:
            wb_config.update({
                "sparsity": args.sparsity,
                "init_temperature": args.temperature,
                "temp_schedule": args.temp_schedule,
                "temp_start": temp_start,
                "temp_end": args.temp_end,
                "temp_min": temp_min,
                "temp_max": temp_max,
                "warmup_epochs": args.temp_warmup_epochs,
                "gamma": args.temp_gamma,
                "step_size": args.temp_step_size,
                "cycles": args.temp_cycles,
                "alpha_nnz_threshold": args.alpha_nnz_threshold,
                "alpha_freeze_mode": args.alpha_freeze_mode,
                "alpha_freeze_epoch": args.alpha_freeze_epoch,
            })
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=wb_config)


    def try_freeze_alpha(epoch_one_based: int):
        if not uses_temp_layers:
            return
        if args.alpha_freeze_mode == "at_start" and epoch_one_based == 1:
            for layer in iter_temp_layers(model):
                layer.freeze_topk() 
        elif args.alpha_freeze_mode == "at_epoch" and epoch_one_based == args.alpha_freeze_epoch:
            for layer in iter_temp_layers(model):
                layer.freeze_topk()

    for epoch in range(1, args.epochs + 1):
        current_temp = None
        if uses_temp_layers and temp_sched is not None:
            current_temp = temp_sched.get(epoch - 1)
            for layer in iter_temp_layers(model):
                layer.set_temperature(current_temp)

        try_freeze_alpha(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, test_loader, device)
        if uses_temp_layers and current_temp is not None:
            print(f"Epoch {epoch:02d} | temp={current_temp:.4f} | loss={train_loss:.4f} | acc={acc*100:.2f}%")
        else:
            print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | acc={acc*100:.2f}%")

        if uses_temp_layers:
            alpha_stats = {}
            layer_idx = 0
            for layer in iter_temp_layers(model):
                weights = layer.get_alpha_weights().detach()
                nnz = (weights.abs() > args.alpha_nnz_threshold).sum().item()
                total = int(weights.numel())
                alpha_stats[f"layer{layer_idx}/alpha_nnz"] = nnz
                alpha_stats[f"layer{layer_idx}/alpha_density"] = nnz / max(1, total)
                
                effective_k = layer.get_effective_k(threshold=args.alpha_nnz_threshold)
                effective_sparsity = layer.get_effective_sparsity(threshold=args.alpha_nnz_threshold)
                target_k = layer.K
                
                alpha_stats[f"layer{layer_idx}/effective_k"] = effective_k
                alpha_stats[f"layer{layer_idx}/target_k"] = target_k
                alpha_stats[f"layer{layer_idx}/effective_sparsity"] = effective_sparsity
                alpha_stats[f"layer{layer_idx}/k_ratio"] = effective_k / max(1, target_k)
                
                layer_idx += 1
            if alpha_stats:
                k_str = ", ".join([f"{k.split('/')[0]}: k_eff={alpha_stats[k+'/effective_k']:.1f}/{alpha_stats[k+'/target_k']:.1f} (sp={alpha_stats[k+'/effective_sparsity']:.3f})" 
                                   for k in sorted({k.split('/')[0] for k in alpha_stats if k.endswith('/effective_k')})])
                try:
                    print(f"  {k_str}")
                except Exception:
                    print(f"  alpha stats: {alpha_stats}")

        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/acc": acc,
                "lr": optimizer.param_groups[0].get("lr", None),
            }
            if uses_temp_layers and current_temp is not None:
                log_dict["temperature"] = current_temp
            if uses_temp_layers:
                log_dict.update(alpha_stats)
            wandb.log(log_dict)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
