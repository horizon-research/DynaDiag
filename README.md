# DynaDiag
This repository contains the implementation of our ICML 2025 paper titled "[Dynamic Sparse Training of Diagonally Sparse Networks](https://arxiv.org/abs/2506.11449)"

# Setup environment
Use the `environment.yml` file to create a conda environment as:
`conda env create -f environment.yml`

# Usage
To sparsify a lineary layer with DynaDiag, just specify `--linear-type temp_diag`, set sparsity for the layer and provide the temperatures and a temperature scheduler as shown in the MLP example below.

# MLP Example

Run the MNIST MLP training from the repository root. Example (TempSoftmaxDiagLinear):

```bash
python mlp.py \
  --linear-type temp_diag \
  --hidden-dims 256 256 \
  --sparsity 0.5 \
  --temp-schedule cosine \
  --temp-start 5.0 \
  --temp-end 0.05 \
  --temp-min 0.05 \
  --temp-max 5.0 \
  --epochs 100 \
  --alpha-freeze-mode at_epoch \
  --alpha-freeze-epoch 30 \
  --lr 1e-3 \
  --batch-size 128 \
  --wandb
```

Notes: the script will download MNIST to `--data-dir` if missing. Use `--linear-type dense` for a dense baseline, and `--force-cpu` to run on CPU.

# ToDos
1) MLP Example (without speedup)
2) CUDA Kernel
3) Integrated CUDA kernel in PyTorch
4) Upload the pre-trained models
