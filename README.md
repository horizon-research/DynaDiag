# DynaDiag
This repository contains the implementation of our ICML 2025 paper titled "[Dynamic Sparse Training of Diagonally Sparse Networks](https://arxiv.org/abs/2506.11449)"

# Setup environment
Use the `environment.yml` file to create a conda environment as:
`conda env create -f environment.yml`

# MLP Example (Without Speedup)

Run the MNIST MLP training from the repository root. Example (TempSoftmaxDiagLinear):

```bash
python MLP/mlp.py --linear-type temp_diag --hidden-dims 256 256 --epochs 100 --batch-size 128 --data-dir ./mnist
```

Notes: the script will download MNIST to `--data-dir` if missing. Use `--linear-type dense` for a dense baseline, and `--force-cpu` to run on CPU.

# ToDos
1) MLP Example (without speedup)
2) CUDA Kernel
3) Integrated CUDA kernel in PyTorch
4) Upload the pre-trained models
