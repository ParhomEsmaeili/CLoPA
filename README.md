# CLoPA — Continual Learning for Promptable Annotation

Interactive segmentation inference backend with continual adaptation.

## Installation

### Prerequisites

```bash
conda create -n clopa python=3.10
conda activate clopa
```

### CUDA support (optional)

Required for GPU inference. Install CUDA-compatible PyTorch **before** CLoPA so pip doesn't pull the CPU-only default:

```bash
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu126
```

### Install CLoPA

```bash
git clone https://github.com/ParhomEsmaeili/CLoPA.git
cd CLoPA

# Using pip
pip install -e .

# Or using uv (faster)
uv sync
```

## Architecture

- [Interaction event flow](docs/interaction-event-flow.md) — how a widget interaction becomes an `InferApp` request: `CLoPASession` accumulation, request building, the RAS orientation round-trip.

## Branches

- **main** — standalone, all dependencies explicit
- **ui-integration** — depends on `is-validate[clopa]` for framework integration

