# Installation

Run the commands below from the HUMANO repository root on Linux x86_64. Use a machine with an NVIDIA GPU and driver suitable for Isaac Sim, the CUDA toolkit for building cuRobo, a C++ compiler, Git with Git LFS, and uv. Keep your policy server in its own model environment.

## Python environment

The repository targets Python 3.10. Install the locked dependencies first so cuRobo can build its CUDA extensions against the project's PyTorch installation.

```bash
uv sync --locked --python 3.10 --no-install-package nvidia-curobo
source .venv/bin/activate
bash scripts/setup/install_curobo.sh
```

The build uses the visible GPU architecture. Set `TORCH_CUDA_ARCH_LIST` only when targeting a different GPU or building without a visible GPU. Set `CUDA_HOME` to your toolkit location if the compiler cannot find it. The build script checks that the cuRobo CUDA extensions import successfully. To check the evaluation entry point, run:

```bash
humano-eval --help
```

For local resource paths and rendering settings, copy `.env.sample` to `.env` and edit it for your machine. Review the NVIDIA license terms before using Isaac Sim and follow its acceptance prompt.

## SONIC and scene resources

The default SONIC decoder runs on CPU. The project selects its ONNX Runtime build through uv. If you choose GPU decoding with `--sonic-provider gpu`, install the compatible CUDA 12 TensorRT 8.6.1 libraries and expose their library directory through `LD_LIBRARY_PATH`. The rendering and policy inference devices are configured separately.

Download the SONIC model files into the checkout:

```bash
python scripts/setup/download_model.py
```

For a fixed model revision, pass `--revision` to the downloader. Prepare the robot and scene assets required by your evaluation dataset using the [simulation resource installation guide](data/resources.md). Use the matching dataset version, scene bindings and policy checkpoint in the [single-task quickstart](quickstart.md).
