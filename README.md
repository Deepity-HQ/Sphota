# Sphota

A WIP Deepity Large Language Model, perceiving text via streams, and updating online using Predictive Coding.

## Development

These are recommended commands for development:

```bash
# Deepity comes from PyPI (a pre-release; the [cuda] extra bundles the CUDA runtime and cuBLAS).
# It is a project dependency, so `uv run` installs it. To install or upgrade by hand:
uv pip install --pre --upgrade "pydeepity[cuda]"
# The module is imported as `deepity`.

# Run Sphota, either by sbatch or manually:
uv run python -m sphota.main
# or
sbatch run.slurm

# View logs:
cat logs/*.log
```