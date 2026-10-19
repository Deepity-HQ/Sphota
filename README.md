# Sphota

A WIP Deepity Large Language Model, perceiving text via streams, and updating online using Predictive Coding.

## Development

These are recommended commands for development:

```bash
# Build the Deepity library:
cd /path/to/Deepity
python build.py Release MKL --clean --native --DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

# Copy to pydeepity:
cp build/Release/bin/pydeepity.cpython-312-x86_64-linux-gnu.so pydeepity/pydeepity.cpython-312-x86_64-linux-gnu.so
# Copy pydeepity to Sphota:
cp pydeepity /path/to/Sphota && cd /path/to/Sphota

# Run Sphota, either by sbatch or manually:
uv run python -m sphota.main
# or
sbatch run.slurm

# View logs:
cat logs/*.log
```