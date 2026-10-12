#!/usr/bin/env bash
# Launch the learned serve-bread policy from this repository.
set -euo pipefail
H2R_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
H2R_REPO_ROOT="$(cd -- "$H2R_SCRIPT_DIR/.." && pwd)"
: "${ISAAC_SIM_DIR:?Set ISAAC_SIM_DIR to your Isaac Sim 6.1 installation}"
if [[ ! -x "$ISAAC_SIM_DIR/python.sh" ]]; then
  echo "Missing executable: $ISAAC_SIM_DIR/python.sh" >&2
  exit 2
fi
unset LD_PRELOAD CUDA_VISIBLE_DEVICES PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi
H2R_CUDA_LIB="$ISAAC_SIM_DIR/exts/isaacsim.pip.nv/pip_prebundle/nvidia/cuda_runtime/lib"
if [[ -d "$H2R_CUDA_LIB" ]]; then
  export LD_LIBRARY_PATH="$H2R_CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
cd -- "$H2R_REPO_ROOT"
exec "$ISAAC_SIM_DIR/python.sh" "$H2R_REPO_ROOT/inference/serve_bread.py" \
  --headless --gpu "${H2R_GPU:-1}" "$@"
