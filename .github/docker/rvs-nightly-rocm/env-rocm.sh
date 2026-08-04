# Source in container shells: sets ROCm runtime paths (TheRock multiarch layout).
ROCM_PATH="${ROCM_PATH:-/opt/rocm/install}"
TARGET_ROCM_PATH="${TARGET_ROCM_PATH:-$ROCM_PATH}"
export ROCM_PATH TARGET_ROCM_PATH
export PATH="${ROCM_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib/rocm_sysdeps/lib:${ROCM_PATH}/lib/llvm/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
