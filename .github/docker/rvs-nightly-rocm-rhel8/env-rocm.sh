# Source in container shells: RPM-installed ROCm Core + RVS extras.
# Nightly amdrocm RPMs land under /opt/rocm/core-<ver>, not /opt/rocm/install.
if [ -z "${ROCM_PATH:-}" ] || [ ! -x "${ROCM_PATH}/bin/rocminfo" ]; then
  if [ -x /opt/rocm/bin/rocminfo ]; then
    ROCM_PATH=/opt/rocm
  else
    for _core in /opt/rocm/core-*; do
      if [ -x "${_core}/bin/rocminfo" ]; then
        ROCM_PATH="${_core}"
        break
      fi
    done
    unset _core
  fi
fi
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
TARGET_ROCM_PATH="${TARGET_ROCM_PATH:-$ROCM_PATH}"
if [ -z "${EXTRAS_PATH:-}" ]; then
  for _extras in /opt/rocm/extras-*; do
    if [ -d "$_extras" ]; then
      EXTRAS_PATH="$_extras"
    fi
  done
  unset _extras
fi
EXTRAS_PATH="${EXTRAS_PATH:-}"
export ROCM_PATH TARGET_ROCM_PATH EXTRAS_PATH
export PATH="${EXTRAS_PATH:+${EXTRAS_PATH}/bin:}${ROCM_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${EXTRAS_PATH:+${EXTRAS_PATH}/lib:}${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib:${ROCM_PATH}/lib/llvm/lib/x86_64-unknown-linux-gnu:${ROCM_PATH}/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
