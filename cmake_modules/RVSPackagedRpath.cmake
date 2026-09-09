# Canonical relocatable RUNPATH for packaged RVS artifacts (DEB/RPM/TGZ).
# Used by CMAKE_INSTALL_RPATH and cpack-patch-rpath.cmake (via configure_file).
#
# Expects ROCM_PATH and ROCM_MAJOR_VERSION in the including scope.

# libomp.so lives under lib/llvm/lib/<host-triple>/ when ROCm LLVM uses
# LLVM_ENABLE_PER_TARGET_RUNTIME_DIR.
set(RVS_HOST_TARGET_TRIPLE "")
find_program(RVS_ROCM_CLANG
  NAMES amdclang++ clang++
  HINTS "${ROCM_PATH}/bin" "${ROCM_PATH}/lib/llvm/bin" "/opt/rocm" "/opt/rocm/core-${ROCM_MAJOR_VERSION}"
  PATH_SUFFIXES bin llvm/bin
  NO_CACHE)
if(RVS_ROCM_CLANG)
  execute_process(
    COMMAND "${RVS_ROCM_CLANG}" --print-target-triple
    OUTPUT_VARIABLE RVS_HOST_TARGET_TRIPLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rvs_triple_rc
    ERROR_QUIET)
  if(_rvs_triple_rc EQUAL 0 AND RVS_HOST_TARGET_TRIPLE)
    message(STATUS
      "RVS LLVM host runtime dir (libomp): /opt/rocm/lib/llvm/lib/${RVS_HOST_TARGET_TRIPLE}")
  else()
    message(WARNING
      "Could not query --print-target-triple from ${RVS_ROCM_CLANG}; "
      "libomp may not be found at runtime.")
    set(RVS_HOST_TARGET_TRIPLE "")
  endif()
else()
  message(WARNING
    "No ROCm clang found under ${ROCM_PATH}; libomp per-target RPATH omitted.")
endif()

function(rvs_get_packaged_rpath_list out_var)
  set(_rpath
    "\$ORIGIN"
    "\$ORIGIN/../lib"
    "\$ORIGIN/../lib/rvs"
    "/opt/rocm/core-${ROCM_MAJOR_VERSION}/lib"
    "/opt/rocm/core-${ROCM_MAJOR_VERSION}/lib/llvm/lib"
    "/opt/rocm/lib"
    "/opt/rocm/lib/llvm/lib")
  if(RVS_HOST_TARGET_TRIPLE)
    list(APPEND _rpath
      "/opt/rocm/core-${ROCM_MAJOR_VERSION}/lib/llvm/lib/${RVS_HOST_TARGET_TRIPLE}"
      "/opt/rocm/lib/llvm/lib/${RVS_HOST_TARGET_TRIPLE}")
  endif()
  set(${out_var} ${_rpath} PARENT_SCOPE)
endfunction()

function(rvs_get_packaged_rpath_colon out_var)
  rvs_get_packaged_rpath_list(_list)
  string(JOIN ":" _colon ${_list})
  set(${out_var} "${_colon}" PARENT_SCOPE)
endfunction()
