# ROCm nightly-test runtime image — same SDK tarball layout as build_packages_local.sh
# (therock-dist-linux-<GPU_FAMILY>-<ROCM_VERSION>.tar.gz from multiarch or release listing).
#
# Build on any docker host (typically the self-hosted runner):
#   ./.github/docker/rvs-nightly-rocm/build-rocm-image.sh \
#     --rocm-version 7.14.0a20260528 --gpu-family gfx110X-all

FROM ubuntu:24.04

ARG ROCM_VERSION
ARG GPU_FAMILY=gfx110X-all
ARG ROCM_SDK_BASE_URL
ARG ROCM_INSTALL_PATH=/opt/rocm/install

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    tar \
    pciutils \
    kmod \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN test -n "${ROCM_VERSION}" && test -n "${ROCM_SDK_BASE_URL}"

RUN mkdir -p "${ROCM_INSTALL_PATH}" \
    && wget -q -O /tmp/rocm-sdk.tar.gz \
         "${ROCM_SDK_BASE_URL%/}/therock-dist-linux-${GPU_FAMILY}-${ROCM_VERSION}.tar.gz" \
    && tar -xzf /tmp/rocm-sdk.tar.gz -C "${ROCM_INSTALL_PATH}" --strip-components=1 \
    && rm -f /tmp/rocm-sdk.tar.gz \
    && test -x "${ROCM_INSTALL_PATH}/bin/rocminfo" \
    && "${ROCM_INSTALL_PATH}/bin/amd-smi" version

COPY env-rocm.sh /etc/profile.d/rocm-env.sh

ENV ROCM_PATH=${ROCM_INSTALL_PATH}
ENV TARGET_ROCM_PATH=${ROCM_INSTALL_PATH}
ENV ROCM_VERSION=${ROCM_VERSION}
ENV GPU_FAMILY=${GPU_FAMILY}
ENV LD_LIBRARY_PATH=${ROCM_INSTALL_PATH}/lib/rocm_sysdeps/lib:${ROCM_INSTALL_PATH}/lib/llvm/lib:${ROCM_INSTALL_PATH}/lib

WORKDIR /workspace
