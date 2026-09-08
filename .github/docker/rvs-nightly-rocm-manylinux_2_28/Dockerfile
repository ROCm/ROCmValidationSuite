# ROCm nightly-test runtime image on manylinux_2_28 (AlmaLinux 8).
# Same SDK tarball layout as build_packages_local.sh
# (therock-dist-linux-<GPU_FAMILY>-<ROCM_VERSION>.tar.gz from multiarch or release listing).
#
# Build on any docker host (typically the self-hosted runner):
#   ./.github/docker/rvs-nightly-rocm-manylinux_2_28/build-rocm-image.sh \
#     --rocm-version 10.1.0a20260819 --gpu-family multiarch

FROM quay.io/pypa/manylinux_2_28_x86_64

ARG ROCM_VERSION
ARG GPU_FAMILY=multiarch
ARG ROCM_SDK_BASE_URL
ARG ROCM_INSTALL_PATH=/opt/rocm/install

RUN yum install -y wget tar pciutils kmod python3 ca-certificates \
    && yum clean all

RUN test -n "${ROCM_VERSION}" && test -n "${ROCM_SDK_BASE_URL}"

RUN mkdir -p "${ROCM_INSTALL_PATH}" \
    && wget -q -O /tmp/rocm-sdk.tar.gz \
         "${ROCM_SDK_BASE_URL%/}/therock-dist-linux-${GPU_FAMILY}-${ROCM_VERSION}.tar.gz" \
    && tar -xzf /tmp/rocm-sdk.tar.gz -C "${ROCM_INSTALL_PATH}" --strip-components=1 \
    && rm -f /tmp/rocm-sdk.tar.gz \
    && test -x "${ROCM_INSTALL_PATH}/bin/rocminfo" \
    && test -x "${ROCM_INSTALL_PATH}/bin/amd-smi"

COPY env-rocm.sh /etc/profile.d/rocm-env.sh

ENV ROCM_PATH=${ROCM_INSTALL_PATH}
ENV TARGET_ROCM_PATH=${ROCM_INSTALL_PATH}
ENV ROCM_VERSION=${ROCM_VERSION}
ENV GPU_FAMILY=${GPU_FAMILY}
ENV LD_LIBRARY_PATH=${ROCM_INSTALL_PATH}/lib/rocm_sysdeps/lib:${ROCM_INSTALL_PATH}/lib/llvm/lib:${ROCM_INSTALL_PATH}/lib

WORKDIR /workspace
