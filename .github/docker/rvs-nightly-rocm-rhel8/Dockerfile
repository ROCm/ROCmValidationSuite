# ROCm + RVS runtime image on RHEL 8 (Rocky Linux 8).
# ROCm comes from nightly yum; RVS from extras yum (nightly if published, else stable).
#
# Build:
#   ./.github/docker/rvs-nightly-rocm-rhel8/build-rocm-image.sh --channel nightly

FROM rockylinux:8

ARG ROCM_VERSION
ARG ROCM_REPO_BASEURL
ARG RVS_REPO_BASEURL
ARG ROCM_GPG_KEY=https://stable.repo.amd.com/rocm/gpg/packages.gpg
ARG GPU_TARGET=gfx942
ARG ROCM_PACKAGE
ARG RVS_PACKAGE
ARG ROCM_INSTALL_PATH=/opt/rocm

RUN dnf -y update \
    && dnf -y install sudo wget ca-certificates dnf-plugins-core \
    && dnf clean all

RUN test -n "${ROCM_REPO_BASEURL}" && test -n "${RVS_REPO_BASEURL}"

# Nightly ROCm RPMs are unsigned; gpgcheck=0 is required.
# Install only the gfx-specific meta (e.g. amdrocm10.1-gfx942), not unversioned amdrocm-*.
RUN printf '%s\n' \
      '[amdrocm-nightly]' \
      'name=ROCm Nightly' \
      "baseurl=${ROCM_REPO_BASEURL}" \
      'enabled=1' \
      'gpgcheck=0' \
      "gpgkey=${ROCM_GPG_KEY}" \
    | tee /etc/yum.repos.d/amdrocm-nightly.repo \
    && dnf clean all \
    && if [ -n "${ROCM_PACKAGE}" ]; then \
         dnf -y --nogpgcheck install "${ROCM_PACKAGE}"; \
       else \
         pkg="$(dnf repoquery --latest-only --qf '%{name}' "amdrocm*-${GPU_TARGET}" | grep -E "^amdrocm[0-9.]+-${GPU_TARGET}$" | sort -V | tail -n 1)"; \
         test -n "${pkg}"; \
         dnf -y --nogpgcheck install "${pkg}"; \
       fi \
    && dnf clean all

COPY install-rvs.sh /tmp/install-rvs.sh
RUN chmod +x /tmp/install-rvs.sh \
    && RVS_REPO_BASEURL="${RVS_REPO_BASEURL}" \
       RVS_PACKAGE="${RVS_PACKAGE}" \
       ROCM_GPG_KEY="${ROCM_GPG_KEY}" \
       /tmp/install-rvs.sh \
    && rm -f /tmp/install-rvs.sh

COPY env-rocm.sh /etc/profile.d/rocm-env.sh

# rocminfo lives in /opt/rocm/core-<ver>/bin after nightly RPM install.
RUN bash -lc 'source /etc/profile.d/rocm-env.sh; \
    test -x "${ROCM_PATH}/bin/rocminfo"; \
    test -n "${EXTRAS_PATH}" && test -x "${EXTRAS_PATH}/bin/rvs"'

ENV ROCM_PATH=${ROCM_INSTALL_PATH}
ENV TARGET_ROCM_PATH=${ROCM_INSTALL_PATH}
ENV ROCM_VERSION=${ROCM_VERSION}
ENV GPU_TARGET=${GPU_TARGET}
ENV RHEL_VERSION=8
ENV PATH=${ROCM_INSTALL_PATH}/bin:${PATH}

WORKDIR /workspace
