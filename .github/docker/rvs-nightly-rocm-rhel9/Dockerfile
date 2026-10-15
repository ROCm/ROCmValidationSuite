# ROCm + RVS runtime image on RHEL 9 (redhat/ubi9:9).
# ROCm comes from nightly yum; RVS from extras yum (nightly if published, else stable).
#
# Build:
#   ./.github/docker/build-rocm-sdk-image.sh --context .github/docker/rvs-nightly-rocm-rhel9 --channel nightly

FROM redhat/ubi9:9

ARG ROCM_VERSION
ARG ROCM_REPO_BASEURL
ARG RVS_REPO_BASEURL
# Public key URL, not a secret. Nightly RPMs stay unsigned (gpgcheck=0).
ARG ROCM_GPG_URL=https://stable.repo.amd.com/rocm/gpg/packages.gpg
ARG GPU_TARGET=multiarch
ARG ROCM_PACKAGE
ARG RVS_PACKAGE
ARG ROCM_INSTALL_PATH=/opt/rocm

RUN dnf -y update \
    && dnf -y install sudo wget ca-certificates dnf-plugins-core \
    && dnf clean all

RUN test -n "${ROCM_REPO_BASEURL}" && test -n "${RVS_REPO_BASEURL}"

# Nightly ROCm RPMs are unsigned; gpgcheck=0 is required.
# Default GPU_TARGET=multiarch installs the ISA-less meta (e.g. amdrocm10.1).
# Set GPU_TARGET=gfx942 (etc.) for a single-ISA package.
RUN printf '%s\n' \
      '[amdrocm-nightly]' \
      'name=ROCm Nightly' \
      "baseurl=${ROCM_REPO_BASEURL}" \
      'enabled=1' \
      'gpgcheck=0' \
      "gpgkey=${ROCM_GPG_URL}" \
    | tee /etc/yum.repos.d/amdrocm-nightly.repo \
    && dnf clean all \
    && if [ -n "${ROCM_PACKAGE}" ]; then \
         dnf -y --nogpgcheck install "${ROCM_PACKAGE}"; \
       elif [ "${GPU_TARGET}" = "multiarch" ] || [ "${GPU_TARGET}" = "all" ]; then \
         pkg="$(dnf repoquery --latest-only --qf '%{name}' 'amdrocm*.*' | grep -E '^amdrocm[0-9]+\.[0-9]+$' | sort -V | tail -n 1)"; \
         test -n "${pkg}"; \
         dnf -y --nogpgcheck install "${pkg}"; \
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
       ROCM_GPG_URL="${ROCM_GPG_URL}" \
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
ENV RHEL_VERSION=9
ENV PATH=${ROCM_INSTALL_PATH}/bin:${PATH}

WORKDIR /workspace

# Tests run as rvs. video and render are the device groups for /dev/kfd and /dev/dri.
# The launch script adds the host GIDs, which often differ from these image GIDs.
RUN groupadd -f video \
    && groupadd -f render \
    && useradd --create-home --shell /bin/bash --groups video,render rvs \
    && mkdir -p /workspace \
    && chown rvs:rvs /workspace

USER rvs

# $$ keeps ROCM_PATH for runtime. rocm-env.sh points it at rocminfo.
HEALTHCHECK CMD bash -lc 'test -x "$${ROCM_PATH}/bin/rocminfo"'
