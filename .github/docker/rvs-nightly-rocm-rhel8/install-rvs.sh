#!/usr/bin/env bash
# Install RVS RPM without resolving unversioned amdrocm-* deps from the
# nightly ROCm repo (those metas pull every GPU ISA).
set -euo pipefail

RVS_REPO_BASEURL="${RVS_REPO_BASEURL:?}"
RVS_PACKAGE="${RVS_PACKAGE:-amdrocm-rvs}"
ROCM_GPG_KEY="${ROCM_GPG_KEY:-https://stable.repo.amd.com/rocm/gpg/packages.gpg}"

printf '%s\n' \
  '[rvs]' \
  'name=ROCm Validation Suite' \
  "baseurl=${RVS_REPO_BASEURL}" \
  'enabled=1' \
  'gpgcheck=0' \
  "gpgkey=${ROCM_GPG_KEY}" \
  'priority=50' \
  | tee /etc/yum.repos.d/rvs.repo

dnf clean all
mkdir -p /tmp/rvs-rpm
# Nightly ROCm must stay disabled: amdrocm10-rvs Requires: (amdrocm-blas or rocblas)
# and the unversioned amdrocm-blas meta pulls every gfx* package.
dnf -y download --nogpgcheck \
  --disablerepo='*' --enablerepo=rvs \
  --destdir /tmp/rvs-rpm \
  "${RVS_PACKAGE}"
rpm -ivh --nodeps --replacefiles /tmp/rvs-rpm/*.rpm
rm -rf /tmp/rvs-rpm

test -x /opt/rocm/extras-*/bin/rvs 2>/dev/null || true
rvs_bin=""
for f in /opt/rocm/extras-*/bin/rvs; do
  if [ -x "$f" ]; then
    rvs_bin="$f"
    break
  fi
done
if [ -z "$rvs_bin" ]; then
  echo "rvs binary missing after RPM install" >&2
  rpm -ql "${RVS_PACKAGE}" 2>/dev/null | head >&2 || true
  exit 1
fi
echo "Installed ${rvs_bin}"
