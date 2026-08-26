#!/usr/bin/env bash
# Run on the self-hosted GPU runner host to build the Ubuntu 26.04 ROCm-matched docker image.
#
#   cd ROCmValidationSuite
#   ./.github/docker/rvs-nightly-rocm-ubuntu26.04/setup-on-runner.sh --from-tarball amdrocm10-rvs-....tar.gz

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
chmod +x .github/docker/rvs-nightly-rocm-ubuntu26.04/build-rocm-image.sh
exec .github/docker/rvs-nightly-rocm-ubuntu26.04/build-rocm-image.sh "$@"
