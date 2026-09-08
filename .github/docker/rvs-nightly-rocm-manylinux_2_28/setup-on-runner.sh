#!/usr/bin/env bash
# Run on the self-hosted GPU runner host to build the manylinux_2_28 ROCm-matched docker image.
#
#   cd ROCmValidationSuite
#   ./.github/docker/rvs-nightly-rocm-manylinux_2_28/setup-on-runner.sh --from-tarball amdrocm10-rvs-....tar.gz

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
chmod +x .github/docker/rvs-nightly-rocm-manylinux_2_28/build-rocm-image.sh
exec .github/docker/rvs-nightly-rocm-manylinux_2_28/build-rocm-image.sh "$@"
