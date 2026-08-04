#!/usr/bin/env bash
# Run on the self-hosted GPU runner host to build the ROCm-matched docker image.
#
#   cd ROCmValidationSuite
#   ./.github/docker/rvs-nightly-rocm/setup-on-runner.sh [--channel nightly|release]
#   ./.github/docker/rvs-nightly-rocm/setup-on-runner.sh --from-tarball amdrocm7-rvs-....tar.gz

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
chmod +x .github/docker/rvs-nightly-rocm/build-rocm-image.sh
exec .github/docker/rvs-nightly-rocm/build-rocm-image.sh "$@"
