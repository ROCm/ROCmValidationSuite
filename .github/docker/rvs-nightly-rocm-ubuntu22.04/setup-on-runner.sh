#!/usr/bin/env bash
# Run on the self-hosted GPU runner host to build the Ubuntu 22.04 ROCm-matched docker image.
#
#   cd ROCmValidationSuite
#   ./.github/docker/rvs-nightly-rocm-ubuntu22.04/setup-on-runner.sh --from-tarball amdrocm10-rvs-....tar.gz

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
chmod +x .github/docker/build-rocm-sdk-image.sh
exec .github/docker/build-rocm-sdk-image.sh --context .github/docker/rvs-nightly-rocm-ubuntu22.04 "$@"
