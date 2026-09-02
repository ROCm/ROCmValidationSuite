#!/usr/bin/env bash
# Run on the self-hosted GPU runner host to build the RHEL 8 ROCm+RVS docker image.
#
#   cd ROCmValidationSuite
#   ./.github/docker/rvs-nightly-rocm-rhel8/setup-on-runner.sh --channel nightly

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
chmod +x .github/docker/rvs-nightly-rocm-rhel8/build-rocm-image.sh
exec .github/docker/rvs-nightly-rocm-rhel8/build-rocm-image.sh "$@"
