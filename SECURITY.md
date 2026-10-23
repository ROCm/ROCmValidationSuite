# Security Policy

## Reporting a vulnerability

If you have discovered a security vulnerability in the ROCm Validation Suite,
do NOT report it publicly using a GitHub issue. Instead, report the issue
through the [AMD Product Security website](https://www.amd.com/en/resources/product-security.html).

## Automated security scanning

Alongside the reporting path above, this repository is scanned automatically.
The scanners are not implemented here: the ROCm Validation Suite calls the shared
[`ROCm/rocm-security-gh`](https://github.com/ROCm/rocm-security-gh)
`security-baseline.yml` reusable workflow, so the scanner versions and
behavior are maintained centrally for ROCm, and this repository supplies only
its own configuration under
[`.github/scan_tools_configs/`](.github/scan_tools_configs/).

| Scanner                                          | Looks for                                                   |
| ------------------------------------------------ | ----------------------------------------------------------- |
| [gitleaks](https://github.com/gitleaks/gitleaks) | Secrets and credentials in tracked files and git history    |
| [bandit](https://bandit.readthedocs.io/)         | Unsafe patterns in our Python scripts                       |
| [zizmor](https://docs.zizmor.sh/)                | GitHub Actions workflow vulnerabilities                     |
| [trivy](https://trivy.dev/)                      | Dockerfile misconfigurations and dependency vulnerabilities |
| [CodeQL](https://codeql.github.com/)             | Semantic code analysis of our C/C++ and Python              |

Two workflows run them, and where a finding shows up depends on which one
produced it:

- [`.github/workflows/pr-security-scan.yml`](.github/workflows/pr-security-scan.yml)
  runs on every pull request, scoped to what that pull request changed. It
  reports in the job summary and a build artifact, and deliberately does not
  upload to the Security tab, so pull requests from forks behave the same as
  those from branches in this repository.
- [`.github/workflows/weekly-security-scan.yml`](.github/workflows/weekly-security-scan.yml)
  runs on a schedule across the whole repository and uploads SARIF to this
  repository's Security tab, which is the authoritative view of the current
  state.

Only the ROCm Validation Suite tree is in scope. `external/TransferBench` is a
git submodule and is scanned in its own repository.

> [!IMPORTANT]
> A finding from these scanners is not a vulnerability report. If a scanner
> finding turns out to be an exploitable vulnerability in shipped ROCm
> software, report it through the AMD Product Security website above rather
> than in a public issue or pull request.

Each file under [`.github/scan_tools_configs/`](.github/scan_tools_configs/)
includes a local-scan command that uses the same configuration CI uses.
