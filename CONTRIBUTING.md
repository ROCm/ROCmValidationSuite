# Contributing to the ROCm Validation Suite

We are enthusiastic about contributions to our code and documentation. Please
feel free to file issues where documentation or functionality is lacking or,
even better, volunteer to contribute to help close these gaps!

## Developer policies

These policies apply to all forms of activity and engagement in this project.

> [!IMPORTANT]
> AMD employees must also follow the ROCm open source software
> contributing policies at http://u.amd.com/rocm-oss-policies.

### Licensing

Code contributions to this project are covered under the terms of the
[LICENSE](LICENSE) file.

### Communication channels

The ROCm Validation Suite uses **GitHub** as the source of truth for all issue
tracking, project planning, and code contributions. This makes it easy for
developers across companies to observe the project and contribute.

### AI tool use policy

We allow the use of AI tools to help author issues, pull requests, reviews,
comments, and more.

While we don't have a formal AI tool use policy in the ROCm GitHub organization
at the moment, we are actively monitoring contribution patterns and take
inspiration from policies in neighboring ecosystem projects:

- https://llvm.org/docs/AIToolPolicy.html
- https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md

Of particular note, from the LLVM AI Tool Use Policy:

> The contributor is always the author and is fully accountable for their
> contributions. Contributors should be sufficiently confident that the
> contribution is high enough quality that asking for a review is a good use of
> scarce maintainer time, and they should be able to answer questions about
> their work during review.

When posting significant portions of AI-generated content on issues or pull
requests, we also encourage contributors to clearly disclose which sections
of a comment are human or AI-generated (e.g. using a code or quote block, or
by posting the output to https://gist.github.com/) along with accompanying
human commentary explaining the relevance and accuracy of the content.

### Branch creation policy

Most contributions from AMD employees should be made via branches in the shared
repository and not personal forks according to the
http://u.amd.com/rocm-oss-contributing policy. Branches in the shared repository
benefit from code security scanning, code quality tooling, and easier
collaboration.

For external contributors, pull requests from forks are of course accepted and
most workflows are compatible with this contribution model.

> [!NOTE]
> GitHub Actions workflows that use self-hosted runners, including the nightly
> and pull-request test workflows, can only be triggered with
> [workflow_dispatch](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
> from branches in the shared repository. If a change needs that testing, a
> branch in the shared repository may need to be created.

### Branch naming policy

Branches in personal forks can use any name.

Branches created in the shared repository should match one of these patterns so
branches are easily sortable and can be audited by repository maintainers:

| Branch name pattern                      | Example                          |
| ---------------------------------------- | -------------------------------- |
| `users/[USERNAME]/[feature-or-bug-name]` | `users/cooldeveloper/my-feature` |
| `shared/[feature-or-bug-name]`           | `shared/rvs-config-update`       |

A few exceptions are granted for automation:

- `bump-*`
- `revert-*`
- `dependabot/**/*`

Long-lived branches use other patterns:

- [`master`](https://github.com/ROCm/ROCmValidationSuite/tree/master)
- [`release/*`](https://github.com/ROCm/ROCmValidationSuite/branches/all?query=release%2F)

## Development workflows and contributing guide

### Finding an issue to work on

If you want to contribute but don't have a specific problem in mind, the
[issue tracker](https://github.com/ROCm/ROCmValidationSuite/issues) is the best
place to start. Open issues that are understood well enough for someone outside
the core team to pick up are the right place to begin. Smaller issues are a
good first change.

> [!TIP]
> Don't have the hardware to reproduce an issue? A precise reproduction, a
> narrowed-down regression range, or better diagnostics are valuable
> contributions on their own.

If an issue is missing information you need, say so in the thread and a
maintainer will take another look.

#### Do I need to ask before working on an issue?

No, but please comment on the issue to say you are picking it up so others don't
duplicate your effort. We don't lock issues to one person, and an existing
assignee doesn't stop you from opening a pull request. If multiple pull requests
address the same issue, we take the highest quality one (or the first one, if
they are comparable).

> [!NOTE]
> Some issues filed here are bugs in `external/TransferBench`. That directory is
> a git submodule, and the fix belongs in
> [ROCm/TransferBench](https://github.com/ROCm/TransferBench) instead.

When your change is ready, follow
[Creating pull requests](#contributing-via-github-pull-requests).

### Using GitHub Issues for bug reporting

Before filing a new issue, please search through
[existing issues](https://github.com/ROCm/ROCmValidationSuite/issues) to make
sure your issue hasn't already been reported.

General issue guidelines:

- Use your best judgement for new issue creation. If you find a similar open (or
  closed) issue already reported, upvote the issue and leave a comment with
  any new details, such as how you reproduced it.
- When filing an issue, be sure to provide as much information as possible,
  including reproduction steps, the ROCm and RVS versions, the GPU, and
  complete command output so we can triage efficiently.
- Check your issue regularly, as we may require additional information to
  resolve the issue.
- Do not file security vulnerabilities as public issues. Report those through
  the process in [SECURITY.md](SECURITY.md).

### Using GitHub Issues for feature development

Discussion about new features is welcome via a
[GitHub issue](https://github.com/ROCm/ROCmValidationSuite/issues) or
[on Discord](https://discord.com/invite/amd-dev).

> [!TIP]
> When planning complex changes, please solicit feedback and announce your
> intent to work on a pull request early in development, as this gives other
> contributors time to offer advice and avoid duplicating effort.

### Contributing via GitHub pull requests

To keep code quality high across the project, we hold pull requests to the
following standards:

| Check description         | Enforced via                                                                                  | Details                                              |
| ------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Branch naming patterns    | Repository maintainers                                                                        | [Branch naming policy](#branch-naming-policy)        |
| Changes should build      | [`.github/workflows/build-relocatable-packages.yml`](.github/workflows/build-relocatable-packages.yml) | [Building from source](README.md#building-from-source) |
| Security scans            | [`.github/workflows/pr-security-scan.yml`](.github/workflows/pr-security-scan.yml)           | [Security scanners](#security-scanners)              |

Build instructions are in [README.md](README.md).

#### Linking pull requests to GitHub issues

All pull requests should be associated with at least one GitHub issue.

This lets reviewers see the context for contributions and helps link bugs and
their fixes together. See also the
[Using GitHub Issues for bug reporting](#using-github-issues-for-bug-reporting)
and
[Using GitHub Issues for feature development](#using-github-issues-for-feature-development)
sections above.

Reference the issue in the pull request description, for example:

```markdown
Fixes https://github.com/ROCm/ROCmValidationSuite/issues/123
```

Contributions by AMD employees may also reference a JIRA ID, though GitHub
issues are preferred for open source development.

#### Security scanners

Separately from the build, we scan the repository for secrets, unsafe Python,
workflow vulnerabilities, and image misconfigurations. These run in CI via
[`.github/workflows/pr-security-scan.yml`](.github/workflows/pr-security-scan.yml),
which calls the shared
[`ROCm/rocm-security-gh`](https://github.com/ROCm/rocm-security-gh) reusable
workflow. See [SECURITY.md](SECURITY.md) for how the pull-request and weekly
scans fit together.

Each scanner is runnable locally against the same configuration CI uses, which
is faster than pushing a commit to see what CI says. The configurations live in
[`.github/scan_tools_configs/`](.github/scan_tools_configs/):

```bash
# Secrets, over the full git history (installed separately, see gitleaks docs).
gitleaks detect --source . --config .github/scan_tools_configs/gitleaks.toml \
  --redact --verbose --no-banner

# Secrets, working tree only. Much faster, and usually what you want locally.
gitleaks detect --source . --config .github/scan_tools_configs/gitleaks.toml \
  --redact --no-banner --no-git

# Unsafe patterns in Python (pip install bandit).
bandit --recursive . --configfile .github/scan_tools_configs/bandit.yml

# GitHub Actions workflow vulnerabilities (pip install zizmor).
zizmor --persona regular --config .github/scan_tools_configs/zizmor.yml .

# Dockerfile misconfigurations and dependency vulnerabilities (see trivy docs).
trivy fs --config .github/scan_tools_configs/trivy.yml .
```

> [!NOTE]
> Pull-request CI fails on high-severity findings. A local run can report
> lower severities as well.
>
> The commands also scan the whole repository, while pull request runs default
> to scanning only what the pull request changed. A full-history `gitleaks` run
> in particular reports pre-existing findings that the pull request check does
> not.

CodeQL is not in the list above because it needs the CodeQL CLI and a built
database. It runs in CI through the shared security workflow.

#### Using draft pull requests

If you are not looking for a review on a pull request yet, please mark that pull
request as a draft:

- GitHub Docs: [Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
- GitHub Docs: [Changing the stage of a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request)

> [!WARNING]
> Reviews from CODEOWNERS are automatically requested for non-draft pull
> requests. See
> [About code owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners).
>
> If a pull request modifies files covered under
> [`.github/CODEOWNERS`](.github/CODEOWNERS), maintainers will be asked to
> review the changes. If a pull request is only intended for testing, mark it
> as a draft to limit review request notifications.

#### Requesting a code review

When you are ready for a review, mark the pull request as "ready for review"
if it was a draft, then request a review from a reviewer or maintainer. See
[Requesting a pull request review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review).

You can check the git history to see who recently authored or approved pull
requests in the same files or folders:

- [`.github/CODEOWNERS`](.github/CODEOWNERS)
- GitHub Docs: [About code owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- GitHub Docs: [Viewing and understanding files](https://docs.github.com/en/repositories/working-with-files/using-files/viewing-and-understanding-files)

#### Responding to review feedback and re-requesting review

After addressing review feedback, please
[re-request review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review#requesting-reviews-from-collaborators-and-organization-members)
so the pull request shows up again for reviewers.

#### Merging approved changes

Once a pull request has been approved, pull request authors with write access
can merge their own changes, typically using the "squash and merge" strategy
(see
[Pull request merges](https://docs.github.com/en/pull-requests/reference/pull-request-merges))
with the pull request description as the commit message. Contributors without
write access can request that a maintainer merge on their behalf.
