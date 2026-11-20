# Signed publish to the public repos

This document describes how a scheduled RVS build and a release-candidate promotion become a public extras-rvs repository behind `nightly.repo.amd.com`, `rc.repo.amd.com`, and stable-staging.

Two existing workflows prepare the RVS bucket. A third workflow, [publish-signed-public.yml](./publish-signed-public.yml), copies a complete signed set to the public buckets. The signing of rpm and deb packages happens in a separate job that is not in this repository.

## Flow

```mermaid
flowchart LR
  nightlyJob[Scheduled nightly build] --> unsignedPkg[nightly/unsigned packages]
  nightlyJob --> signedTar[nightly/signed tarball and latest.json]
  rcJob[RC promotion] --> releaseUnsigned[release/unsigned packages]
  rcJob --> releaseSignedTar[release/signed tarball and latest.json]
  signer[External signing job] --> signedPkg[nightly/signed and release/signed packages]
  signedTar --> publisher[Every 2 hours]
  releaseSignedTar --> publisher
  signedPkg --> publisher
  publisher --> publicNightly[nightly.repo.amd.com]
  publisher --> publicRc[rc.repo.amd.com]
  publicRc --> stableStaging[Manual build number to stable-staging]
```

## What each workflow writes

The RVS bucket is `vars.AWS_S3_BUCKET`. These jobs assume `secrets.AWS_ROLE_ARN`. They accumulate objects and do not delete anything.

| Step | Workflow | Writes immediately | Left for the signing job |
|------|----------|--------------------|--------------------------|
| Scheduled default-branch nightly | [build-relocatable-packages.yml](./build-relocatable-packages.yml) | `nightly/unsigned/packages/deb/`, `nightly/unsigned/packages/rpm/x86_64/` and their repo metadata; `nightly/unsigned/tarball/` and `nightly/signed/tarball/` (`.tar.gz` and `.sha256`); `nightly/unsigned/latest.json` and `nightly/signed/latest.json` | rpm/deb bytes and repo metadata under `nightly/signed/packages/` |
| Manual RC promotion | [unsigned-release-candidate-promotion.yml](./unsigned-release-candidate-promotion.yml) | `release/unsigned/packages/` and `release/unsigned/tarball/`; the same tarball and sidecar on `release/signed/tarball/`; `release/unsigned/latest.json` and `release/signed/latest.json` | rpm/deb bytes and repo metadata under `release/signed/packages/` |

`nightly/unsigned/latest.json` and `release/unsigned/latest.json` stay the signing job's input. The signed `latest.json` files are the public publisher's checklist. Their tarball keys already point at `*/signed/tarball/`. Their deb and rpm keys point at `*/signed/packages/`, which are empty until the signing job copies them. Their `metadata` list names the full signed repo metadata the signing job must provide under `*/signed/packages/`:

- deb (`deb/dists/stable/`): `Release`, `InRelease`, `Release.gpg`, `main/binary-amd64/Packages`, `main/binary-amd64/Packages.gz`
- rpm (`rpm/x86_64/repodata/`): `repomd.xml`, `repomd.xml.asc`, `primary.xml.gz`, `filelists.xml.gz`, `other.xml.gz`

The signature files (`InRelease`, `Release.gpg`, `repomd.xml.asc`) are part of the gate, so an unsigned repo is never published.

## Public publish

[publish-signed-public.yml](./publish-signed-public.yml) runs on the cron `0 */2 * * *`. It can also be started manually with `run_number` left empty.

For `nightly/signed` and `release/signed` it:

1. Assumes `secrets.AWS_ROLE_ARN` and downloads that prefix's `latest.json` from the RVS bucket.
2. Checks that every `s3_key`, sha256 sidecar, package key, and metadata key in the file exists.
3. Skips that stream with a warning if `latest.json` or any listed object is missing. It does not publish a partial set. The other stream still publishes when it is complete.
4. Assumes the public role and copies each verified object to the public bucket. `latest.json` itself is not uploaded.
5. Checks that each public object exists after the copy. A missing public object fails that stream.
6. Syncs the complete signed `packages/deb/dists/` and `packages/rpm/x86_64/repodata/` directories to the public bucket (no `--delete`), once the gate has passed. Any other file in those directories, such as `by-hash` entries or other components, is copied too, so every file referenced by `Release` and `repomd.xml` is public. The rest of `packages/` is not mirrored.

Public object keys keep the path under the component prefix. The `v5` prefix is part of the S3 key and is omitted from the served URL.

| Stream | RVS prefix | Public bucket | Served as |
|--------|------------|---------------|-----------|
| `nightly` | `nightly/signed/` | `therock-repo-amd-nightly-extras-rvs` | `https://nightly.repo.amd.com/rocm/extras/rvs/...` |
| `rc` | `release/signed/` | `therock-repo-amd-rc-extras-rvs` | `https://rc.repo.amd.com/rocm/extras/rvs/...` |

Example:

```
s3://<rvs-bucket>/nightly/signed/tarball/<file>.tar.gz
s3://therock-repo-amd-nightly-extras-rvs/v5/rocm/extras/rvs/tarball/<file>.tar.gz
https://nightly.repo.amd.com/rocm/extras/rvs/tarball/<file>.tar.gz
```

`packages/deb/` and `packages/rpm/` keep the same repository layout they have on the RVS bucket (`pool/` and `dists/` for deb, `x86_64/` and `repodata/` for rpm).

## Manual copy to stable-staging

Dispatch [publish-signed-public.yml](./publish-signed-public.yml) with `run_number` set to the build number embedded in the package filenames, the same number used by the RC promotion workflow.

The job reads the public RC bucket and writes `therock-repo-amd-stable-staging-extras-rvs`. It copies the one matching `.deb`, `.rpm`, `.tar.gz`, and `.tar.gz.sha256`, plus the current `packages/deb/dists` and `packages/rpm` repodata so the staging repo stays installable. The stable-staging DNS name is not the publish target; DevOps copies stable-staging to stable at GA without changing the layout.

## Credentials

| Secret | Used for |
|--------|----------|
| `AWS_ROLE_ARN` | Read and write the RVS bucket (`nightly/signed/`, `release/signed/`, and the unsigned prefixes). |
| `AWS_PUB_ARN` | Assume the role that writes the public buckets. This is the only public role ARN. |

`AWS_PUB_ARN` is a full role ARN, not the account number alone, and it must contain the literal token `<stream>`. Before `assume-role`, the job replaces that token with `nightly`, `rc`, or `stable-staging`. One secret therefore covers all three public writes. If the secret is missing or has no `<stream>` token, the public job fails and does not guess a role. The workflow and this document do not contain the account id or the role name.

No long-lived AWS keys are used. The jobs request `id-token: write` and assume a role with the GitHub OIDC token. Public copies accumulate objects and do not delete them.
