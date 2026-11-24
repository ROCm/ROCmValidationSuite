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
2. Validates the manifest before anything else. It must name at least one deb package, the rpm, the tarball and its sha256 sidecar. An empty or partial `latest.json` fails the stream, even if the metadata files exist. Every key must be normalized and sit under `<stream>/signed/packages/` or `<stream>/signed/tarball/`. A key with `..`, `.` or empty path segments, a backslash, a control character or a leading `/` (for example `nightly/signed/packages/../runs/latest.json`) fails the stream before anything is staged or copied. The same strict check applies to package paths read from the signed index and to keys listed from S3. Only then are the required deb and rpm metadata and signature files added to the list.
3. Maps every key that points at `<stream>/unsigned/` to the same path under `<stream>/signed/`. The manifest may list the unsigned copy of a package (for example `nightly/unsigned/packages/deb/pool/main/x.deb`), but the publisher always requires and copies the signed one (`nightly/signed/packages/deb/pool/main/x.deb`). `runs/` and `latest.json` are never published.
4. Checks that every mapped signed object (package, rpm, tarball, sha256 sidecar, metadata) exists in the RVS bucket. It skips that stream with a warning if `latest.json` or any signed object is missing, which means the signing job has not run yet. It does not publish a partial set. The other stream still publishes when it is complete.
5. Checks that the public repository matches the signed one, by file name only (no content comparison). The stream is up to date, and is skipped with the log line `No new packages or index changes for <stream>`, only when all four hold:
   1. every manifest object (deb, rpm, tarball, sidecar) exists publicly;
   2. every file in the signed `packages/deb/dists/` and `packages/rpm/x86_64/repodata/` directories (listed from S3, so `by-hash` and any other component files are included) exists publicly under the same path;
   3. the package set referenced by the public `Packages` and `primary.xml.gz` equals the package set referenced by the signed ones;
   4. every package referenced by the signed `Packages` and `primary.xml.gz`, including older builds that are not in this `latest.json`, exists publicly.

   Skipping is not an error: nightly and release builds run less often than this job. If a package referenced by the signed index is missing publicly, it must exist in the signed prefix, otherwise the job fails because the public repository cannot be made complete.
6. Otherwise copies only what is missing publicly: the manifest objects and referenced packages that are not public yet. Objects that are already public are not downloaded or copied again, so a metadata repair does not recopy packages. `latest.json` itself is not uploaded.
7. Uploads in an order that keeps the index valid for clients: first the packages, rpm, tarball and sidecar; then the complete signed `packages/deb/dists/` and `packages/rpm/x86_64/repodata/` directories (no `--delete`) except the top-level index and signature files; and last `Release`, `InRelease`, `Release.gpg`, `repomd.xml` and `repomd.xml.asc`.
8. Verifies by name that every copied object exists publicly and that every signed `dists/` and `repodata/` file is now in the public listing. A missing object fails that stream.

**Trust assumption.** The publisher treats the signed prefix as the latest complete set and assumes the signing job does not modify it while the publisher runs. It copies the signed files byte for byte with `aws s3 cp` and `aws s3 sync` and never edits, recompresses, regenerates or re-signs them. Only the signing job writes `<stream>/signed/packages/`. The publisher does not compare file content, so a re-sign that keeps the same file names and the same package set is not detected by the up-to-date check.

**Credentials.** Each stream runs in its own process. Role assumption clears any earlier credentials and every step is checked, so a failed role assumption ends the stream and never continues with the previous role. The public role needs `s3:ListBucket` on the public bucket for the listing checks.

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
