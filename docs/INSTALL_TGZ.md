# Installing RVS from the Linux TGZ (`.tar.gz`) archive

This page describes how to **install and run** the ROCm Validation Suite (RVS) when you have the **relocatable tar.gz** produced by the project’s packaging (e.g. `amdrocm7-rvs-1.3.15-r0711.20260423-Linux.tar.gz`).

- **Build pipeline, file naming, and S3 download paths** are documented in [README_BUILD_PACKAGES (GitHub workflow)](../.github/workflows/README_BUILD_PACKAGES.md#any-linux-distribution-tgz---relocatable).
- The **same install steps** are mirrored here so the main [README](../README.md) can link to a single, copy-paste-friendly user-facing guide.

---

## Pre-install: ROCm

Before you install the RVS TGZ, **ROCm must be installed, configured, and on your `PATH` / `LD_LIBRARY_PATH` as in AMD’s documentation** so HIP, HSA, LLVM, and other ROCm libraries are discoverable. Install a stack that includes **ROCm’s LLVM** (for example the **`rocm-llvm`** package, or an equivalent meta-package from your ROCm repo). Follow **rocm docs**:

- **ROCm documentation (start here)**: <https://rocm.docs.amd.com/>
- **Linux install / deployment (paths, env, post-install)**: <https://rocm.docs.amd.com/en/latest/deploy/linux/index.html>

**Assumptions (TGZ use):** ROCm is set up on the machine, `ROCM_PATH` (or your install prefix) is correct, and the runtime can load ROCm libraries. The RVS build records **`RPATH`** entries with **`$ORIGIN`-relative** paths for the extras install, plus **`/opt/rocm/lib`**, **`/opt/rocm/lib/llvm/lib`** (older ROCm layouts), **`/opt/rocm/core-<ROCmMajor>/lib`**, and **`/opt/rocm/core-<ROCmMajor>/lib/llvm/lib`** for the ROCm stack and LLVM (including OpenMP from ROCm’s LLVM). The TGZ only ships RVS; it does not replace a full ROCm stack.

---

## Install RVS run-time dependencies (on the target system)

Install **PCI** access for GPU enumeration where your distro does not already provide it:

| Family | Typical package |
|--------|-----------------|
| **Debian / Ubuntu** | `libpci3` (or `libpci-3-0-0` on some releases) |
| **RHEL / Rocky / Alma** | `pciutils-libs` |
| **SUSE / openSUSE** | `libpci3` / `pciutils` as appropriate for the release |

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y libpci3

# RHEL / Rocky / Alma 8+
sudo dnf install -y pciutils-libs

# openSUSE / SUSE
sudo zypper install libpci3
```

The **DEB/RPM** packages from this project declare a dependency on **`rocm-llvm`** (ROCm LLVM, including the OpenMP runtime used at link time). Prefer installing RVS via those packages when possible so dependencies resolve automatically.

---

## Extract the TGZ

```bash
# Example for ROCm major 7: match the extras path to your layout and the tarball name.
sudo mkdir -p /opt/rocm/extras-7
sudo tar -xzf amdrocm7-rvs-*.tar.gz -C /opt/rocm/extras-7
```

---

## Post-install: `PATH` and `LD_LIBRARY_PATH`

Point the shell at the extracted RVS prefix and ROCm (adjust paths to match your install):

```bash
export ROCM_PATH=/opt/rocm              # or your real ROCm root, per rocm docs
export PATH=/opt/rocm/extras-7/bin:$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/extras-7/lib:$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:${LD_LIBRARY_PATH}
```

**`rvs`** embeds **`RPATH`** for the ROCm stack and LLVM (see **Pre-install: ROCm**), including **`/opt/rocm/lib/llvm/lib`** where applicable, so it usually does not need **`LD_LIBRARY_PATH`** for them. The export still adds **`$ROCM_PATH/lib/llvm/lib`** so your environment matches a typical ROCm tree and works for other tools; if LLVM on your system is only under **`$ROCM_PATH/core-<major>/lib/llvm/lib`**, include that path instead or as well.

---

## Run RVS

```bash
rvs --help
```

(You can also add `/opt/rocm/extras-7/bin` to your user profile or system `PATH` so `rvs` is on the default path in new shells.)

---

## See also

- [CI / packaging: README_BUILD_PACKAGES.md](../.github/workflows/README_BUILD_PACKAGES.md) — building packages, S3, DEB/RPM, and the full **Installing generated packages** section
- [User guide: module configuration and examples](ug1main.md)
