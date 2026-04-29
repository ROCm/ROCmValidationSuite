# Installing RVS from the Linux TGZ (`.tar.gz`) archive

This page describes how to **install and run** the ROCm Validation Suite (RVS) when you have the **relocatable tar.gz** produced by the project’s packaging (e.g. `amdrocm7-rvs-1.3.15-r0711.20260423-Linux.tar.gz`).

- **Build pipeline, file naming, and S3 download paths** are documented in [README_BUILD_PACKAGES (GitHub workflow)](../.github/workflows/README_BUILD_PACKAGES.md#any-linux-distribution-tgz---relocatable).
- The **same install steps** are mirrored here so the main [README](../README.md) can link to a single, copy-paste-friendly user-facing guide.

---

## Pre-install: ROCm

Before you install the RVS TGZ, **ROCm must be installed, configured, and on your `PATH` / `LD_LIBRARY_PATH` as in AMD’s documentation** so HIP, HSA, and other ROCm libraries are discoverable. Follow the current Linux install guide in **rocm docs**:

- **ROCm documentation (start here)**: <https://rocm.docs.amd.com/>
- **Linux install / deployment (paths, env, post-install)**: <https://rocm.docs.amd.com/en/latest/deploy/linux/index.html>

**Assumptions (TGZ use):** ROCm is set up on the machine, `ROCM_PATH` (or your install prefix) is correct, and the runtime can load ROCm/LLVM libraries. The TGZ only ships RVS; it does not replace a full ROCm stack.

---

## Install RVS run-time dependencies (on the target system)

The archive is built against ROCm, but the host still needs a few system libraries. Install at least **PCI (libpci)** and **LLVM OpenMP** (**`libomp.so`**).

| Family | PCI | LLVM OpenMP (`libomp.so`) |
|--------|-----|----------------------------|
| **Debian / Ubuntu** | `libpci3` (or `libpci-3-0-0`) | **`libomp5`** on many releases (runtime SONAME may vary — use `apt search '^libomp'`). |
| **RHEL / Rocky / Alma 8+** | `pciutils-libs` | **`libomp`** (LLVM OpenMP for Clang, **AppStream**). |
| **openSUSE / SUSE** | `libpci3`, `pciutils` as needed | **Versioned LLVM packages**, e.g. **`libomp18`**, **`libomp17`** on Tumbleweed — names track the LLVM stack. Use **`zypper search -s libomp`** and install the **runtime** package (not only `-devel`) that matches your distro. |

Examples (adjust names to your release):

```bash
# Ubuntu / Debian — LLVM OpenMP runtime
sudo apt update
sudo apt install -y libpci3 libomp5

# RHEL / Rocky / Alma 8+ — libomp is the standard LLVM OpenMP RPM (AppStream)
sudo dnf install -y pciutils-libs libomp

# openSUSE Tumbleweed (example; Leap/SLE may use different version suffixes)
sudo zypper install libpci3 libomp18
# zypper search -s libomp    # list alternatives
```

**Note:** If **`libomp.so`** is still missing, it is **usually** provided with **ROCm** under **`$ROCM_PATH/lib/llvm/lib`** — the environment block below adds that path first.

---

## Extract the TGZ

```bash
# Example for ROCm major 7: match the extras path to your layout and the tarball name.
sudo mkdir -p /opt/rocm/extras-7
sudo tar -xzf amdrocm7-rvs-*.tar.gz -C /opt/rocm/extras-7
```

---

## Post-install: `PATH` and `LD_LIBRARY_PATH`

Point the shell at the extracted RVS prefix and the ROCm you installed (replace paths with your real `ROCM_PATH` and extras major version). **Copy and paste the block as one unit:**

```bash
export ROCM_PATH=/opt/rocm              # or your real ROCm root, per rocm docs
export PATH=/opt/rocm/extras-7/bin:$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/extras-7/lib:\
$ROCM_PATH/lib:\
$ROCM_PATH/lib/llvm/lib:\
$LD_LIBRARY_PATH
```

**`libomp` (LLVM OpenMP):** `$ROCM_PATH/lib/llvm/lib` is already in the `export` block above — **`libomp.so` is often there** (ROCm’s LLVM). If your layout differs, run `find $ROCM_PATH -name 'libomp.so' 2>/dev/null` and add that directory to `LD_LIBRARY_PATH`, or install the host **`libomp`** packages from the table above.

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
