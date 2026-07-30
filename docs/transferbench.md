# TransferBench in RVS

[TransferBench](https://github.com/ROCm/TransferBench) is a low-level utility
for measuring host-to-device, device-to-device, and NIC transfer performance on
AMD platforms. It is vendored into RVS as a git submodule at
`external/TransferBench` and built as part of the RVS build.

## What ships in the RVS package

Building RVS produces a single DEB or RPM that contains:

- The `rvs` launcher and all RVS test modules, including `pebb.so` and
  `pbqt.so`, which use the TransferBench headers internally as their transfer
  backend.
- The standalone `TransferBench` CLI binary, installed alongside `rvs` under
  the same prefix (for example `/opt/rocm/extras-<major>/bin/TransferBench`).

No separate TransferBench package needs to be installed.

## Why the CLI is bundled

The `TransferBench` CLI is bundled **for compatibility**. Existing tooling,
scripts, and CI pipelines that invoke `TransferBench` directly continue to
work after installing RVS, without requiring a second package.

For new work, the recommended entry points are:

- **RVS** — use the `pebb` and `pbqt` modules with a YAML config to drive
  TransferBench through RVS's standard CLI, result schema, and logging.
  See the [User guide](ug1main.md) for the supported `transfer_method`,
  `transferbench_test`, `executor`, and related options.
- **The TransferBench API** — headers live under
  `external/TransferBench/src/header`. Link against the headers from your
  own application when you need programmatic access to the transfer engine.

The standalone CLI should be reserved for reproducing legacy benchmark
invocations or for ad-hoc one-off measurements outside an RVS run.

## Build options

The CLI is **off by default** in `build_packages_local.sh` and in CMake (`BUILD_TRANSFERBENCH_CLI=OFF`). CI enables it unless overridden. To build it locally:

```
BUILD_TRANSFERBENCH_CLI=ON ./build_packages_local.sh
# or
cmake -DBUILD_TRANSFERBENCH_CLI=ON ...
```

Skipping the CLI does not affect RVS's `pebb`/`pbqt` modules — they consume
the TransferBench *headers* from the submodule, not the CLI binary.

When the CLI is built, it is installed next to `rvs` under the same prefix. The
`TransferBench` binary links `libnuma` and ROCm libraries; package metadata
requires **`libnuma1`** (Debian/Ubuntu) or **`numactl-libs`** (RHEL-family RPM).
RUNPATH entries point at the ROCm stack (`/opt/rocm/lib`, `/opt/rocm/core-<N>/lib`).
See [`CMakeTransferBenchRPATH.cmake.in`](../CMakeTransferBenchRPATH.cmake.in).

### GPU targets vs SDK tarball family

`GPU_FAMILY` (for example `gfx110X-all`, `gfx1151`) selects which **ROCm SDK
tarball** `build_packages_local.sh` downloads. It does **not** control which GPU
architectures the `TransferBench` binary is compiled for.

Offload architectures are set by `GPU_TARGETS` / `TRANSFERBENCH_GPU_TARGETS`.
By default RVS uses the same list as upstream TransferBench packaging
(`external/TransferBench/build_packages_local.sh`):

```
gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201
```

Override when building:

```
GPU_TARGETS="gfx90a;gfx942" BUILD_TRANSFERBENCH_CLI=ON ./build_packages_local.sh
# or
cmake -DBUILD_TRANSFERBENCH_CLI=ON -DTRANSFERBENCH_GPU_TARGETS="gfx90a;gfx942" ...
```

The sub-build also passes `-DHIP_PLATFORM=amd` and disables optional TransferBench
features (NIC executor, MPI, DMA-BUF, local-GPU-only mode) to match upstream
relocatable packaging and keep the bundled CLI dependency-light.

`GPU_TARGETS` is injected via the ExternalProject initial cache (`-C` file), not
`-DGPU_TARGETS=...` on the command line, because CMake `list(APPEND)` splits
semicolon-separated values into separate list entries.

CMake logs forwarded args at parent configure time. In **GitHub Actions**, the
TransferBench sub-build runs `cmake --build ... --verbose` with `LOG_BUILD=OFF` so
compile/link lines appear in the workflow log. Local builds keep stamp logs under
`build/TransferBenchCLI-prefix/src/TransferBenchCLI-stamp/`.

## Submodule layout

```
ROCmValidationSuite/
  external/
    TransferBench/        # git submodule, pinned in the RVS tree
      src/
        client/Client.cpp     # CLI entry point
        header/               # public headers consumed by pebb/pbqt
```

After cloning RVS, initialise the submodule if you did not pass
`--recurse-submodules`:

```
git submodule update --init --recursive
```

## Version pinning

The TransferBench submodule is pinned to a specific commit in the RVS tree,
so a given RVS tag always bundles a known-good TransferBench revision. To
move to a newer TransferBench, update the submodule pointer in a normal pull
request against RVS.

TransferBench v1.69+ adds `third-party/ibverbs/` headers (`IbvDynLoad.hpp`).
RVS auto-detects that directory at configure time and adds it to the PEBB/PBQT
include path (`TRANSFERBENCH_IBVERBS_INC_DIR` in the root `CMakeLists.txt`).
When the folder is absent (older submodule pins), no extra includes or `libdl`
link are applied. Runtime `libibverbs` is loaded dynamically only if present;
PEBB/PBQT gfx/dma tests do not require it.

## Further reading

- [TransferBench project](https://github.com/ROCm/TransferBench)
- [RVS User guide — `pebb` and `pbqt` modules](ug1main.md)
