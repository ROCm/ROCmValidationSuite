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

When the CLI is built, it is installed next to `rvs` under the same prefix and
uses the **same relocatable RUNPATH** as RVS (`$ORIGIN`, `/opt/rocm/core-<N>/lib`, etc.)
via [`CMakeTransferBenchRPATH.cmake.in`](../CMakeTransferBenchRPATH.cmake.in) (initial
cache for the sub-build; `$ORIGIN` must use bracket literals so CMake does not expand it
to empty).

The set of GPU architectures the CLI is compiled for can be narrowed with:

```
cmake -DTRANSFERBENCH_GPU_TARGETS="gfx90a;gfx942;gfx950" ...
```

By default RVS builds the CLI for the same arch set RVS itself ships on.

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

## Further reading

- [TransferBench project](https://github.com/ROCm/TransferBench)
- [RVS User guide — `pebb` and `pbqt` modules](ug1main.md)
