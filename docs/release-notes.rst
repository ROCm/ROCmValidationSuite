.. meta::
   :description: ROCm Validation Suite release notes
   :keywords: ROCm Validation Suite, RVS, release notes, ROCm

***********************************************
ROCm Validation Suite (RVS) 1.7.0 release notes
***********************************************

These release notes describe notable changes since the previous RVS release.

Added
-----

- Level-based test configurations (``-r`` 1–5) for Radeon platforms: nv21, nv31, nv32, gfx1200, gfx1201, RX9060, RX9070, RX9070GRE, and R9600D.
- PCI device ID auto-detection for Radeon GPUs in ``gpu_get_platform_name()``.
- Support for GEMM bandwidth.
- GEMM consistency check support.
- New read kernel in Babel for gfx1250.

Changed
-------

- Update fp32 and fp64 targets for MI300X, MI300X-HF and MI325X.

Removed
-------

- Deprecated modules **GM**, **SMQT**, and **PESM** (since Nov 2024). GPU listing (``-g`` / ``--listGpus``) no longer depends on PESM.
- Unused **PERF** module (``perf.so``).
- Residual **EDP** sources (``edp.so``; product path removed earlier). 
