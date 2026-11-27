/********************************************************************************
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "include/probe_registry.h"

#include "include/probes/driver_overhead_probe.h"
#include "include/probes/gemm_probe.h"
#include "include/probes/host_device_probe.h"
#include "include/probes/kernel_launch_probe.h"
#include "include/probes/membw_probe.h"
#include "include/probes/platform_checks_probe.h"
#include "include/probes/rocm_version_probe.h"

namespace gpim {

std::vector<std::string> ProbeRegistry::names() {
  return {
    "kernel_launch",
    "gemm",
    "memory_bandwidth",
    "host_device_bandwidth",
    "runtime_api_latency",
    "platform_checks",
    "rocm_version"
  };
}

std::unique_ptr<ProbeBase> ProbeRegistry::create(const std::string& name) {
  if (name == "kernel_launch") return std::unique_ptr<ProbeBase>(new KernelLaunchProbe());
  if (name == "gemm") return std::unique_ptr<ProbeBase>(new GEMMProbe());
  if (name == "memory_bandwidth") return std::unique_ptr<ProbeBase>(new MemBandwidthProbe());
  if (name == "host_device_bandwidth") return std::unique_ptr<ProbeBase>(new HostDeviceProbe());
  if (name == "runtime_api_latency") return std::unique_ptr<ProbeBase>(new DriverOverheadProbe());
  if (name == "platform_checks") return std::unique_ptr<ProbeBase>(new PlatformChecksProbe());
  if (name == "rocm_version") return std::unique_ptr<ProbeBase>(new ROCmVersionProbe());
  return std::unique_ptr<ProbeBase>();
}

}  // namespace gpim
