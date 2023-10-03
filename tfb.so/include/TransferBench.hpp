/*
Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <vector>
#include <sstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <set>
#include <unistd.h>
#include <map>
#include <iostream>
#include <sstream>

#include "Compatibility.hpp"

// Helper macro for catching HIP errors
#define HIP_CALL(cmd)                                                                   \
    do {                                                                                \
        hipError_t error = (cmd);                                                       \
        if (error != hipSuccess)                                                        \
        {                                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)          \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << "\n";   \
            exit(-1);                                                                   \
        }                                                                               \
    } while (0)

#include "EnvVars.hpp"

// Simple configuration parameters
size_t const DEFAULT_BYTES_PER_TRANSFER = (1<<26);  // Amount of data transferred per Transfer

// Different src/dst memory types supported
typedef enum
{
  MEM_CPU          = 0, // Coarse-grained pinned CPU memory
  MEM_GPU          = 1, // Coarse-grained global GPU memory
  MEM_CPU_FINE     = 2, // Fine-grained pinned CPU memory
  MEM_GPU_FINE     = 3, // Fine-grained global GPU memory
  MEM_CPU_UNPINNED = 4, // Unpinned CPU memory
  MEM_NULL         = 5, // NULL memory - used for empty
} MemType;

typedef enum
{
  EXE_CPU          = 0, // CPU executor              (subExecutor = CPU thread)
  EXE_GPU_GFX      = 1, // GPU kernel-based executor (subExecutor = threadblock/CU)
  EXE_GPU_DMA      = 2, // GPU SDMA-based executor   (subExecutor = streams)
} ExeType;

bool IsGpuType(MemType m) { return (m == MEM_GPU || m == MEM_GPU_FINE); }
bool IsCpuType(MemType m) { return (m == MEM_CPU || m == MEM_CPU_FINE || m == MEM_CPU_UNPINNED); };
bool IsGpuType(ExeType e) { return (e == EXE_GPU_GFX || e == EXE_GPU_DMA); };
bool IsCpuType(ExeType e) { return (e == EXE_CPU); };

char const MemTypeStr[7] = "CGBFUN";
char const ExeTypeStr[4] = "CGD";
char const ExeTypeName[3][4] = {"CPU", "GPU", "DMA"};

MemType inline CharToMemType(char const c)
{
  char const* val = strchr(MemTypeStr, toupper(c));
  if (*val) return (MemType)(val - MemTypeStr);
  printf("[ERROR] Unexpected memory type (%c)\n", c);
  exit(1);
}

ExeType inline CharToExeType(char const c)
{
  char const* val = strchr(ExeTypeStr, toupper(c));
  if (*val) return (ExeType)(val - ExeTypeStr);
  printf("[ERROR] Unexpected executor type (%c)\n", c);
  exit(1);
}

// Each Transfer performs reads from source memory location(s), sums them (if multiple sources are specified)
// then writes the summation to each of the specified destination memory location(s)
struct Transfer
{
  int                       transferIndex;      // Transfer identifier (within a Test)
  ExeType                   exeType;            // Transfer executor type
  int                       exeIndex;           // Executor index (NUMA node for CPU / device ID for GPU)
  int                       numSubExecs;        // Number of subExecutors to use for this Transfer
  size_t                    numBytes;           // # of bytes requested to Transfer (may be 0 to fallback to default)
  size_t                    numBytesActual;     // Actual number of bytes to copy
  double                    transferTime;       // Time taken in milliseconds

  int                       numSrcs;            // Number of sources
  std::vector<MemType>      srcType;            // Source memory types
  std::vector<int>          srcIndex;           // Source device indice
  std::vector<float*>       srcMem;             // Source memory

  int                       numDsts;            // Number of destinations
  std::vector<MemType>      dstType;            // Destination memory type
  std::vector<int>          dstIndex;           // Destination device index
  std::vector<float*>       dstMem;             // Destination memory

  std::vector<SubExecParam> subExecParam;       // Defines subarrays assigned to each threadblock
  SubExecParam*             subExecParamGpuPtr; // Pointer to GPU copy of subExecParam

  // Prepares src/dst subarray pointers for each SubExecutor
  void PrepareSubExecParams(EnvVars const& ev);

  // Prepare source arrays with input data
  bool PrepareSrc(EnvVars const& ev);

  // Validate that destination data contains expected results
  void ValidateDst(EnvVars const& ev);

  // Prepare reference buffers
  void PrepareReference(EnvVars const& ev, std::vector<float>& buffer, int bufferIdx);

  // String representation functions
  std::string SrcToStr() const;
  std::string DstToStr() const;
};

struct ExecutorInfo
{
  std::vector<Transfer*>   transfers;        // Transfers to execute
  size_t                   totalBytes;       // Total bytes this executor transfers
  int                      totalSubExecs;    // Total number of subExecutors to use

  // For GPU-Executors
  SubExecParam*            subExecParamGpu;  // GPU copy of subExecutor parameters
  std::vector<hipStream_t> streams;
  std::vector<hipEvent_t>  startEvents;
  std::vector<hipEvent_t>  stopEvents;

  // Results
  double totalTime;
};

typedef std::pair<ExeType, int> Executor;
typedef std::map<Executor, ExecutorInfo> TransferMap;

// Display usage instructions
void DisplayUsage(char const* cmdName);

// Display detected GPU topology / CPU numa nodes
void DisplayTopology(bool const outputToCsv);

// Build array of test sizes based on sampling factor
void PopulateTestSizes(size_t const numBytesPerTransfer, int const samplingFactor,
                       std::vector<size_t>& valuesofN);

void ParseMemType(std::string const& token, int const numCpus, int const numGpus,
                  std::vector<MemType>& memType, std::vector<int>& memIndex);
void ParseExeType(std::string const& token, int const numCpus, int const numGpus,
                  ExeType& exeType, int& exeIndex);

void ParseTransfers(char* line, int numCpus, int numGpus,
                    std::vector<Transfer>& transfers);

void ExecuteTransfers(EnvVars const& ev, int const testNum, size_t const N,
                      std::vector<Transfer>& transfers, bool verbose = true);

void EnablePeerAccess(int const deviceId, int const peerDeviceId);
void AllocateMemory(MemType memType, int devIndex, size_t numBytes, void** memPtr);
void DeallocateMemory(MemType memType, void* memPtr, size_t const size = 0);
void CheckPages(char* byteArray, size_t numBytes, int targetId);
void RunTransfer(EnvVars const& ev, int const iteration, ExecutorInfo& exeInfo, int const transferIdx);
void RunPeerToPeerBenchmarks(EnvVars const& ev, size_t N);
void RunSweepPreset(EnvVars const& ev, size_t const numBytesPerTransfer, int const numGpuSubExec, int const numCpuSubExec, bool const isRandom);

// Return the maximum bandwidth measured for given (src/dst) pair
double GetPeakBandwidth(EnvVars const& ev, size_t  const  N,
                        int     const  isBidirectional,
                        MemType const  srcType, int const srcIndex,
                        MemType const  dstType, int const dstIndex);

std::string GetLinkTypeDesc(uint32_t linkType, uint32_t hopCount);

int RemappedIndex(int const origIdx, bool const isCpuType);
int GetWallClockRate(int deviceId);
void LogTransfers(FILE *fp, int const testNum, std::vector<Transfer> const& transfers);
std::string PtrVectorToStr(std::vector<float*> const& strVector, int const initOffset);
