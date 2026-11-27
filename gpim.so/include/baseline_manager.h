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
 #ifndef GPIM_SO_INCLUDE_BASELINE_MANAGER_H_
 #define GPIM_SO_INCLUDE_BASELINE_MANAGER_H_
 
 #include <string>
 #include <map>
 #include <vector>
 #include <cstdint>
 
 #include "include/probes/probe_base.h"
 
 namespace gpim {
 
 /**
  * @struct Baseline
  * @brief Performance baseline for a specific probe and GPU
  */
 struct Baseline {
   std::string probe_name;       //!< Probe identifier
   uint16_t gpu_id;              //!< GPU device ID
   double value;                 //!< Baseline median value
   double std_dev;               //!< Standard deviation
   uint64_t timestamp;           //!< When baseline was created (microseconds)
   int sample_count;             //!< Number of samples used
   std::string metadata;         //!< JSON metadata (driver version, etc.)
 
   Baseline() : gpu_id(0), value(0.0), std_dev(0.0),
                timestamp(0), sample_count(0) {}
 };
 
 /**
  * @class BaselineManager
  * @brief Manages performance baselines (create, read, update, export/import)
  *
  * Baselines are stored in JSON format for easy sharing across nodes.
  * File format:
  * {
  *   "version": 1,
  *   "created": "2024-03-10T12:00:00Z",
  *   "node_info": { "hostname": "...", "driver_version": "..." },
  *   "baselines": [
  *     {
  *       "probe_name": "kernel_launch_latency",
  *       "gpu_id": 0,
  *       "value": 8.5,
  *       "std_dev": 0.3,
  *       "sample_count": 50,
  *       "timestamp": 1710086400000000
  *     },
  *     ...
  *   ]
  * }
  */
 class BaselineManager {
  public:
   /**
    * @brief Constructor
    * @param baseline_file Path to baseline JSON file
    */
   explicit BaselineManager(const std::string& baseline_file);
   ~BaselineManager();
 
   /**
    * @brief Load baselines from file
    * @return 0 on success, non-zero on error
    */
   int load();
 
   /**
    * @brief Save baselines to file
    * @return 0 on success, non-zero on error
    */
   int save();
 
   /**
    * @brief Create baseline from multiple probe results
    * @param probe_name Probe identifier
    * @param gpu_id GPU device ID
    * @param samples Vector of probe results
    * @return 0 on success, non-zero on error
    */
   int create_baseline(const std::string& probe_name,
                      uint16_t gpu_id,
                      const std::vector<ProbeResult>& samples);
 
   /**
    * @brief Get baseline for a specific probe and GPU
    * @param probe_name Probe identifier
    * @param gpu_id GPU device ID
    * @param baseline Output parameter for baseline
    * @return 0 on success, 1 if not found, negative on error
    */
   int get_baseline(const std::string& probe_name,
                   uint16_t gpu_id,
                   Baseline* baseline) const;
 
   /**
    * @brief Check if baseline exists
    */
   bool has_baseline(const std::string& probe_name, uint16_t gpu_id) const;
 
   /**
    * @brief List all baselines for a GPU (or all GPUs if gpu_id == -1)
    */
   std::vector<Baseline> list_baselines(int gpu_id = -1) const;
 
   /**
    * @brief Delete baseline
    */
   int delete_baseline(const std::string& probe_name, uint16_t gpu_id);
 
   /**
    * @brief Export baselines to JSON string
    */
   std::string export_json() const;
 
   /**
    * @brief Import baselines from JSON string
    * @param json JSON string containing baselines
    * @param merge If true, merge with existing baselines; if false, replace
    * @return 0 on success, non-zero on error
    */
   int import_json(const std::string& json, bool merge = true);
 
   /**
    * @brief Include the node hostname in saved baseline metadata
    */
   void set_record_hostname(bool enabled) { record_hostname_ = enabled; }
 
  private:
   std::string baseline_file_;   //!< Path to baseline JSON file
   bool record_hostname_ = false;  //!< Write hostname into node_info
 
   // In-memory baseline storage: key = "probe_name:gpu_id"
   std::map<std::string, Baseline> baselines_;
 
   /**
    * @brief Generate key for baseline lookup
    */
   static std::string make_key(const std::string& probe_name, uint16_t gpu_id);
 
   /**
    * @brief Collect system metadata for baseline
    */
   std::string collect_metadata() const;
 
   /**
    * @brief Get ROCm installation version
    */
   std::string get_rocm_version() const;
 };
 
 }  // namespace gpim
 
 #endif  // GPIM_SO_INCLUDE_BASELINE_MANAGER_H_
 