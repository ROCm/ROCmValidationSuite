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
#ifndef GPIM_SO_INCLUDE_BASELINE_REPOSITORY_H_
#define GPIM_SO_INCLUDE_BASELINE_REPOSITORY_H_

#include <string>
#include <map>
#include <vector>

namespace gpim {

/**
 * @class BaselineRepository
 * @brief Manages fetching, validation, and caching of reference baselines
 *
 * Implements a tiered baseline lookup:
 * 1. Local cache directory
 * 2. Baselines shipped with RVS
 * 3. Optional remote repository (no default; must be configured)
 *
 * Not built into the module yet.
 */
class BaselineRepository {
 public:
  BaselineRepository();
  ~BaselineRepository();

  /**
   * @brief Baseline availability status
   */
  enum class BaselineStatus {
    LOCAL_AVAILABLE,      //!< Found in local cache or shipped
    REMOTE_AVAILABLE,     //!< Available for download
    NOT_AVAILABLE         //!< Need to create new baseline
  };

  /**
   * @brief Query parameters for baseline lookup
   */
  struct BaselineQuery {
    std::string gpu_model;         //!< "MI250X", "MI300X", etc.
    std::string rocm_version;      //!< "6.0.2", "6.1.0", etc.
    std::string firmware_version;  //!< Optional: specific firmware
    bool include_community;        //!< Include community-contributed baselines

    BaselineQuery() : include_community(true) {}
  };

  /**
   * @brief Check if baselines exist for given GPU model and ROCm version
   * @param query Baseline query parameters
   * @return Status indicating availability
   */
  BaselineStatus check_availability(const BaselineQuery& query);

  /**
   * @brief Fetch baseline from repository (local or remote)
   * @param query Baseline query parameters
   * @param output_path Where to save fetched baseline
   * @return 0 on success, non-zero on error
   */
  int fetch_baseline(const BaselineQuery& query, const std::string& output_path);

  /**
   * @brief Upload new baseline to community repository (opt-in)
   * @param baseline_path Local baseline file to upload
   * @param metadata Additional metadata (data center, cooling config, etc.)
   * @return 0 on success, non-zero on error
   */
  int contribute_baseline(const std::string& baseline_path,
                         const std::map<std::string, std::string>& metadata);

  /**
   * @brief Validate baseline file integrity
   * @param path Path to baseline JSON file
   * @return true if valid, false otherwise
   */
  bool validate_baseline_file(const std::string& path);

  /**
   * @brief Get recommended baseline search paths
   * Priority: Local cache > Shipped > Remote official > Community
   */
  std::vector<std::string> get_baseline_search_paths(const BaselineQuery& query);

  /**
   * @brief Detect GPU model from device ID
   * @param gpu_id GPU device ID
   * @return Model string (e.g., "MI250X")
   */
  static std::string detect_gpu_model(uint16_t gpu_id);

  /**
   * @brief Get current ROCm version
   * @return Version string (e.g., "6.0.2")
   */
  static std::string get_rocm_version();

 private:
  std::string local_cache_dir_;      //!< Local baseline cache directory
  std::string shipped_dir_;          //!< Baselines shipped with RVS
  std::string remote_repo_url_;      //!< Remote repository URL (empty = disabled)
  std::string community_repo_url_;   //!< Community repository URL (empty = disabled)

  /**
   * @brief Check if baseline exists in local cache
   * @param query Baseline query
   * @param path_out Output parameter for full path if found
   * @return true if exists, false otherwise
   */
  bool exists_in_cache(const BaselineQuery& query, std::string& path_out);

  /**
   * @brief Check if baseline shipped with RVS package
   * @param query Baseline query
   * @param path_out Output parameter for full path if found
   * @return true if exists, false otherwise
   */
  bool exists_in_shipped(const BaselineQuery& query, std::string& path_out);

  /**
   * @brief Check if baseline exists in remote repository
   * @param query Baseline query
   * @param url_out Output parameter for download URL if found
   * @return true if exists, false otherwise
   */
  bool exists_in_remote(const BaselineQuery& query, std::string& url_out);

  /**
   * @brief Download baseline from remote repository
   * @param url URL to download from
   * @param output_path Where to save downloaded file
   * @return 0 on success, non-zero on error
   */
  int download_from_remote(const std::string& url, const std::string& output_path);

  /**
   * @brief Create contribution package (baseline + metadata)
   */
  std::string create_contribution_package(
      const std::string& baseline_path,
      const std::map<std::string, std::string>& metadata);
};

} // namespace gpim

#endif  // GPIM_SO_INCLUDE_BASELINE_REPOSITORY_H_
