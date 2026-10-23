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
#ifndef GPIM_SO_INCLUDE_BASELINE_VERSION_MANAGER_H_
#define GPIM_SO_INCLUDE_BASELINE_VERSION_MANAGER_H_

#include <string>
#include <cstdint>

namespace gpim {

/**
 * @struct BaselineVersion
 * @brief Version information for baseline files
 */
struct BaselineVersion {
  int major;                    //!< Major version (breaking changes)
  int minor;                    //!< Minor version (feature additions)
  int patch;                    //!< Patch version (bug fixes)
  std::string gpu_model;        //!< GPU model (e.g., "MI300X", "MI250X")
  std::string rocm_version;     //!< ROCm version (e.g., "6.0.2")
  std::string driver_version;   //!< Driver version
  std::string date;             //!< Creation date (YYYYMMDD format)

  BaselineVersion() : major(1), minor(0), patch(0) {}

  /**
   * @brief Get version as string "major.minor.patch"
   */
  std::string to_string() const;

  /**
   * @brief Parse version from string
   */
  static BaselineVersion from_string(const std::string& version_str);
};

/**
 * @class BaselineVersionManager
 * @brief Manages baseline versioning and compatibility
 *
 * Version format: {GPU_MODEL}_{ROCM_VERSION}_{DRIVER_VERSION}_{DATE}.json
 * Example: MI300X_6.0.2_6.11.0_20260325.json
 *
 * Semantic versioning for baseline format:
 * - MAJOR: Breaking changes to baseline structure
 * - MINOR: New probes or metrics added
 * - PATCH: Bug fixes or metadata updates
 */
class BaselineVersionManager {
 public:
  BaselineVersionManager();
  ~BaselineVersionManager();

  /**
   * @brief Generate versioned baseline filename
   * @param base_path Base directory path
   * @param gpu_model GPU model string
   * @param rocm_version ROCm version
   * @param driver_version Driver version
   * @return Full path to versioned baseline file
   */
  std::string generate_baseline_filename(const std::string& base_path,
                                         const std::string& gpu_model,
                                         const std::string& rocm_version,
                                         const std::string& driver_version) const;

  /**
   * @brief Parse version information from filename
   * @param filename Baseline filename
   * @param version Output parameter for version info
   * @return 0 on success, non-zero on error
   */
  int parse_filename(const std::string& filename, BaselineVersion* version) const;

  /**
   * @brief Check if two baseline versions are compatible
   * @return true if compatible (can use baseline), false otherwise
   */
  bool is_compatible(const BaselineVersion& v1, const BaselineVersion& v2) const;

  /**
   * @brief Get current baseline format version
   */
  BaselineVersion get_current_version() const;

  /**
   * @brief Set baseline format version (for migration)
   */
  void set_version(int major, int minor, int patch);

  /**
   * @brief Check if baseline file needs migration
   * @param file_version Version from file
   * @return true if migration needed
   */
  bool needs_migration(const BaselineVersion& file_version) const;

  /**
   * @brief Get GPU model string from system
   * @param gpu_id GPU device ID
   * @return GPU model string (e.g., "MI300X")
   */
  std::string detect_gpu_model(uint16_t gpu_id) const;

  /**
   * @brief Get driver version from system
   * @return Driver version string
   */
  std::string detect_driver_version() const;

  /**
   * @brief Get ROCm version from system
   * @return ROCm version string
   */
  std::string detect_rocm_version() const;

 private:
  BaselineVersion current_version_;  //!< Current baseline format version

  /**
   * @brief Get current date as string (YYYYMMDD)
   */
  std::string get_current_date() const;

  /**
   * @brief Sanitize string for filename (remove invalid chars)
   */
  std::string sanitize_for_filename(const std::string& str) const;
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_BASELINE_VERSION_MANAGER_H_
