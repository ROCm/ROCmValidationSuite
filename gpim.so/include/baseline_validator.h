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
#ifndef GPIM_SO_INCLUDE_BASELINE_VALIDATOR_H_
#define GPIM_SO_INCLUDE_BASELINE_VALIDATOR_H_

#include <string>
#include <vector>
#include <cstdint>

#include "include/probes/probe_base.h"
#include "include/baseline_manager.h"

namespace gpim {

/**
 * @enum QualityGrade
 * @brief Baseline quality grade
 */
enum class QualityGrade {
  A,  // Excellent (CV < 0.05, no outliers)
  B,  // Good (CV < 0.10, few outliers)
  C,  // Acceptable (CV < 0.20, some outliers)
  D,  // Poor (CV >= 0.20, many outliers)
  F   // Failed validation
};

/**
 * @struct ValidationResult
 * @brief Result of baseline validation
 */
struct ValidationResult {
  bool passed;                          //!< Overall validation result
  QualityGrade grade;                   //!< Quality grade
  double coefficient_of_variation;      //!< CV = std_dev / mean
  double signal_to_noise_ratio;         //!< SNR = mean / std_dev
  int outlier_count;                    //!< Number of outliers detected
  int sample_count;                     //!< Total samples
  double skewness;                      //!< Distribution skewness
  double kurtosis;                      //!< Distribution kurtosis
  std::string message;                  //!< Validation message
  std::vector<std::string> warnings;    //!< Validation warnings

  ValidationResult() : passed(false), grade(QualityGrade::F),
                      coefficient_of_variation(0.0),
                      signal_to_noise_ratio(0.0),
                      outlier_count(0), sample_count(0),
                      skewness(0.0), kurtosis(0.0) {}

  /**
   * @brief Get quality grade as string
   */
  std::string grade_to_string() const {
    switch (grade) {
      case QualityGrade::A: return "A (Excellent)";
      case QualityGrade::B: return "B (Good)";
      case QualityGrade::C: return "C (Acceptable)";
      case QualityGrade::D: return "D (Poor)";
      case QualityGrade::F: return "F (Failed)";
    }
    return "Unknown";
  }
};

/**
 * @class BaselineValidator
 * @brief Validates baseline quality and removes outliers
 *
 * Validation checks:
 * - Minimum sample count (default: 10)
 * - Outlier detection using IQR method
 * - Coefficient of variation (CV) within limits
 * - Signal-to-noise ratio (SNR) above threshold
 * - Metadata completeness
 * - Cross-probe consistency
 */
class BaselineValidator {
 public:
  BaselineValidator();
  ~BaselineValidator();

  /**
   * @brief Validate baseline samples and assign quality grade
   * @param probe_name Probe identifier
   * @param samples Vector of probe results
   * @param result Output validation result
   * @return 0 on success, non-zero on error
   */
  int validate(const std::string& probe_name,
              const std::vector<ProbeResult>& samples,
              ValidationResult* result);

  /**
   * @brief Remove outliers from samples using IQR method
   * @param samples Input samples
   * @param cleaned_samples Output samples with outliers removed
   * @return Number of outliers removed
   */
  int remove_outliers(const std::vector<ProbeResult>& samples,
                     std::vector<ProbeResult>* cleaned_samples);

  /**
   * @brief Calculate coefficient of variation
   * @param values Vector of values
   * @return CV = std_dev / mean
   */
  double calculate_cv(const std::vector<double>& values) const;

  /**
   * @brief Calculate signal-to-noise ratio
   * @param values Vector of values
   * @return SNR = mean / std_dev
   */
  double calculate_snr(const std::vector<double>& values) const;

  /**
   * @brief Calculate skewness
   * @param values Vector of values
   * @return Skewness measure
   */
  double calculate_skewness(const std::vector<double>& values) const;

  /**
   * @brief Calculate kurtosis
   * @param values Vector of values
   * @return Kurtosis measure
   */
  double calculate_kurtosis(const std::vector<double>& values) const;

  /**
   * @brief Check metadata completeness
   * @param baseline Baseline to check
   * @return true if metadata is complete
   */
  bool check_metadata_completeness(const Baseline& baseline) const;

  /**
   * @brief Validate cross-probe consistency
   * @param baselines Vector of baselines to check
   * @return true if consistent
   */
  bool validate_cross_probe_consistency(const std::vector<Baseline>& baselines) const;

  /**
   * @brief Set minimum sample count
   */
  void set_min_sample_count(int count) { min_sample_count_ = count; }

  /**
   * @brief Set coefficient of variation thresholds for grading
   */
  void set_cv_thresholds(double excellent, double good, double acceptable) {
    cv_threshold_excellent_ = excellent;
    cv_threshold_good_ = good;
    cv_threshold_acceptable_ = acceptable;
  }

  /**
   * @brief Enable/disable outlier removal
   */
  void set_outlier_removal_enabled(bool enabled) {
    outlier_removal_enabled_ = enabled;
  }

  /**
   * @brief Set IQR multiplier for outlier detection (default: 1.5)
   */
  void set_iqr_multiplier(double multiplier) {
    iqr_multiplier_ = multiplier;
  }

 private:
  int min_sample_count_;                //!< Minimum samples required (default: 10)
  double cv_threshold_excellent_;       //!< CV threshold for grade A (default: 0.05)
  double cv_threshold_good_;            //!< CV threshold for grade B (default: 0.10)
  double cv_threshold_acceptable_;      //!< CV threshold for grade C (default: 0.20)
  bool outlier_removal_enabled_;        //!< Enable outlier removal (default: true)
  double iqr_multiplier_;               //!< IQR multiplier (default: 1.5)

  /**
   * @brief Assign quality grade based on metrics
   */
  QualityGrade assign_quality_grade(double cv, int outlier_count,
                                   int sample_count) const;

  /**
   * @brief Calculate basic statistics
   */
  void calculate_statistics(const std::vector<double>& values,
                           double* mean, double* std_dev,
                           double* median, double* min, double* max) const;
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_BASELINE_VALIDATOR_H_
