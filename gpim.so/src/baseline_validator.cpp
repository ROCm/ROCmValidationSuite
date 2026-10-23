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

#include "include/baseline_validator.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace gpim {

BaselineValidator::BaselineValidator()
    : min_sample_count_(10),
      cv_threshold_excellent_(0.05),
      cv_threshold_good_(0.10),
      cv_threshold_acceptable_(0.20),
      outlier_removal_enabled_(true),
      iqr_multiplier_(1.5) {
}

BaselineValidator::~BaselineValidator() {
}

int BaselineValidator::validate(const std::string& probe_name,
                                const std::vector<ProbeResult>& samples,
                                ValidationResult* result) {
  if (!result) return -1;

  result->sample_count = samples.size();

  // Check minimum sample count
  if (static_cast<int>(samples.size()) < min_sample_count_) {
    result->passed = false;
    result->grade = QualityGrade::F;
    std::ostringstream oss;
    oss << "Insufficient samples: " << samples.size()
        << " (minimum: " << min_sample_count_ << ")";
    result->message = oss.str();
    return -1;
  }

  // Extract values
  std::vector<double> values;
  values.reserve(samples.size());
  for (const auto& sample : samples) {
    values.push_back(sample.value);
  }

  // Remove outliers if enabled
  std::vector<ProbeResult> cleaned_samples = samples;
  if (outlier_removal_enabled_) {
    std::vector<ProbeResult> temp;
    result->outlier_count = remove_outliers(samples, &temp);
    if (!temp.empty()) {
      cleaned_samples = temp;

      // Recalculate values without outliers
      values.clear();
      for (const auto& sample : cleaned_samples) {
        values.push_back(sample.value);
      }
    }
  }

  // Calculate statistics
  double mean, std_dev, median, min_val, max_val;
  calculate_statistics(values, &mean, &std_dev, &median, &min_val, &max_val);

  // Calculate quality metrics
  result->coefficient_of_variation = calculate_cv(values);
  result->signal_to_noise_ratio = calculate_snr(values);
  result->skewness = calculate_skewness(values);
  result->kurtosis = calculate_kurtosis(values);

  // Assign quality grade
  result->grade = assign_quality_grade(result->coefficient_of_variation,
                                      result->outlier_count,
                                      result->sample_count);

  // Check if passed (grade C or better)
  result->passed = (result->grade == QualityGrade::A ||
                   result->grade == QualityGrade::B ||
                   result->grade == QualityGrade::C);

  // Generate validation message
  std::ostringstream oss;
  oss << "Baseline validation: " << result->grade_to_string()
      << " (CV=" << std::fixed << std::setprecision(4) << result->coefficient_of_variation
      << ", SNR=" << result->signal_to_noise_ratio
      << ", outliers=" << result->outlier_count << ")";
  result->message = oss.str();

  // Add warnings if applicable
  if (result->coefficient_of_variation > cv_threshold_acceptable_) {
    result->warnings.push_back("High coefficient of variation (unstable measurements)");
  }

  if (result->outlier_count > static_cast<int>(samples.size()) * 0.1) {
    result->warnings.push_back("Many outliers detected (>10% of samples)");
  }

  if (std::abs(result->skewness) > 1.0) {
    result->warnings.push_back("High skewness (non-normal distribution)");
  }

  if (result->kurtosis > 3.0 || result->kurtosis < -3.0) {
    result->warnings.push_back("High kurtosis (heavy tails or outliers)");
  }

  return 0;
}

int BaselineValidator::remove_outliers(const std::vector<ProbeResult>& samples,
                                      std::vector<ProbeResult>* cleaned_samples) {
  if (!cleaned_samples) return -1;
  cleaned_samples->clear();

  if (samples.empty()) return 0;

  // Extract values for IQR calculation
  std::vector<double> values;
  values.reserve(samples.size());
  for (const auto& sample : samples) {
    values.push_back(sample.value);
  }

  // Sort values for percentile calculation
  std::vector<double> sorted_values = values;
  std::sort(sorted_values.begin(), sorted_values.end());

  // Calculate Q1, Q3, and IQR
  size_t n = sorted_values.size();
  double q1 = sorted_values[n / 4];
  double q3 = sorted_values[(3 * n) / 4];
  double iqr = q3 - q1;

  // Calculate outlier bounds
  double lower_bound = q1 - iqr_multiplier_ * iqr;
  double upper_bound = q3 + iqr_multiplier_ * iqr;

  // Filter out outliers
  int outlier_count = 0;
  for (const auto& sample : samples) {
    if (sample.value >= lower_bound && sample.value <= upper_bound) {
      cleaned_samples->push_back(sample);
    } else {
      outlier_count++;
    }
  }

  return outlier_count;
}

double BaselineValidator::calculate_cv(const std::vector<double>& values) const {
  if (values.empty()) return 0.0;

  double mean, std_dev, median, min_val, max_val;
  calculate_statistics(values, &mean, &std_dev, &median, &min_val, &max_val);

  if (mean == 0.0) return 0.0;

  return std_dev / mean;
}

double BaselineValidator::calculate_snr(const std::vector<double>& values) const {
  if (values.empty()) return 0.0;

  double mean, std_dev, median, min_val, max_val;
  calculate_statistics(values, &mean, &std_dev, &median, &min_val, &max_val);

  if (std_dev == 0.0) return INFINITY;

  return mean / std_dev;
}

double BaselineValidator::calculate_skewness(const std::vector<double>& values) const {
  if (values.size() < 3) return 0.0;

  double mean, std_dev, median, min_val, max_val;
  calculate_statistics(values, &mean, &std_dev, &median, &min_val, &max_val);

  if (std_dev == 0.0) return 0.0;

  double sum_cubed_dev = 0.0;
  for (double v : values) {
    double dev = (v - mean) / std_dev;
    sum_cubed_dev += dev * dev * dev;
  }

  return sum_cubed_dev / values.size();
}

double BaselineValidator::calculate_kurtosis(const std::vector<double>& values) const {
  if (values.size() < 4) return 0.0;

  double mean, std_dev, median, min_val, max_val;
  calculate_statistics(values, &mean, &std_dev, &median, &min_val, &max_val);

  if (std_dev == 0.0) return 0.0;

  double sum_fourth_dev = 0.0;
  for (double v : values) {
    double dev = (v - mean) / std_dev;
    sum_fourth_dev += dev * dev * dev * dev;
  }

  // Excess kurtosis (normal distribution has kurtosis of 3)
  return (sum_fourth_dev / values.size()) - 3.0;
}

bool BaselineValidator::check_metadata_completeness(const Baseline& baseline) const {
  // Check required fields
  if (baseline.probe_name.empty()) return false;
  if (baseline.value == 0.0 && baseline.std_dev == 0.0) return false;
  if (baseline.sample_count == 0) return false;
  if (baseline.timestamp == 0) return false;

  return true;
}

bool BaselineValidator::validate_cross_probe_consistency(
    const std::vector<Baseline>& baselines) const {
  if (baselines.size() < 2) return true;  // Nothing to compare

  // Check for same GPU ID
  uint16_t gpu_id = baselines[0].gpu_id;
  for (const auto& b : baselines) {
    if (b.gpu_id != gpu_id) {
      return false;  // Mixed GPU IDs
    }
  }

  // Check timestamps are reasonably close (within 1 hour)
  const uint64_t ONE_HOUR_US = 3600ULL * 1000000ULL;
  uint64_t min_timestamp = baselines[0].timestamp;
  uint64_t max_timestamp = baselines[0].timestamp;

  for (const auto& b : baselines) {
    if (b.timestamp < min_timestamp) min_timestamp = b.timestamp;
    if (b.timestamp > max_timestamp) max_timestamp = b.timestamp;
  }

  if (max_timestamp - min_timestamp > ONE_HOUR_US) {
    return false;  // Baselines created too far apart
  }

  // Could add more checks here (e.g., GEMM performance vs memory bandwidth correlation)

  return true;
}

QualityGrade BaselineValidator::assign_quality_grade(double cv, int outlier_count,
                                                    int sample_count) const {
  // Excellent: Low CV, no outliers
  if (cv < cv_threshold_excellent_ && outlier_count == 0) {
    return QualityGrade::A;
  }

  // Good: Moderate CV, few outliers
  if (cv < cv_threshold_good_ && outlier_count < sample_count * 0.05) {
    return QualityGrade::B;
  }

  // Acceptable: Higher CV but still usable
  if (cv < cv_threshold_acceptable_ && outlier_count < sample_count * 0.10) {
    return QualityGrade::C;
  }

  // Poor: High CV or many outliers
  if (cv < 0.50) {
    return QualityGrade::D;
  }

  // Failed: Unusable baseline
  return QualityGrade::F;
}

void BaselineValidator::calculate_statistics(const std::vector<double>& values,
                                             double* mean, double* std_dev,
                                             double* median, double* min, double* max) const {
  if (values.empty()) {
    *mean = *std_dev = *median = *min = *max = 0.0;
    return;
  }

  // Calculate mean
  double sum = 0.0;
  for (double v : values) sum += v;
  *mean = sum / values.size();

  // Calculate std dev
  double sum_sq_diff = 0.0;
  for (double v : values) {
    double diff = v - *mean;
    sum_sq_diff += diff * diff;
  }
  *std_dev = (values.size() > 1) ? std::sqrt(sum_sq_diff / (values.size() - 1)) : 0.0;

  // Calculate median
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  *median = sorted[sorted.size() / 2];

  // Calculate min/max
  *min = *std::min_element(sorted.begin(), sorted.end());
  *max = *std::max_element(sorted.begin(), sorted.end());
}

}  // namespace gpim
