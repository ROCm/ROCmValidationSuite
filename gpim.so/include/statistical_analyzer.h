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
#ifndef GPIM_SO_INCLUDE_STATISTICAL_ANALYZER_H_
#define GPIM_SO_INCLUDE_STATISTICAL_ANALYZER_H_

#include <vector>
#include <string>

namespace gpim {

/**
 * @struct StatisticalTestResult
 * @brief Result of statistical significance testing
 */
struct StatisticalTestResult {
  double p_value;               //!< P-value (probability of observing this difference by chance)
  double effect_size;           //!< Cohen's d (standardized effect size)
  double confidence_lower;      //!< Lower bound of 95% confidence interval
  double confidence_upper;      //!< Upper bound of 95% confidence interval
  double z_score;               //!< Z-score for outlier detection
  bool is_significant;          //!< True if p < significance_level
  std::string test_type;        //!< Type of test performed

  StatisticalTestResult() : p_value(1.0), effect_size(0.0),
                           confidence_lower(0.0), confidence_upper(0.0),
                           z_score(0.0), is_significant(false) {}
};

/**
 * @class StatisticalAnalyzer
 * @brief Performs statistical significance testing for regression detection
 *
 * Implements:
 * - Prediction-interval t-test for a single new observation
 * - Welch's two-sample t-test (unequal variances)
 * - Mann-Whitney U test (non-parametric, tie-corrected)
 * - Cohen's d effect size
 * - Confidence/prediction intervals
 * - Z-score calculation
 */
class StatisticalAnalyzer {
 public:
  StatisticalAnalyzer();
  ~StatisticalAnalyzer();

  /**
   * @brief Test whether a single new observation belongs to the baseline
   *        distribution.
   *
   * Uses t = (x - mean) / (s * sqrt(1 + 1/n)) with n-1 degrees of freedom.
   * confidence_lower/upper hold the 95% prediction interval of the
   * difference, expressed as percent of the baseline mean.
   *
   * @param current_value Current measured value
   * @param baseline_samples Baseline sample values
   * @param significance_level Alpha level (default: 0.05 for 95% confidence)
   * @return Statistical test result
   */
  StatisticalTestResult t_test(double current_value,
                               const std::vector<double>& baseline_samples,
                               double significance_level = 0.05);

  /**
   * @brief Welch's t-test comparing current samples against baseline samples
   *
   * confidence_lower/upper hold the confidence interval of the difference of
   * means, expressed as percent of the baseline mean.
   */
  StatisticalTestResult welch_t_test(
      const std::vector<double>& current_samples,
      const std::vector<double>& baseline_samples,
      double significance_level = 0.05);

  /**
   * @brief Perform Mann-Whitney U test (non-parametric alternative to t-test)
   * @param current_samples Current measurement samples
   * @param baseline_samples Baseline samples
   * @param significance_level Alpha level
   * @return Statistical test result
   */
  StatisticalTestResult mann_whitney_u_test(
      const std::vector<double>& current_samples,
      const std::vector<double>& baseline_samples,
      double significance_level = 0.05);

  /**
   * @brief Two-sided p-value of Student's t distribution
   * @param t_stat t statistic
   * @param df Degrees of freedom (may be non-integer, e.g. Welch)
   */
  static double student_t_two_sided_p(double t_stat, double df);

  /**
   * @brief Two-sided critical value t such that P(|T| > t) = alpha
   */
  static double student_t_critical(double alpha, double df);

  /**
   * @brief Regularized incomplete beta function I_x(a, b)
   */
  static double regularized_incomplete_beta(double x, double a, double b);

  /**
   * @brief Calculate Cohen's d effect size
   * @param current_mean Mean of current measurements
   * @param baseline_mean Mean of baseline
   * @param pooled_std_dev Pooled standard deviation
   * @return Effect size (small: 0.2, medium: 0.5, large: 0.8)
   */
  double calculate_effect_size(double current_mean,
                               double baseline_mean,
                               double pooled_std_dev);

  /**
   * @brief Calculate 95% prediction interval of (current - baseline_mean)
   *        as percent of baseline_mean
   * @param current_value Current value
   * @param baseline_mean Baseline mean
   * @param baseline_std_dev Baseline standard deviation
   * @param sample_count Number of baseline samples
   * @param lower Output: lower bound (%)
   * @param upper Output: upper bound (%)
   */
  void calculate_confidence_interval(double current_value,
                                     double baseline_mean,
                                     double baseline_std_dev,
                                     int sample_count,
                                     double* lower,
                                     double* upper);

  /**
   * @brief Calculate Z-score for outlier detection
   * @param value Value to test
   * @param mean Population mean
   * @param std_dev Population standard deviation
   * @return Z-score (|z| > 3 typically indicates outlier)
   */
  double calculate_z_score(double value, double mean, double std_dev);

  /**
   * @brief Set significance level (alpha)
   */
  void set_significance_level(double alpha) {
    significance_level_ = alpha;
  }

 private:
  double significance_level_;  //!< Alpha level for hypothesis testing (default: 0.05)

  /**
   * @brief Prediction-interval t statistic for one new observation
   */
  double calculate_t_statistic(double current_value, double baseline_mean,
                               double baseline_std_dev, int sample_count);

  /**
   * @brief Calculate pooled standard deviation
   */
  double calculate_pooled_std_dev(double std_dev1, int n1,
                                  double std_dev2, int n2);
};

}  // namespace gpim

#endif  // GPIM_SO_INCLUDE_STATISTICAL_ANALYZER_H_
