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

#include "include/statistical_analyzer.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <utility>

namespace gpim {

namespace {

void mean_and_std_dev(const std::vector<double>& values,
                      double* mean, double* std_dev) {
  *mean = 0.0;
  *std_dev = 0.0;
  if (values.empty()) return;

  *mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  if (values.size() < 2) return;

  double sum_sq_diff = 0.0;
  for (double v : values) {
    double diff = v - *mean;
    sum_sq_diff += diff * diff;
  }
  *std_dev = std::sqrt(sum_sq_diff / (values.size() - 1));
}

// Continued fraction for the incomplete beta function (modified Lentz method).
double incomplete_beta_cf(double x, double a, double b) {
  const int kMaxIter = 300;
  const double kEps = 1e-15;
  const double kTiny = 1e-300;

  double qab = a + b;
  double qap = a + 1.0;
  double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::fabs(d) < kTiny) d = kTiny;
  d = 1.0 / d;
  double h = d;

  for (int m = 1; m <= kMaxIter; m++) {
    int m2 = 2 * m;
    double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kTiny) d = kTiny;
    c = 1.0 + aa / c;
    if (std::fabs(c) < kTiny) c = kTiny;
    d = 1.0 / d;
    h *= d * c;

    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kTiny) d = kTiny;
    c = 1.0 + aa / c;
    if (std::fabs(c) < kTiny) c = kTiny;
    d = 1.0 / d;
    double delta = d * c;
    h *= delta;
    if (std::fabs(delta - 1.0) < kEps) break;
  }
  return h;
}

}  // namespace

StatisticalAnalyzer::StatisticalAnalyzer()
    : significance_level_(0.05) {
}

StatisticalAnalyzer::~StatisticalAnalyzer() {
}

double StatisticalAnalyzer::regularized_incomplete_beta(double x, double a,
                                                        double b) {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;

  double log_front = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                     a * std::log(x) + b * std::log1p(-x);
  double front = std::exp(log_front);

  // The continued fraction converges fastest for x < (a+1)/(a+b+2);
  // use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) otherwise.
  if (x < (a + 1.0) / (a + b + 2.0)) {
    return front * incomplete_beta_cf(x, a, b) / a;
  }
  return 1.0 - front * incomplete_beta_cf(1.0 - x, b, a) / b;
}

double StatisticalAnalyzer::student_t_two_sided_p(double t_stat, double df) {
  if (!(df > 0.0) || std::isnan(t_stat)) return 1.0;
  if (std::isinf(t_stat)) return 0.0;

  double x = df / (df + t_stat * t_stat);
  return regularized_incomplete_beta(x, df / 2.0, 0.5);
}

double StatisticalAnalyzer::student_t_critical(double alpha, double df) {
  if (!(df > 0.0) || !(alpha > 0.0) || !(alpha < 1.0)) return 0.0;

  // p(t) is monotonically decreasing in t; bracket then bisect.
  double lo = 0.0;
  double hi = 1.0;
  while (student_t_two_sided_p(hi, df) > alpha && hi < 1e6) {
    hi *= 2.0;
  }
  for (int i = 0; i < 200 && (hi - lo) > 1e-12 * hi; i++) {
    double mid = 0.5 * (lo + hi);
    if (student_t_two_sided_p(mid, df) > alpha) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

StatisticalTestResult StatisticalAnalyzer::t_test(
    double current_value,
    const std::vector<double>& baseline_samples,
    double significance_level) {

  StatisticalTestResult result;
  result.test_type = "t-test (prediction interval)";

  if (baseline_samples.size() < 2) {
    return result;
  }

  double baseline_mean, baseline_std_dev;
  mean_and_std_dev(baseline_samples, &baseline_mean, &baseline_std_dev);
  int n = static_cast<int>(baseline_samples.size());

  double t_stat = calculate_t_statistic(current_value, baseline_mean,
                                        baseline_std_dev, n);

  result.p_value = student_t_two_sided_p(t_stat, n - 1);
  result.is_significant = (result.p_value < significance_level);
  result.effect_size = calculate_effect_size(current_value, baseline_mean,
                                             baseline_std_dev);
  calculate_confidence_interval(current_value, baseline_mean, baseline_std_dev,
                                n, &result.confidence_lower,
                                &result.confidence_upper);
  result.z_score = calculate_z_score(current_value, baseline_mean,
                                     baseline_std_dev);

  return result;
}

StatisticalTestResult StatisticalAnalyzer::welch_t_test(
    const std::vector<double>& current_samples,
    const std::vector<double>& baseline_samples,
    double significance_level) {

  StatisticalTestResult result;
  result.test_type = "welch-t-test";

  if (current_samples.size() < 2 || baseline_samples.size() < 2) {
    return result;
  }

  double m1, s1, m2, s2;
  mean_and_std_dev(current_samples, &m1, &s1);
  mean_and_std_dev(baseline_samples, &m2, &s2);
  double n1 = static_cast<double>(current_samples.size());
  double n2 = static_cast<double>(baseline_samples.size());

  double v1 = s1 * s1 / n1;
  double v2 = s2 * s2 / n2;
  double se = std::sqrt(v1 + v2);

  if (se == 0.0) {
    result.p_value = (m1 == m2) ? 1.0 : 0.0;
    result.is_significant = (result.p_value < significance_level);
    return result;
  }

  double t_stat = (m1 - m2) / se;
  double df = (v1 + v2) * (v1 + v2) /
              (v1 * v1 / (n1 - 1.0) + v2 * v2 / (n2 - 1.0));

  result.p_value = student_t_two_sided_p(t_stat, df);
  result.is_significant = (result.p_value < significance_level);
  result.z_score = t_stat;
  result.effect_size = calculate_effect_size(
      m1, m2,
      calculate_pooled_std_dev(s1, static_cast<int>(n1), s2, static_cast<int>(n2)));

  if (m2 != 0.0) {
    double margin = student_t_critical(significance_level, df) * se;
    result.confidence_lower = ((m1 - m2) - margin) / m2 * 100.0;
    result.confidence_upper = ((m1 - m2) + margin) / m2 * 100.0;
  }

  return result;
}

StatisticalTestResult StatisticalAnalyzer::mann_whitney_u_test(
    const std::vector<double>& current_samples,
    const std::vector<double>& baseline_samples,
    double significance_level) {

  StatisticalTestResult result;
  result.test_type = "mann-whitney-u";

  if (current_samples.empty() || baseline_samples.empty()) {
    return result;
  }

  // value, group (0 = baseline, 1 = current)
  std::vector<std::pair<double, int>> combined;
  combined.reserve(current_samples.size() + baseline_samples.size());
  for (double v : baseline_samples) combined.push_back({v, 0});
  for (double v : current_samples) combined.push_back({v, 1});

  std::sort(combined.begin(), combined.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  // Average ranks over ties and accumulate the tie correction term.
  double rank_sum_baseline = 0.0;
  double tie_term = 0.0;
  size_t i = 0;
  while (i < combined.size()) {
    size_t j = i;
    while (j + 1 < combined.size() && combined[j + 1].first == combined[i].first) {
      j++;
    }
    double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j + 1)) / 2.0;
    for (size_t k = i; k <= j; k++) {
      if (combined[k].second == 0) rank_sum_baseline += avg_rank;
    }
    double t = static_cast<double>(j - i + 1);
    tie_term += t * t * t - t;
    i = j + 1;
  }

  double n1 = static_cast<double>(baseline_samples.size());
  double n2 = static_cast<double>(current_samples.size());
  double n = n1 + n2;

  double u1 = rank_sum_baseline - n1 * (n1 + 1.0) / 2.0;
  double expected_u = n1 * n2 / 2.0;
  double var_u = n1 * n2 / 12.0 * ((n + 1.0) - tie_term / (n * (n - 1.0)));

  if (var_u <= 0.0) {
    return result;  // All values identical
  }

  // Continuity correction toward the mean.
  double diff = u1 - expected_u;
  double corrected = (std::fabs(diff) > 0.5) ? (std::fabs(diff) - 0.5) : 0.0;
  double z = std::copysign(corrected, diff) / std::sqrt(var_u);

  // Positive z means current samples rank higher than baseline.
  result.z_score = -z;
  result.p_value = std::erfc(std::fabs(z) / std::sqrt(2.0));
  result.is_significant = (result.p_value < significance_level);

  return result;
}

double StatisticalAnalyzer::calculate_effect_size(double current_mean,
                                                  double baseline_mean,
                                                  double pooled_std_dev) {
  if (pooled_std_dev == 0.0) return 0.0;

  return std::abs(current_mean - baseline_mean) / pooled_std_dev;
}

void StatisticalAnalyzer::calculate_confidence_interval(
    double current_value,
    double baseline_mean,
    double baseline_std_dev,
    int sample_count,
    double* lower,
    double* upper) {

  if (sample_count <= 1 || baseline_mean == 0.0) {
    *lower = 0.0;
    *upper = 0.0;
    return;
  }

  double t_crit = student_t_critical(significance_level_, sample_count - 1);
  double margin = t_crit * baseline_std_dev * std::sqrt(1.0 + 1.0 / sample_count);
  double diff = current_value - baseline_mean;

  *lower = (diff - margin) / baseline_mean * 100.0;
  *upper = (diff + margin) / baseline_mean * 100.0;
}

double StatisticalAnalyzer::calculate_z_score(double value, double mean,
                                              double std_dev) {
  if (std_dev == 0.0) return 0.0;

  return (value - mean) / std_dev;
}

double StatisticalAnalyzer::calculate_t_statistic(
    double current_value,
    double baseline_mean,
    double baseline_std_dev,
    int sample_count) {

  if (baseline_std_dev == 0.0 || sample_count <= 1) {
    return 0.0;
  }

  return (current_value - baseline_mean) /
         (baseline_std_dev * std::sqrt(1.0 + 1.0 / sample_count));
}

double StatisticalAnalyzer::calculate_pooled_std_dev(
    double std_dev1, int n1,
    double std_dev2, int n2) {

  if (n1 + n2 <= 2) return 0.0;

  double var1 = std_dev1 * std_dev1;
  double var2 = std_dev2 * std_dev2;

  double pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);

  return std::sqrt(pooled_var);
}

}  // namespace gpim
