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
#ifndef PULSE_SO_INCLUDE_PULSE_PATTERN_H_
#define PULSE_SO_INCLUDE_PULSE_PATTERN_H_

#include <algorithm>
#include <cmath>
#include <string>

// Built-in pulse waveforms. Index is the stable selector used by "patterns".
// custom (index -1) uses the action's pulse_rate and high_phase_ratio.

static constexpr int kPulsePatternCount = 6;
static constexpr int kPulsePatternCustom = -1;
static constexpr double kPulseMinPhaseMs = 10.0;

struct PulseWave {
  const char* name;
  double high_ms;
  double low_ms;
};

inline const char* pulse_pattern_name(int index) {
  switch (index) {
    case 0: return "square_slow";
    case 1: return "square_mid";
    case 2: return "square_fast";
    case 3: return "burst";
    case 4: return "freq_sweep";
    case 5: return "duty_sweep";
    case kPulsePatternCustom: return "custom";
    default: return "unknown";
  }
}

inline void pulse_clamp_phases(double& high_ms, double& low_ms) {
  high_ms = std::max(high_ms, kPulseMinPhaseMs);
  low_ms = std::max(low_ms, kPulseMinPhaseMs);
}

inline void pulse_square(double hz, double ratio, double& high_ms, double& low_ms) {
  if (hz < 0.001) hz = 0.001;
  if (ratio < 0.0) ratio = 0.0;
  if (ratio > 1.0) ratio = 1.0;
  const double period = 1000.0 / hz;
  high_ms = period * ratio;
  low_ms = period * (1.0 - ratio);
  pulse_clamp_phases(high_ms, low_ms);
}

// Fraction of the pattern that has elapsed, clamped to [0, 1].
inline double pulse_progress(double elapsed_ms, double pattern_duration_ms) {
  if (pattern_duration_ms <= 0.0) return 0.0;
  double u = elapsed_ms / pattern_duration_ms;
  if (u < 0.0) u = 0.0;
  if (u > 1.0) u = 1.0;
  return u;
}

inline PulseWave pulse_wave(int index, double elapsed_ms, double pattern_duration_ms,
                            double custom_hz, double custom_ratio) {
  PulseWave wave;
  wave.name = pulse_pattern_name(index);
  const double u = pulse_progress(elapsed_ms, pattern_duration_ms);
  switch (index) {
    case 0:
      pulse_square(1.0, 0.5, wave.high_ms, wave.low_ms);
      break;
    case 1:
      pulse_square(5.0, 0.5, wave.high_ms, wave.low_ms);
      break;
    case 2:
      pulse_square(20.0, 0.5, wave.high_ms, wave.low_ms);
      break;
    case 3:
      pulse_square(2.0, 0.1, wave.high_ms, wave.low_ms);
      break;
    case 4: {
      // 1 Hz at the start, 50 Hz at the end, equal high and low.
      const double hz = std::pow(50.0, u);
      pulse_square(hz, 0.5, wave.high_ms, wave.low_ms);
      break;
    }
    case 5: {
      const double ratio = 0.1 + 0.8 * u;
      pulse_square(2.0, ratio, wave.high_ms, wave.low_ms);
      break;
    }
    case kPulsePatternCustom:
    default:
      pulse_square(custom_hz, custom_ratio, wave.high_ms, wave.low_ms);
      break;
  }
  return wave;
}

#endif  // PULSE_SO_INCLUDE_PULSE_PATTERN_H_
