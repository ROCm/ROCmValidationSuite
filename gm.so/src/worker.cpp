/*******************************************************************************
*
 *
 * Copyright (c) 2018 ROCm Developer Tools
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to 
do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
all
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
#include "include/worker.h"

#include <map>
#include <string>
#include <memory>
#include <utility>

#include "include/rvs_module.h"
#include "include/gpu_util.h"
#include "include/rvs_util.h"
#include "include/rvsloglp.h"
#include "include/rvstimer.h"
#include "include/rsmi_util.h"

#define MODULE_NAME_CAPS                "GM"

#define PCI_ALLOC_ERROR               "pci_alloc() error"
#define GM_RESULT_FAIL_MESSAGE        "FALSE"
#define IRQ_PATH_MAX_LENGTH           256
#define MODULE_NAME                   "gm"
#define GM_TEMP                       "temp"
#define GM_CLOCK                      "clock"
#define GM_MEM_CLOCK                  "mem_clock"
#define GM_FAN                        "fan"
#define GM_POWER                      "power"


// collection of allowed metrics
const char* metric_names[] =
        { GM_TEMP, GM_CLOCK, GM_MEM_CLOCK, GM_FAN, GM_POWER
        };


Worker::Worker() {
  force = false;
}
Worker::~Worker() {}

/**
 * @brief Prints current metric values at every log_interval msec.
 */
void Worker::do_metric_values() {
  std::string msg;
  unsigned int sec;
  unsigned int usec;
  void* r;

  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);
  // add JSON output
  r = rvs::lp::LogRecordCreate("gm", action_name.c_str(), rvs::loginfo,
                               sec, usec);

  for (auto it = met_avg.begin(); it !=
            met_avg.end(); it++) {
    if (bounds[GM_TEMP].mon_metric) {
      msg = "[" + action_name + "] gm " +
          std::to_string((it->second).gpu_id) + " " + GM_TEMP +
          " " + std::to_string(met_value[it->first].temp) + "C";
      rvs::lp::Log(msg, rvs::loginfo, sec, usec);
      rvs::lp::AddString(r,  "info ", msg);
    }
    if (bounds[GM_CLOCK].mon_metric) {
      msg = "[" + action_name + "] gm " +
          std::to_string((it->second).gpu_id) + " " + GM_CLOCK +
          " " + std::to_string(met_value[it->first].clock) + "Mhz";
      rvs::lp::Log(msg, rvs::loginfo, sec, usec);
      rvs::lp::AddString(r,  "info ", msg);
    }
    if (bounds[GM_MEM_CLOCK].mon_metric) {
      msg = "[" + action_name + "] gm " +
          std::to_string((it->second).gpu_id) + " " + GM_MEM_CLOCK +
          " " + std::to_string(met_value[it->first].mem_clock) + "Mhz";
      rvs::lp::Log(msg, rvs::loginfo, sec, usec);
      rvs::lp::AddString(r,  "info ", msg);
    }
    if (bounds[GM_FAN].mon_metric) {
      msg = "[" + action_name + "] gm " +
        std::to_string((it->second).gpu_id) + " " + GM_FAN +
        " " + std::to_string(met_value[it->first].fan) + "%";
      rvs::lp::Log(msg, rvs::loginfo, sec, usec);
      rvs::lp::AddString(r,  "info ", msg);
    }
    if (bounds[GM_POWER].mon_metric) {
      msg = "[" + action_name + "] gm " +
        std::to_string((it->second).gpu_id) + " " + GM_POWER +
        " " + std::to_string(static_cast<float>(met_value[it->first].power) /
                            1e6) + "Watts";
      rvs::lp::Log(msg, rvs::loginfo, sec, usec);
      rvs::lp::AddString(r,  "info ", msg);
    }
  }
  rvs::lp::LogRecordFlush(r);
}

/**
 * @brief Thread function
 *
 * Loops while brun == TRUE and performs polled monitoring avery 1msec.
 *
 * */
void Worker::run() {
  brun = true;
//  std::string val_str;
//  std::vector<std::string> val_vec;

  std::string msg;
  rsmi_status_t status;
  rsmi_frequencies f;
  uint32_t sensor_ind = 0;
  int64_t  temperature;
  int64_t  speed;
  uint64_t power;

  unsigned int sec;
  unsigned int usec;
  void* r;

  rvs::timer<Worker> timer_running(&Worker::do_metric_values, this);

  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);

  // add JSON output
  r = rvs::lp::LogRecordCreate("gm", action_name.c_str(), rvs::loginfo,
                               sec, usec);

  // iterate over devices
  for (auto it = dv_ind.begin(); it != dv_ind.end(); it++) {
    RVSTRACE_
    // fill in the info
    met_avg.insert(std::pair<uint16_t, Metric_avg>
          (it->first, {it->second, 0, 0, 0, 0, 0}));
    met_violation.insert(std::pair<uint16_t, Metric_violation>
          (it->first, {it->second, 0, 0, 0, 0, 0}));
    met_value.insert(std::pair<uint16_t, Metric_value>
          (it->first, {it->second, 0, 0, 0, 0, 0}));

    msg = "[" + action_name + "] gm " + std::to_string(it->second) +
          " started";
    rvs::lp::Log(msg, rvs::logresults, sec, usec);
    rvs::lp::AddString(r, "device", std::to_string(it->second));
    for (auto itb = bounds.begin(); itb != bounds.end(); itb++) {
      RVSTRACE_

      if (itb->second.mon_metric) {
        msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(it->second) + " " + "monitoring " +
            itb->first;
        if (itb->second.check_bounds) {
          msg+= " bounds min: " + std::to_string(itb->second.min_val) +
          "  max: " + std::to_string(itb->second.max_val);
        }
        rvs::lp::Log(msg, rvs::loginfo);
        rvs::lp::AddString(r, itb->first, msg);
      }
    }
  }

  rvs::lp::LogRecordFlush(r);
  // if log_interval timer starts
  if (log_interval) {
    timer_running.start(log_interval);
  }

  count = 0;

  // worker thread has started
  while (brun) {
    RVSTRACE_

    for (auto it = dv_ind.begin(); it != dv_ind.end(); it++) {
      uint32_t ix = it->first;
      int32_t gpuid = it->second;
      RVSTRACE_
      if (bounds[GM_MEM_CLOCK].mon_metric) {
        RVSTRACE_
        status = rsmi_dev_gpu_clk_freq_get(ix, RSMI_CLK_TYPE_MEM, &f);
        uint32_t mhz = f.current;
        met_value[ix].mem_clock = mhz;
        if (!(mhz >= bounds[GM_MEM_CLOCK].min_val && mhz <=
                        bounds[GM_MEM_CLOCK].max_val) &&
                        bounds[GM_MEM_CLOCK].check_bounds) {
          RVSTRACE_
          // write info and increase number of violations
          msg = "[" + action_name  + "] " + MODULE_NAME + " " +
                std::to_string(gpuid) + " " +
                GM_MEM_CLOCK  + " " + "bounds violation " +
                std::to_string(mhz) + "Mhz";
          rvs::lp::Log(msg, rvs::loginfo);
          met_violation[ix].mem_clock_violation++;
          if (term) {
            RVSTRACE_
            if (force) {
              RVSTRACE_
              // stop logging
              rvs::lp::Stop(1);
              // force exit
              exit(EXIT_FAILURE);
            } else {
              RVSTRACE_
              // just signal stop processing
              rvs::lp::Stop(0);
            }
            brun = false;
          }
          RVSTRACE_
        }
        RVSTRACE_
        met_avg[ix].av_mem_clock += mhz;
      }
      RVSTRACE_

      if (bounds[GM_CLOCK].mon_metric) {
        RVSTRACE_
        status = rsmi_dev_gpu_clk_freq_get(ix,
                              RSMI_CLK_TYPE_SYS, &f);
        uint32_t mhz = f.current;
        met_value[ix].clock = mhz;
        if (!(mhz >= bounds[GM_CLOCK].min_val && mhz <=
                    bounds[GM_CLOCK].max_val) &&
                    bounds[GM_CLOCK].check_bounds) {
          RVSTRACE_
          // write info
          msg = "[" + action_name  + "] " + MODULE_NAME + " " +
              std::to_string(met_avg[ix].gpu_id) + " " +
              GM_CLOCK + " " + "bounds violation " +
              std::to_string(mhz) + "Mhz";
          rvs::lp::Log(msg, rvs::loginfo);
          met_violation[ix].clock_violation++;
          if (term) {
            RVSTRACE_
            if (force) {
              RVSTRACE_
              // stop logging
              rvs::lp::Stop(1);
              // force exit
              exit(EXIT_FAILURE);
            } else {
              RVSTRACE_
              // just signal stop processing
              rvs::lp::Stop(0);
            }
            RVSTRACE_
            brun = false;
          }
          RVSTRACE_
        }
        met_avg[ix].av_clock += mhz;
        RVSTRACE_
      }

      RVSTRACE_
      if (bounds[GM_TEMP].mon_metric) {
        RVSTRACE_
        status = rsmi_dev_temp_metric_get(ix, sensor_ind,
                        RSMI_TEMP_CURRENT, &temperature);

#ifdef UT_TCD_1
        status = RSMI_STATUS_UNKNOWN_ERROR;
#endif  // UT_TCD_1
        if (status == RSMI_STATUS_SUCCESS) {
          RVSTRACE_
          uint32_t temper = temperature/1000;
          met_value[ix].temp = temper;
          met_avg[ix].av_temp += temper;
          if (!(temper >= bounds[GM_TEMP].min_val && temper <=
                        bounds[GM_TEMP].max_val) &&
                        bounds[GM_TEMP].check_bounds) {
            RVSTRACE_
            // write info
            msg = "[" + action_name  + "] " + MODULE_NAME + " " +
                std::to_string(met_avg[ix].gpu_id) + " " +
                + GM_TEMP + " " + "bounds violation " +
                std::to_string(temper) + "C";
            rvs::lp::Log(msg, rvs::loginfo);
            met_violation[ix].temp_violation++;
            if (term) {
              RVSTRACE_
              if (force) {
                RVSTRACE_
                // stop logging
                rvs::lp::Stop(1);
                // force exit
                RVSTRACE_
                exit(EXIT_FAILURE);
              } else {
                RVSTRACE_
                // just signal stop processing
                rvs::lp::Stop(0);
              }
              brun = false;
              RVSTRACE_
            }
            RVSTRACE_
          }
          RVSTRACE_
        } else {
          RVSTRACE_
          msg = "[" + action_name  + "] " + MODULE_NAME + " " +
          std::to_string(met_avg[ix].gpu_id) + " " +
          GM_TEMP + " Not available";
          rvs::lp::Log(msg, rvs::loginfo);
        }
        RVSTRACE_
      }

      RVSTRACE_
      if (bounds[GM_FAN].mon_metric) {
        RVSTRACE_
        status = rsmi_dev_fan_speed_get(ix,
                                        sensor_ind, &speed);
#ifdef UT_TCD_1
        status = RSMI_STATUS_UNKNOWN_ERROR;
#endif  // UT_TCD_1
        if (status == RSMI_STATUS_SUCCESS) {
          RVSTRACE_
          met_value[ix].fan = speed;
          met_avg[ix].av_fan += speed;
          if (!(speed >= bounds[GM_FAN].min_val && speed <=
                      bounds[GM_FAN].max_val) &&
                      bounds[GM_FAN].check_bounds) {
            RVSTRACE_
            // write info
            msg = "[" + action_name  + "] " + MODULE_NAME + " " +
                  std::to_string(met_avg[ix].gpu_id) + " " +
                  + GM_FAN + " " + "bounds violation " +
                  std::to_string(speed) + "%";
            rvs::lp::Log(msg, rvs::loginfo);
            met_violation[ix].fan_violation++;
            if (term) {
              RVSTRACE_
              if (force) {
                RVSTRACE_
                // stop logging
                rvs::lp::Stop(1);
                // force exit
                exit(EXIT_FAILURE);
              } else {
                RVSTRACE_
                // just signal stop processing
                rvs::lp::Stop(0);
              }
              brun = false;
              RVSTRACE_
              break;
            }
            RVSTRACE_
          }
          RVSTRACE_
        } else {
          RVSTRACE_
          msg = "[" + action_name  + "] " + MODULE_NAME + " " +
          std::to_string(met_avg[ix].gpu_id) + " " +
          GM_FAN + " Not available";
          rvs::lp::Log(msg, rvs::loginfo);
        }
        RVSTRACE_
      }

      RVSTRACE_
      if (bounds[GM_POWER].mon_metric) {
        RVSTRACE_
        status = rsmi_dev_power_ave_get(ix, sensor_ind, &power);
        met_value[ix].power = power;
        met_avg[ix].av_power += power;
        if (bounds[GM_POWER].check_bounds) {
          RVSTRACE_
          if (power < bounds[GM_POWER].min_val * 1000000 ||
              power > bounds[GM_POWER].max_val * 1000000) {
            RVSTRACE_
            // write info
            msg = "[" + action_name  + "] " + MODULE_NAME + " " +
                  std::to_string(met_avg[ix].gpu_id) + " " +
                  GM_POWER + " " + "bounds violation " +
                  std::to_string(static_cast<float>(power) / 1e6) + "Watts";
            rvs::lp::Log(msg, rvs::loginfo);
            met_violation[ix].power_violation++;
            if (term) {
              RVSTRACE_
              if (force) {
                RVSTRACE_
                // stop logging
                rvs::lp::Stop(1);
                // force exit
                exit(EXIT_FAILURE);
              } else {
                RVSTRACE_
                // just signal stop processing
                rvs::lp::Stop(0);
              }
              brun = false;
              RVSTRACE_
            }
            RVSTRACE_
          }
          RVSTRACE_
        }
        RVSTRACE_
      }
      RVSTRACE_
    }
    count++;
    sleep(sample_interval);
    RVSTRACE_
  }

  RVSTRACE_
  timer_running.stop();
  sleep(200);

  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);

  for (auto it = met_avg.begin();
        it != met_avg.end(); it++) {
    RVSTRACE_
    // add std::string output
    msg = "[" + action_name + "] gm " +
        std::to_string((it->second).gpu_id) + " stopped";
    rvs::lp::Log(msg, rvs::logresults, sec, usec);
  }

  RVSTRACE_
}


/**
 * @brief Stops monitoring
 *
 * Sets brun member to FALSE thus signaling end of monitoring.
 * Then it waits for std::thread to exit before returning.
 *
 * */
void Worker::stop() {
  RVSTRACE_
  rvs::lp::Log("[" + stop_action_name + "] gm in Worker::stop()",
               rvs::logtrace);
  std::string msg;
  unsigned int sec;
  unsigned int usec;
  void* r;
  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);
    // add JSON output
  r = rvs::lp::LogRecordCreate("result", action_name.c_str(), rvs::logresults,
                               sec, usec);
  // reset "run" flag
  brun = false;
  // (give thread chance to finish processing and exit)
  sleep(200);

  if (count != 0) {
    RVSTRACE_
    for (auto it = met_avg.begin(); it !=
            met_avg.end(); it++) {
      RVSTRACE_
      if (bounds[GM_TEMP].mon_metric) {
        RVSTRACE_
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " +
            GM_TEMP + " violations " +
            std::to_string(met_violation[it->first].temp_violation);
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        rvs::lp::AddString(r, "result", msg);
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " "+ GM_TEMP + " average " +
            std::to_string((it->second).av_temp/count) + "C";
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        rvs::lp::AddString(r, "result", msg);
      }
      RVSTRACE_
      if (bounds[GM_CLOCK].mon_metric) {
        RVSTRACE_
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " +
            GM_CLOCK + " violations " +
            std::to_string(met_violation[it->first].clock_violation);
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        rvs::lp::AddString(r, "result", msg);
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " + GM_CLOCK + " average " +
            std::to_string((it->second).av_clock/count) + "Mhz";
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        rvs::lp::AddString(r, "result", msg);
      }
      RVSTRACE_
      if (bounds[GM_MEM_CLOCK].mon_metric) {
        RVSTRACE_
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) +
            " " + GM_MEM_CLOCK + " violations " +
            std::to_string(met_violation[it->first].mem_clock_violation);
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        rvs::lp::AddString(r, "result", msg);
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " +
            GM_MEM_CLOCK + " average " +
            std::to_string((it->second).av_mem_clock/count) + "Mhz";
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        rvs::lp::AddString(r, "result", msg);
      }
      RVSTRACE_
      if (bounds[GM_FAN].mon_metric) {
        RVSTRACE_
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " + GM_FAN +" violations " +
            std::to_string(met_violation[it->first].fan_violation);
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " + GM_FAN + " average " +
            std::to_string((it->second).av_fan/count) + "%";
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
      }
      RVSTRACE_
      if (bounds[GM_POWER].mon_metric) {
        RVSTRACE_
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " +
            GM_POWER + " violations " +
            std::to_string(met_violation[it->first].power_violation);
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        rvs::lp::AddString(r, "result", msg);
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " + GM_POWER + " average " +
            std::to_string((static_cast<float>((it->second).av_power) /
                            count/1e6)) + "Watts";
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        rvs::lp::AddString(r, "result", msg);
      }
      RVSTRACE_
    }
    RVSTRACE_
  }
  RVSTRACE_
  rvs::lp::LogRecordFlush(r);

  // wait a bit to make sure thread has exited
  try {
    if (t.joinable())
      t.join();
    }
  catch(...) {
  }
}
