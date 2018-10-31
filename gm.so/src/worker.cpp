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
#include "worker.h"

#include <assert.h>
#include <stdlib.h>

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include <utility>
#include <fstream>

#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>
#include <linux/pci.h>
#ifdef __cplusplus
}
#endif

#include "rvs_module.h"
#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsloglp.h"
#include "rvstimer.h"

using std::string;
using std::vector;
using std::map;
using std::fstream;

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
  bfiltergpu = false;
  force = false;
  auto metric_length = std::end(metric_names) - std::begin(metric_names);
    for (int i = 0; i < metric_length; i++) {
        bounds.insert(std::pair<string, Metric_bound>(metric_names[i],
                                                      {false, false, 0, 0}));
    }
}
Worker::~Worker() {}

/**
 * @brief Sets GPU IDs for filtering
 * @arg GpuIds Array of GPU GpuIds
 */
void Worker::set_gpuids(const std::vector<uint16_t>& GpuIds) {
  gpuids = GpuIds;
  bfiltergpu = true;
}

/**
 * @brief sets values used to check if metrics is monitored
 * @param metr_name represent metric
 * @param metr_true if metric is going to be monitored
 */ 
void Worker::set_metr_mon(string metr_name, bool metr_true) {
  bounds[metr_name].mon_metric = metr_true;
}
/**
 * @brief sets bounding box values for metric
 * @param metr_name represent metric
 * @param met_bound if bound provided
 * @param metr_max max metric value allowed
 * @param metr_min min metric value allowed
 */ 
void Worker::set_bound(std::string metr_name, bool met_bound, int metr_max,
                       int metr_min) {
  bounds[metr_name].check_bounds = met_bound;
  bounds[metr_name].max_val = metr_max;
  bounds[metr_name].min_val = metr_min;
}

/**
 * @brief gets power value for device
 * @param path represents path of device
 * @return power value
 */
int Worker::get_power(const std::string path) {
  string retStr;
  auto tempPath = path;

  tempPath += "/";
  tempPath += "power1_average";

  std::ifstream fs;
  fs.open(tempPath);

  fs >> retStr;
  fs.close();
  return stoi(retStr);
}

/**
 * @brief Prints current metric values at every log_interval msec.
 */
void Worker::do_metric_values() {
  string msg;
  unsigned int sec;
  unsigned int usec;
  void* r;

  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);
  // add JSON output
  r = rvs::lp::LogRecordCreate("gm", action_name.c_str(), rvs::loginfo,
                               sec, usec);

  for (auto it = irq_gpu_ids.begin(); it !=
            irq_gpu_ids.end(); it++) {
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
        " " + std::to_string(met_value[it->first].power) + "Watts";
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
  vector<uint16_t> gpus_location_id;
  std::string val_str;
  std::vector<std::string> val_vec;

  char path[IRQ_PATH_MAX_LENGTH];
  uint32_t value;
  uint32_t value2;
  uint16_t dev_idx = 0;
  string msg;
  int ret;

  struct pci_access *pacc;
  struct pci_dev *dev_pci;

  unsigned int sec;
  unsigned int usec;
  void* r;

  rvs::timer<Worker> timer_running(&Worker::do_metric_values, this);

  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);
  
          rsmi_init(0);
          uint64_t id;
          uint32_t dv_ind=0;
          uint32_t num_devices;
          rsmi_status_t number_status =rsmi_num_monitor_devices(&num_devices);
          for(int i=0; i<num_devices;i++) {
          rsmi_status_t status = rsmi_dev_id_get(i, &id);
          std::cout << i << " " << id << std::endl;
          }


  // add JSON output
  r = rvs::lp::LogRecordCreate("gm", action_name.c_str(), rvs::loginfo,
                               sec, usec);

  // get the pci_access structure
  pacc = pci_alloc();
  // initialize the PCI library
  pci_init(pacc);
  // get the list of devices
  pci_scan_bus(pacc);

    // iterate over devices
    for (dev_pci = pacc->devices; dev_pci; dev_pci = dev_pci->next) {
      // fill in the info
        pci_fill_info(dev_pci,
                PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS);

      // computes the actual dev's location_id (sysfs entry)
      uint16_t dev_location_id = ((((uint16_t) (dev_pci->bus)) << 8)
                 | (dev_pci->func));

        uint32_t gpu_id = rvs::gpulist::GetGpuId(dev_location_id);
                if (gpu_id == -1)
            continue;

        if (std::find(gpuids.begin(), gpuids.end(),
                    gpu_id) !=gpuids.end()) {
          msg = "[" + action_name + "] gm " + std::to_string(gpu_id) +
                " started";
          rvs::lp::Log(msg, rvs::logresults, sec, usec);
          rvs::lp::AddString(r, "device", std::to_string(gpu_id));

          auto metric_length = std::end(metric_names) -
                          std::begin(metric_names);
          for (int i = 0; i < metric_length; i++) {
            if (bounds[metric_names[i]].mon_metric) {
              msg = "[" + action_name + "] " + MODULE_NAME + " " +
                  std::to_string(gpu_id) + " " + " monitoring " +
                  metric_names[i];
              if (bounds[metric_names[i]].check_bounds) {
                            msg+= " bounds min:" +
                            std::to_string(bounds[metric_names[i]].min_val) +
                            " max:" + std::to_string
                            (bounds[metric_names[i]].max_val);
              }
              log(msg.c_str(), rvs::loginfo);
              rvs::lp::AddString(r,  metric_names[i], msg);
            }
          }

          irq_gpu_ids.insert(std::pair<uint16_t, Dev_metrics>
                (dev_idx, {gpu_id, 0, 0, 0, 0, 0}));
          met_violation.insert(std::pair<uint16_t, Metric_violation>
                (dev_idx, {0, 0, 0, 0, 0}));
          met_value.insert(std::pair<uint16_t, Metric_value>
                (dev_idx, {0, 0, 0, 0, 0}));
          count = 0;
                  
          rsmi_status_t status = rsmi_dev_id_get(dev_idx, &id);
        
          dev_idx++;
          break;
        }
    }
    


  rvs::lp::LogRecordFlush(r);
  // if log_interval timer starts
  if (log_interval) {
    timer_running.start(log_interval);
  }
  
  // worker thread has started
  while (brun) {
    rvs::lp::Log("[" + action_name + "] gm worker thread is running...",
                 rvs::logtrace);
    
    rsmi_frequencies f;
    rsmi_temperature_metric t;
    uint32_t sensor_ind = 0;
    int64_t temperature;
    int64_t speed;
    uint64_t power;
    for(dv_ind=0; dv_ind<num_devices;dv_ind++)  {
      // if dv_ind is not in map skip monitoring this device

      auto it = irq_gpu_ids.find(dv_ind);
      if (it == irq_gpu_ids.end()) {
        break;
      }

      if (bounds[GM_MEM_CLOCK].mon_metric) {
        rsmi_status_t status = rsmi_dev_gpu_clk_freq_get(dv_ind,
                                 RSMI_CLK_TYPE_MEM, &f);
            int mhz = f.current;
            met_value[dv_ind].mem_clock = mhz;
            if (!(mhz >= bounds[GM_MEM_CLOCK].min_val && mhz <=
                            bounds[GM_MEM_CLOCK].max_val) &&
                            bounds[GM_MEM_CLOCK].check_bounds) {
              // write info and increase number of violations
              msg = "[" + action_name  + "] " + MODULE_NAME + " " +
                    std::to_string(irq_gpu_ids[dv_ind].gpu_id) + " " +
                    GM_MEM_CLOCK  + " " + " bounds violation " +
                    std::to_string(mhz) + "Mhz";
              log(msg.c_str(), rvs::loginfo);
              met_violation[dv_ind].mem_clock_violation++;
              if (term) {
                if (force) {
                  // stop logging
                  rvs::lp::Stop(1);
                  // force exit
                  exit(EXIT_FAILURE);
                } else {
                  // just signal stop processing
                  rvs::lp::Stop(0);
                }
                brun = false;
                break;
              }
            }
            irq_gpu_ids[dv_ind].av_mem_clock += mhz;
        if (!brun) {
          break;
        }
        //val_vec.clear();
      }

      if (bounds[GM_CLOCK].mon_metric) {
        rsmi_status_t status = rsmi_dev_gpu_clk_freq_get(dv_ind,
                                 RSMI_CLK_TYPE_SYS, &f);
            int mhz = f.current;
            met_value[dv_ind].clock = mhz;
            if (!(mhz >= bounds[GM_CLOCK].min_val && mhz <=
                        bounds[GM_CLOCK].max_val) &&
                        bounds[GM_CLOCK].check_bounds) {
              // write info
              msg = "[" + action_name  + "] " + MODULE_NAME + " " +
                  std::to_string(irq_gpu_ids[dv_ind].gpu_id) + " " +
                  GM_CLOCK + " " + " bounds violation " +
                  std::to_string(mhz) + "Mhz";
              log(msg.c_str(), rvs::loginfo);
              met_violation[dv_ind].clock_violation++;
              if (term) {
                if (force) {
                  // stop logging
                  rvs::lp::Stop(1);
                  // force exit
                  exit(EXIT_FAILURE);
                } else {
                  // just signal stop processing
                  rvs::lp::Stop(0);
                }
                brun = false;
                break;
              }
            }
            irq_gpu_ids[dv_ind].av_clock += mhz;
        if (!brun) {
          break;
        }
      }

      if (bounds[GM_TEMP].mon_metric) {
        rsmi_status_t status = rsmi_dev_temp_metric_get(dv_ind, sensor_ind,
                        RSMI_TEMP_CURRENT, &temperature);
        if ( status == RSMI_STATUS_SUCCESS) {
          int temper = temperature/1000;
          met_value[dv_ind].temp = temper;
          irq_gpu_ids[dv_ind].av_temp += temper;
          if (!(temper >= bounds[GM_TEMP].min_val && temper <=
                        bounds[GM_TEMP].max_val) &&
                        bounds[GM_TEMP].check_bounds) {
            // write info
            msg = "[" + action_name  + "] " + MODULE_NAME + " " +
                std::to_string(irq_gpu_ids[dv_ind].gpu_id) + " " +
                + GM_TEMP + " " + " bounds violation " +
                std::to_string(temper) + "C";
            log(msg.c_str(), rvs::loginfo);
            met_violation[dv_ind].temp_violation++;
            if (term) {
              if (force) {
                // stop logging
                rvs::lp::Stop(1);
                // force exit
                exit(EXIT_FAILURE);
              } else {
                // just signal stop processing
                rvs::lp::Stop(0);
              }
              brun = false;
              break;
            }
          }
        } else {
            msg = "[" + action_name  + "] " + MODULE_NAME + " " +
            std::to_string(irq_gpu_ids[dv_ind].gpu_id) + " " +
            GM_TEMP + " Not available";
            log(msg.c_str(), rvs::loginfo);
          }
      }
 
      if (bounds[GM_FAN].mon_metric) {
        rsmi_status_t status = rsmi_dev_fan_speed_get( dv_ind,
                                         sensor_ind, &speed);
          if (status == RSMI_STATUS_SUCCESS) {
            met_value[dv_ind].fan = speed;
            irq_gpu_ids[dv_ind].av_fan += speed;
            if (!(speed >= bounds[GM_FAN].min_val && speed <=
                        bounds[GM_FAN].max_val) &&
                        bounds[GM_FAN].check_bounds) {
              // write info
              msg = "[" + action_name  + "] " + MODULE_NAME + " " +
                    std::to_string(irq_gpu_ids[dv_ind].gpu_id) + " " +
                    + GM_FAN + " " + " bounds violation " +
                    std::to_string(speed) + "%";
              log(msg.c_str(), rvs::loginfo);
              met_violation[dv_ind].fan_violation++;
              if (term) {
                if (force) {
                  // stop logging
                  rvs::lp::Stop(1);
                  // force exit
                  exit(EXIT_FAILURE);
                } else {
                  // just signal stop processing
                  rvs::lp::Stop(0);
                }
                brun = false;
                break;
              }
            }
          } else {
              msg = "[" + action_name  + "] " + MODULE_NAME + " " +
              std::to_string(irq_gpu_ids[dv_ind].gpu_id) + " " +
              GM_FAN + " Not available";
              log(msg.c_str(), rvs::loginfo);
            }
      }

       if (bounds[GM_POWER].mon_metric) {
            rsmi_status_t status = 
            rsmi_dev_power_ave_get( dv_ind, sensor_ind, &power);
            met_value[dv_ind].power = power;
            irq_gpu_ids[dv_ind].av_power += power;
            if (!(power >= bounds[GM_POWER].min_val && power <=
                        bounds[GM_POWER].max_val) &&
                        bounds[GM_POWER].check_bounds) {
              // write info
              msg = "[" + action_name  + "] " + MODULE_NAME + " " +
                    std::to_string(irq_gpu_ids[dv_ind].gpu_id) + " " +
                    GM_POWER + " " + " bounds violation " +
                    std::to_string(power) + "Watts";
              log(msg.c_str(), rvs::loginfo);
              met_violation[dv_ind].power_violation++;
              if (term) {
                if (force) {
                  // stop logging
                  rvs::lp::Stop(1);
                  // force exit
                  exit(EXIT_FAILURE);
                } else {
                  // just signal stop processing
                  rvs::lp::Stop(0);
                }
                brun = false;
                break;
              }
            }
      }
    }
    count++;
    sleep(sample_interval);
  }

  

  pci_cleanup(pacc);
  rsmi_shut_down();

  timer_running.stop();
  // get timestamp
  rvs::lp::get_ticks(&sec, &usec);

  for (auto it = irq_gpu_ids.begin();
        it != irq_gpu_ids.end(); it++) {
    // add string output
    msg = "[" + action_name + "] gm " +
        std::to_string((it->second).gpu_id) + " stopped";
    rvs::lp::Log(msg, rvs::logresults, sec, usec);
  }

  rvs::lp::Log("[" + stop_action_name + "] gm worker thread has finished",
               rvs::logdebug);
}


/**
 * @brief Stops monitoring
 *
 * Sets brun member to FALSE thus signaling end of monitoring.
 * Then it waits for std::thread to exit before returning.
 *
 * */
void Worker::stop() {
  rvs::lp::Log("[" + stop_action_name + "] gm in Worker::stop()",
               rvs::logtrace);
  string msg;
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
    for (auto it = irq_gpu_ids.begin(); it !=
            irq_gpu_ids.end(); it++) {
      if (bounds[GM_TEMP].mon_metric) {
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
      if (bounds[GM_CLOCK].mon_metric) {
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
      if (bounds[GM_MEM_CLOCK].mon_metric) {
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
      if (bounds[GM_FAN].mon_metric) {
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " + GM_FAN +" violations " +
            std::to_string(met_violation[it->first].fan_violation);
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " + GM_FAN + " average " +
            std::to_string((it->second).av_fan/count) + "%";
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
      }
      if (bounds[GM_POWER].mon_metric) {
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " +
            GM_POWER + " violations " +
            std::to_string(met_violation[it->first].power_violation);
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        rvs::lp::AddString(r, "result", msg);
        msg = "[" + action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " " + GM_POWER + " average " +
            std::to_string((it->second).av_power/count) + "Watts";
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        rvs::lp::AddString(r, "result", msg);
      }
    }
  }
    rvs::lp::LogRecordFlush(r);

  // wait a bit to make sure thread has exited
  try {
    if (t.joinable())
      t.join();
    }
  catch(...) {
  }
}
