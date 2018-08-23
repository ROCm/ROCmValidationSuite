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

#include "rvsliblogger.h"
#include "rvs_module.h"
#include "pci_caps.h"
#include "gpu_util.h"
#include "rvs_util.h"
#include "rvsloglp.h"

using std::string;
using std::vector;
using std::map;
using std::fstream;

#define PCI_ALLOC_ERROR               "pci_alloc() error"
#define GM_RESULT_FAIL_MESSAGE        "FALSE"
#define IRQ_PATH_MAX_LENGTH           256
#define MODULE_NAME                   "gm"

// collection of allowed metrics
const char* metric_names[] =
        { "temp", "clock", "mem_clock", "fan", "power"
        };

/**
 * @brief call-back function to append to a vector of Devices
 * @param d represent device
 * @param p pointer
 * @return true if dev connected to monitor, false otherwise
 */ 
static bool GetMonitorDevices(const std::shared_ptr<amd::smi::Device> &d,
            void *p) {
  std::string val_str;
  assert(p != nullptr);

  std::vector<std::shared_ptr<amd::smi::Device>> *device_list =
    reinterpret_cast<std::vector<std::shared_ptr<amd::smi::Device>> *>(p);

  if (d->monitor() != nullptr) {
    device_list->push_back(d);
  }
  return false;
}

Worker::Worker() {
  bfiltergpu = false;
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
void Worker::set_gpuids(const std::vector<int>& GpuIds) {
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
 * @brief gets irq value for device
 * @param path represents path of device
 * @return irq value
 */
const std::string Worker::get_irq(const std::string path) {
    std::ifstream f_id;
    string irq;
    f_id.open(path);

    f_id >> irq;
    f_id.close();
    return irq;
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

  string msg;
  int ret;

  amd::smi::RocmSMI hw;
  std::vector<std::shared_ptr<amd::smi::Device>> monitor_devices;

  struct pci_access *pacc;
  struct pci_dev *dev_pci;

  unsigned int sec;
  unsigned int usec;
  void* r;

  // get timestamp
  rvs::lp::get_ticks(sec, usec);

  // DiscoverDevices() will seach for devices and monitors and update internal
  // data structures.
  hw.DiscoverDevices();

  // IterateSMIDevices will iterate through all the known devices and apply
  // the provided call-back to each device found.
  hw.IterateSMIDevices(GetMonitorDevices,
      reinterpret_cast<void *>(&monitor_devices));

  // add string output
  msg = "[" + action_name + "] gm " + strgpuids + " started";
  rvs::lp::Log(msg, rvs::logresults, sec, usec);

  // add JSON output
  r = rvs::lp::LogRecordCreate("gm", action_name.c_str(), rvs::logresults,
                               sec, usec);
  rvs::lp::AddString(r, "msg", "started");
  rvs::lp::AddString(r, "device", strgpuids);
  rvs::lp::LogRecordFlush(r);

  // get the pci_access structure
  pacc = pci_alloc();
  // initialize the PCI library
  pci_init(pacc);
  // get the list of devices
  pci_scan_bus(pacc);

  for (auto dev : monitor_devices) {
    // get irq of device
    snprintf(path, IRQ_PATH_MAX_LENGTH, "%s/device/irq", (dev->path()).c_str());
    val_str = get_irq(path);

    // iterate over devices
    for (dev_pci = pacc->devices; dev_pci; dev_pci = dev_pci->next) {
      // fill in the info
      pci_fill_info(dev_pci,
              PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS
              | PCI_FILL_EXT_CAPS | PCI_FILL_CAPS | PCI_FILL_PHYS_SLOT);

      // computes the actual dev's location_id (sysfs entry)
      uint16_t dev_location_id = ((((uint16_t) (dev_pci->bus)) << 8)
                 | (dev_pci->func));

      if (dev_pci->irq == std::stoi(val_str)) {
        uint32_t gpu_id = rvs::gpulist::GetGpuId(dev_location_id);
        if (std::find(gpuids.begin(), gpuids.end(),
                    gpu_id) !=gpuids.end()) {
          msg = "[" + action_name + "] gm " + std::to_string(gpu_id) +
                " started";
          rvs::lp::Log(msg, rvs::logresults, sec, usec);

          auto metric_length = std::end(metric_names) -
                          std::begin(metric_names);
          for (int i = 0; i < metric_length; i++) {
            if (bounds[metric_names[i]].mon_metric) {
              msg = action_name + " " + MODULE_NAME + " " +
                  std::to_string(gpu_id) + " " + " monitoring " +
                  metric_names[i];
              if (bounds[metric_names[i]].check_bounds) {
                            msg+= " bounds min:" +
                            std::to_string(bounds[metric_names[i]].min_val) +
                            " max:" + std::to_string
                            (bounds[metric_names[i]].max_val);
              }
              log(msg.c_str(), rvs::loginfo);
            }
          }

          irq_gpu_ids.insert(std::pair<string, Dev_metrics>
                (val_str, {gpu_id, 0, 0, 0, 0, 0}));
          met_violation.insert(std::pair<string, Metric_violation>
                (val_str, {0, 0, 0, 0, 0}));

          count = 0;
          break;
        }
      }
    }
  }
  // worker thread has started
  while (brun) {
    rvs::lp::Log("[" + action_name + "] gm worker thread is running...",
                 rvs::logtrace);

    for (auto dev : monitor_devices) {
      // if irq is not in map skip monitoring this device
      snprintf(path, IRQ_PATH_MAX_LENGTH, "%s/device/irq",
               (dev->path()).c_str());
      val_str = get_irq(path);

      auto it = irq_gpu_ids.find(val_str);
      if (it == irq_gpu_ids.end()) {
        break;
      }

      if (bounds["mem_clock"].mon_metric) {
        dev->readDevInfo(amd::smi::kDevGPUMClk, &val_vec);
        for (auto vs : val_vec) {
          size_t cur_pos = vs.find('*');

          if (cur_pos != string::npos) {
            size_t cur_pos = vs.find("Mhz");
            string token = vs.substr(2, cur_pos-2);
            int mhz = stoi(token);
            if (!(mhz >= bounds["mem_clock"].min_val && mhz <=
                            bounds["mem_clock"].max_val) &&
                            bounds["mem_clock"].check_bounds) {
              // write info and increase number of violations
              msg = action_name + " " + MODULE_NAME + " " +
                    std::to_string(irq_gpu_ids[val_str].gpu_id) + " " +
                    "mem_clock " + " bounds violation " +
                    std::to_string(mhz) + "Mhz";
              log(msg.c_str(), rvs::loginfo);
              met_violation[val_str].mem_clock_violation++;
              if (term) {
                // exit
                brun = false;
                break;
              }
            }
            irq_gpu_ids[val_str].av_mem_clock += mhz;
          }
        }
        if (!brun) {
          break;
        }
        val_vec.clear();
      }

      if (bounds["clock"].mon_metric) {
        dev->readDevInfo(amd::smi::kDevGPUSClk, &val_vec);
        for (auto vs : val_vec) {
          size_t cur_pos = vs.find('*');
          if (cur_pos != string::npos) {
            size_t cur_pos = vs.find("Mhz");
            string token = vs.substr(2, cur_pos-2);
            int mhz = stoi(token);
            if (!(mhz >= bounds["clock"].min_val && mhz <=
                        bounds["clock"].max_val) &&
                        bounds["clock"].check_bounds) {
              // write info
              msg = action_name + " " + MODULE_NAME + " " +
                  std::to_string(irq_gpu_ids[val_str].gpu_id) + " " +
                  "clock " + " bounds violation " +
                  std::to_string(mhz) + "Mhz";
              log(msg.c_str(), rvs::loginfo);
              met_violation[val_str].clock_violation++;
              if (term) {
                // exit
                brun = false;
                break;
              }
            }
            irq_gpu_ids[val_str].av_clock += mhz;
          }
        }
        val_vec.clear();
        if (!brun) {
          break;
        }
      }

      if (bounds["temp"].mon_metric) {
        ret = dev->monitor()->readMonitor(amd::smi::kMonTemp, &value);
        if (ret != -1) {
          int temper = (static_cast<float>(value)/1000.0);
          irq_gpu_ids[val_str].av_temp += temper;
          if (!(temper >= bounds["temp"].min_val && temper <=
                        bounds["temp"].max_val) &&
                        bounds["temp"].check_bounds) {
            // write info
            msg = action_name + " " + MODULE_NAME + " " +
                std::to_string(irq_gpu_ids[val_str].gpu_id) + " " +
                "temp " + " bounds violation " +
                std::to_string(temper) + "C";
            log(msg.c_str(), rvs::loginfo);
            met_violation[val_str].temp_violation++;
            if (term) {
              // exit
              brun = false;
              break;
            }
          }
        } else {
            std::cout << "Not available" << std::endl;
          }
      }

      if (bounds["fan"].mon_metric) {
        ret = dev->monitor()->readMonitor(amd::smi::kMonMaxFanSpeed,
                    &value);
        if (ret == 0) {
          ret = dev->monitor()->readMonitor(amd::smi::kMonFanSpeed,
                        &value2);
          if (ret != -1) {
            int fan = value2/static_cast<float>(value) * 100;
            irq_gpu_ids[val_str].av_fan += fan;
            if (!(fan >= bounds["fan"].min_val && fan <=
                        bounds["fan"].max_val) &&
                        bounds["fan"].check_bounds) {
              // write info
              msg = action_name + " " + MODULE_NAME + " " +
                    std::to_string(irq_gpu_ids[val_str].gpu_id) + " " +
                    "fan " + " bounds violation " +
                    std::to_string(fan) + "%";
              log(msg.c_str(), rvs::loginfo);
              met_violation[val_str].fan_violation++;
              if (term) {
                // exit
                brun = false;
                break;
              }
            }
          } else {
              std::cout << "Not available" << std::endl;
            }
        }
      }
    }
    count++;
    sleep(sample_interval);
  }

  pci_cleanup(pacc);
  // get timestamp
  rvs::lp::get_ticks(sec, usec);

  for (map<string, Dev_metrics>::iterator it = irq_gpu_ids.begin();
        it != irq_gpu_ids.end(); it++) {
    // add string output
    msg = "[" + stop_action_name + "] gm " +
        std::to_string((it->second).gpu_id) + " stopped";
    rvs::lp::Log(msg, rvs::logresults, sec, usec);
  }

  // add JSON output
  r = rvs::lp::LogRecordCreate("GM", stop_action_name.c_str(),
                            rvs::logresults, sec, usec);
  rvs::lp::AddString(r, "msg", "stopped");
  rvs::lp::LogRecordFlush(r);

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
  // get timestamp
  rvs::lp::get_ticks(sec, usec);
  // reset "run" flag
  brun = false;
  // (give thread chance to finish processing and exit)
  sleep(200);

  if (count != 0) {
    for (map<string, Dev_metrics>::iterator it = irq_gpu_ids.begin(); it !=
            irq_gpu_ids.end(); it++) {
      if (bounds["temp"].mon_metric) {
        msg = "[" + stop_action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " temp " + " violations " +
            std::to_string(met_violation[it->first].temp_violation);
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        msg = "[" + stop_action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " temp " + " average " +
            std::to_string((it->second).av_temp/count) + "C";
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
      }
      if (bounds["clock"].mon_metric) {
        msg = "[" + stop_action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " clock " + " violations " +
            std::to_string(met_violation[it->first].clock_violation);
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        msg = "[" + stop_action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " clock " + " average " +
            std::to_string((it->second).av_clock/count) + "Mhz";
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
      }
      if (bounds["mem_clock"].mon_metric) {
        msg = "[" + stop_action_name + "] gm " +
            std::to_string((it->second).gpu_id) +
            " mem_clock " + " violations " +
            std::to_string(met_violation[it->first].mem_clock_violation);
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        msg = "[" + stop_action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " mem_clock " + " average " +
            std::to_string((it->second).av_mem_clock/count) + "Mhz";
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
      }
      if (bounds["fan"].mon_metric) {
        msg = "[" + stop_action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " fan " + " violations " +
            std::to_string(met_violation[it->first].fan_violation);
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
        msg = "[" + stop_action_name + "] gm " +
            std::to_string((it->second).gpu_id) + " fan " + " average " +
            std::to_string((it->second).av_fan/count) + "%";
        rvs::lp::Log(msg, rvs::logresults, sec, usec);
      }
    }
  }

  // wait a bit to make sure thread has exited
  try {
    if (t.joinable())
      t.join();
    }
  catch(...) {
  }
}
