/********************************************************************************
 *
 * Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef INCLUDE_RVS_KEY_DEF_H_
#define INCLUDE_RVS_KEY_DEF_H_


#define RVS_CONF_NAME_KEY               "name"
#define RVS_CONF_DEVICE_KEY             "device"
#define RVS_CONF_PARALLEL_KEY           "parallel"
#define RVS_CONF_COUNT_KEY              "count"
#define RVS_CONF_WAIT_KEY               "wait"
#define RVS_CONF_DURATION_KEY           "duration"
#define RVS_CONF_DEVICEID_KEY           "deviceid"
#define RVS_CONF_DEVICE_INDEX_KEY       "device_index"
#define RVS_CONF_SAMPLE_INTERVAL_KEY    "sample_interval"
#define RVS_CONF_LOG_INTERVAL_KEY       "log_interval"
#define RVS_CONF_TERMINATE_KEY          "terminate"
#define RVS_CONF_LOG_LEVEL_KEY          "cli.-d"
#define RVS_CONF_BLOCK_SIZE_KEY         "block_size"
#define RVS_CONF_B2B_BLOCK_SIZE_KEY     "b2b_block_size"
#define RVS_CONF_LINK_TYPE_KEY          "link_type"
#define RVS_CONF_MONITOR_KEY            "monitor"
#define RVS_CONF_HOT_CALLS_KEY          "hot_calls"
#define RVS_CONF_WARM_CALLS_KEY         "warm_calls"
#define RVS_CONF_B2B_KEY                "b2b"
#define RVS_CONF_EXECUTOR_KEY           "executor"
#define RVS_CONF_SUBEXECUTOR_KEY        "subexecutor"
#define RVS_CONF_TRANSFER_METHOD_KEY    "transfer_method"

#define DEFAULT_LOG_INTERVAL (1000u)
#define DEFAULT_DURATION     (10000u)
#define DEFAULT_COUNT        (1u)
#define DEFAULT_WAIT         (500u)
#define DEFAULT_HOT_CALLS    (1u)
#define DEFAULT_WARM_CALLS   (1u)
#define DEFAULT_B2B          false
#define DEFAULT_TRANSFER_METHOD "native"
#define DEFAULT_EXECUTOR     "gpu"
#define DEFAULT_SUBEXECUTOR  (2u)

#define YAML_DEVICE_PROPERTY_ERROR      "Error while parsing <device> property"
#define YAML_DEVICEID_PROPERTY_ERROR    "Error while parsing <deviceid> "\
                                        "property"
#define YAML_TARGET_STRESS_PROP_ERROR   "Error while parsing <target_stress> "\
                                        "property"
#define YAML_REGULAR_EXPRESSION_ERROR   "Regular expression error"
#define YAML_DEVICE_PROP_DELIMITER      " "

#define RVS_JSON_LOG_GPU_ID_KEY         "gpu_id"
#define RVS_JSON_LOG_GPU_IDX_KEY        "gpu_index"

#endif  // INCLUDE_RVS_KEY_DEF_H_
