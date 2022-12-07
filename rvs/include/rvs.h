/********************************************************************************
 * 
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
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

#ifndef INCLUDE_RVS_H
#define INCLUDE_RVS_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \enum rvs_status_t
 * RVS status values.
 */
typedef enum {
  RVS_STATUS_SUCCESS = 0, /*!< Success */
  RVS_STATUS_FAILED = -1, /*!< Failed */
  RVS_STATUS_INVALID_ARGUMENT = -2, /*!< Invalid argument to function */
  RVS_STATUS_INVALID_STATE = -3, /*!< Invalid RVS state */
  RVS_STATUS_INVALID_SESSION = -4, /*!< Invalid session */
  RVS_STATUS_INVALID_SESSION_STATE = -5 /*!< Invalid session state */
/*
 * Not Supported,
 * Module not found
 * */
} rvs_status_t;

/*! \enum rvs_session_state_t
 * RVS session states.
 */
typedef enum {
  RVS_SESSION_STATE_IDLE = 0, /*!< Session idle (free session) */
  RVS_SESSION_STATE_CREATED, /*!< Session created */
  RVS_SESSION_STATE_READY, /*!< Session ready for execution */
  RVS_SESSION_STATE_STARTED, /*!< Session launched */
  RVS_SESSION_STATE_INPROGRESS, /*!< Session is in progress */
  RVS_SESSION_STATE_COMPLETED /*!< Session completed execution */
} rvs_session_state_t;

/*!
 * RVS session identifier
 */
typedef unsigned int rvs_session_id_t;

/*! \enum rvs_session_type_t
 * Types of session supported.
 */
typedef enum {
  RVS_SESSION_TYPE_DEFAULT_CONF, /*!< Session uses default RVS configuration */
  RVS_SESSION_TYPE_CUSTOM_CONF, /*!< Session uses application custom configuration */
  RVS_SESSION_TYPE_CUSTOM_ACTION /*!< Session uses application custom action */
} rvs_session_type_t;

/*! \enum rvs_module_t
 * Modules present in RVS.
 */
typedef enum {
  RVS_MODULE_BABEL = 0, /*!< Memory stress Test */
  RVS_MODULE_GPUP, /*!< GPU Properties */
  RVS_MODULE_GST, /*!< GPU Stress Test */
  RVS_MODULE_IET, /*!< Input EDPp Test */ 
  RVS_MODULE_MEM, /*!< Memory Test */
  RVS_MODULE_PEBB, /*!< PCI Express Bandwidth Benchmark */
  RVS_MODULE_PEQT, /*!< PCI Express Qualification Tool */
  RVS_MODULE_PESM, /*!< PCI Express State Monitor */
  RVS_MODULE_PBQT, /*!< P2P Benchmark and Qualification Tool */
  RVS_MODULE_RCQT, /*!< ROCm Configuration Qualification Tool */
  RVS_MODULE_SMQT, /*!< SBIOS Mapping Qualifications Tool */
  RVS_MODULE_MAX /*!< No. of RVS modules */
} rvs_module_t;

 
/*! \struct rvs_session_default_conf_t
 *  \brief Session structure for default configuration parameters.
 */
typedef struct {
  rvs_module_t module; /*!< RVS module's default RVS configuration */
} rvs_session_default_conf_t;

/*! \struct rvs_session_custom_conf_t
 *  \brief Session structure for custom configuration parameters.
 */
typedef struct {
  const char * path; /*!< Application custom configuration file full path */
} rvs_session_custom_conf_t;

/*! \struct rvs_session_custom_action_t
 *  \brief Session structure for custom action parameters.
 */
typedef struct {
  const char * config; /*!< Application custom action configuration data (yaml format) */
} rvs_session_custom_action_t;

/*! \struct rvs_session_property_t
 *  \brief Structure for all session properties.
 */
typedef struct {
  rvs_session_type_t type; /*!< Type of session */
  union {
    rvs_session_default_conf_t default_conf; /*!< Default configuration */
    rvs_session_custom_conf_t custom_conf; /*!< Custom configuration */
    rvs_session_custom_action_t custom_action; /*!< Custom action configuration */
  };
} rvs_session_property_t;

/*! \struct rvs_results_t
 *  \brief Structure for session output results.
 */
typedef struct {
  rvs_status_t status; /*!< Result status */
  rvs_session_state_t state; /*!< Session state */
  const char *output_log; /* Output result log */
} rvs_results_t;

/*!
 * RVS session callback function pointer 
 */
typedef void (*rvs_session_callback) (rvs_session_id_t session_id, const rvs_results_t *results);

/**
 * Initialize RVS(ROCMm Validation Suite) component. 
 * @param None 
 * @return RVS_STATUS_SUCCESS - Successfully initialized
 * @return RVS_STATUS_FAILED - Failed to initialize
 */
rvs_status_t rvs_initialize(void);

/**
 * Create session for a test, benchmark or qualification routine in RVS. 
 * @param[out] session_id - Session identifier 
 * @param[in] session_callback - Session callback function handler 
 * @return RVS_STATUS_SUCCESS - Successfully created session 
 * @return RVS_STATUS_FAILED - Failed to create session
 */
rvs_status_t rvs_session_create(rvs_session_id_t *session_id, rvs_session_callback session_cb);

/**
 * Set property for a session in RVS.
 * @param[in] session_id - Session identifier 
 * @param[in] session_property - Session property of test routine
 * @return RVS_STATUS_SUCCESS - Successfully set property 
 * @return RVS_STATUS_FAILED - Failed to set property
 */
rvs_status_t rvs_session_set_property(rvs_session_id_t session_id, rvs_session_property_t *session_property);

/**
 * Execute session test routine based on property set in RVS.
 * @param[in] session_id - Session identifier 
 * @return RVS_STATUS_SUCCESS - Successfully launched session 
 * @return RVS_STATUS_FAILED - Failed to launch session
 */
rvs_status_t rvs_session_execute(rvs_session_id_t session_id);

/**
 * Destroy/Free session after completion of test routine.
 * @param[in] session_id - Session identifier
 * @return RVS_STATUS_SUCCESS - Successfully destroyed session 
 * @return RVS_STATUS_FAILED - Failed to destroy session
 */
rvs_status_t rvs_session_destroy(rvs_session_id_t session_id);

/**
 * Terminate/Deinitialize RVS(ROCMm Validation Suite) component. 
 * @param None 
 * @return RVS_STATUS_SUCCESS - Successfully deinitialized
 * @return RVS_STATUS_FAILED - Failed to deinitialize
 */
rvs_status_t rvs_terminate(void);

#ifdef __cplusplus
}
#endif

#endif
