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

#include <string.h>
#include <include/rvs.h>
#include <include/rvsinternal.h>
#include <include/rvsexec.h>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif

/*! \var rvs_state_t rvs_state
    \brief RVS current state
    \warning Not thread safe!
*/
rvs_state_t rvs_state = RVS_STATE_UNINITIALIZED;

/*! \var rvs_session_t rvs_session
    \brief RVS sessions details
    \warning Not thread safe!
*/
rvs_session_t rvs_session[RVS_MAX_SESSIONS];

rvs_status_t rvs_get_session_instance(unsigned int *session_idx);
rvs_status_t rvs_validate_session(rvs_session_id_t session_id, unsigned int *session_idx);
void rvs_callback(const rvs_results_t * results, int user_param);

/**
 * Initialize RVS(ROCMm Validation Suite) component. 
 * @param None 
 * @return RVS_STATUS_SUCCESS - Successfully initialized
 * @return RVS_STATUS_FAILED - Failed to initialize
 */
rvs_status_t rvs_initialize(void) {

  if (RVS_STATE_INITIALIZED == rvs_state) {
    return RVS_STATUS_INVALID_STATE;
  }

  rvs_state = RVS_STATE_INITIALIZED;
  memset(rvs_session, 0, RVS_MAX_SESSIONS * sizeof(rvs_session_t));

  return RVS_STATUS_SUCCESS;
}

/**
 * Create session for a test, benchmark or qualification routine in RVS. 
 * @param[out] session_id - Session identifier 
 * @param[in] session_callback - Session callback function handler 
 * @return RVS_STATUS_SUCCESS - Successfully created session 
 * @return RVS_STATUS_FAILED - Failed to create session
 */
rvs_status_t rvs_session_create(rvs_session_id_t *session_id, rvs_session_callback session_cb) {

  unsigned int session_idx;

  if (RVS_STATE_INITIALIZED != rvs_state) {
    return RVS_STATUS_INVALID_STATE;
  }

  if ((NULL == session_id) || (NULL == session_cb)) {
    return RVS_STATUS_INVALID_ARGUMENT;
  }

  if (RVS_STATUS_SUCCESS != rvs_get_session_instance(&session_idx)) {
    return RVS_STATUS_FAILED;
  }

  printf("session_idx -> %d \n",session_idx);

  rvs_session[session_idx].id = session_idx + 1;
  rvs_session[session_idx].state = RVS_SESSION_STATE_CREATED;
  rvs_session[session_idx].callback = session_cb;

  *session_id = rvs_session[session_idx].id;

  return RVS_STATUS_SUCCESS;
}

/**
 * Set property for a session in RVS.
 * @param[in] session_id - Session identifier 
 * @param[in] session_property - Session property of test routine
 * @return RVS_STATUS_SUCCESS - Successfully set property 
 * @return RVS_STATUS_FAILED - Failed to set property
 */
rvs_status_t rvs_session_set_property(rvs_session_id_t session_id, rvs_session_property_t *session_property) {

  unsigned int session_idx;

  if (RVS_STATE_INITIALIZED != rvs_state) {
    return RVS_STATUS_INVALID_STATE;
  }

  if (NULL == session_property) {
    return RVS_STATUS_INVALID_ARGUMENT;
  }

  if (RVS_STATUS_SUCCESS != rvs_validate_session(session_id, &session_idx)) {
    return RVS_STATUS_INVALID_SESSION;
  }

  switch(session_property->type) {

    case RVS_SESSION_TYPE_DEFAULT_CONF:
    case RVS_SESSION_TYPE_CUSTOM_CONF:
    case RVS_SESSION_TYPE_CUSTOM_ACTION:

      if(RVS_SESSION_STATE_CREATED != rvs_session[session_idx].state) {
        return RVS_STATUS_INVALID_SESSION_STATE;
      }
      rvs_session[session_idx].state = RVS_SESSION_STATE_READY;

      memcpy(&(rvs_session[session_idx].property), session_property, sizeof(rvs_session_property_t));

      break;

    default:
      return RVS_STATUS_INVALID_SESSION;
  }

  return RVS_STATUS_SUCCESS;
}

/**
 * Execute session test routine based on property set in RVS.
 * @param[in] session_id - Session identifier 
 * @return RVS_STATUS_SUCCESS - Successfully launched session 
 * @return RVS_STATUS_FAILED - Failed to launch session
 */
rvs_status_t rvs_session_execute(rvs_session_id_t session_id) {

  unsigned int session_idx;

  if (RVS_STATE_INITIALIZED != rvs_state) {
    return RVS_STATUS_INVALID_STATE;
  }

  if (RVS_STATUS_SUCCESS != rvs_validate_session(session_id, &session_idx)) {
    return RVS_STATUS_INVALID_SESSION;
  }

  if(RVS_SESSION_STATE_READY != rvs_session[session_idx].state) {
    return RVS_STATUS_INVALID_SESSION_STATE;
  }

  switch(rvs_session[session_idx].property.type) {

    case RVS_SESSION_TYPE_DEFAULT_CONF:
      {
        std::map<std::string, std::string> opt;

        std::string module[RVS_MODULE_MAX] = {
          "babel",
          "gpup",
          "gst",
          "iet",
          "mem",
          "pebb",
          "peqt",
          "pesm",
          "pbqt",
          "rcqt",
          "smqt"};

        opt.insert({"module", module[rvs_session[session_idx].property.default_conf.module]});

        rvs::exec executor;

        executor.set_callback(rvs_callback, (int)session_id);
        executor.run(opt);
      }
      break;

    case RVS_SESSION_TYPE_CUSTOM_CONF:
      {}
      break;

    case RVS_SESSION_TYPE_CUSTOM_ACTION:
      {
        std::map<std::string, std::string> opt;
        std::string action(rvs_session[session_idx].property.custom_action.config);

        opt.insert({"yaml", action});

        rvs::exec executor;

        executor.set_callback(rvs_callback, (int)session_id);
        executor.run(opt);
      }
      break;

    default:
      {
        return RVS_STATUS_INVALID_SESSION;
      }
  }

  return RVS_STATUS_SUCCESS;
}

/**
 * Destroy/Free session after completion of test routine.
 * @param[in] session_id - Session identifier
 * @return RVS_STATUS_SUCCESS - Successfully destroyed session 
 * @return RVS_STATUS_FAILED - Failed to destroy session
 */
rvs_status_t rvs_session_destroy(rvs_session_id_t session_id){

  unsigned int session_idx;

  if (RVS_STATE_INITIALIZED != rvs_state) {
    return RVS_STATUS_INVALID_STATE;
  }

  if (RVS_STATUS_SUCCESS != rvs_validate_session(session_id, &session_idx)) {
    return RVS_STATUS_INVALID_SESSION;
  }

  if(RVS_SESSION_STATE_INPROGRESS == rvs_session[session_idx].state) {
    return RVS_STATUS_INVALID_SESSION_STATE;
  }

  rvs_session[session_idx].id = 0;
  rvs_session[session_idx].state = RVS_SESSION_STATE_IDLE;
  rvs_session[session_idx].callback = NULL;
  memset(&(rvs_session[session_idx].property), 0, sizeof(rvs_session_property_t));

  return RVS_STATUS_SUCCESS;
}

/**
 * Terminate/Deinitialize RVS(ROCMm Validation Suite) component. 
 * @param None 
 * @return RVS_STATUS_SUCCESS - Successfully deinitialized
 * @return RVS_STATUS_FAILED - Failed to deinitialize
 */
rvs_status_t rvs_terminate(void) {

  if (RVS_STATE_INITIALIZED != rvs_state) {
    return RVS_STATUS_INVALID_STATE;
  }
  rvs_state = RVS_STATE_UNINITIALIZED;

  return RVS_STATUS_SUCCESS;
}

rvs_status_t rvs_get_session_instance(unsigned int *session_idx) {

  unsigned int i = 0;

  for (i = 0; i < RVS_MAX_SESSIONS; i++) {

    if(RVS_SESSION_STATE_IDLE == rvs_session[i].state) {
      *session_idx = i;
      return RVS_STATUS_SUCCESS;
    }
  }

  return RVS_STATUS_FAILED;
}

rvs_status_t rvs_validate_session(rvs_session_id_t session_id, unsigned int *session_idx) {

  unsigned int i = 0;

  for (i = 0; i < RVS_MAX_SESSIONS; i++) {

    if(session_id == rvs_session[i].id) {
      *session_idx = i;
      return RVS_STATUS_SUCCESS;
    }
  }

  return RVS_STATUS_INVALID_SESSION;
}

void rvs_callback(const rvs_results_t * results, int user_param) {

  unsigned int session_idx = 0;

  if (RVS_STATUS_SUCCESS != rvs_validate_session((rvs_session_id_t)user_param, &session_idx)) {
  }

  rvs_session[session_idx].callback((rvs_session_id_t)user_param, results);
}

#ifdef __cplusplus
}
#endif
