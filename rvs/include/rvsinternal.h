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

#include <include/rvs.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \def RVS_MAX_SESSIONS
 * Maximum session supported in RVS at once.
 */
#define RVS_MAX_SESSIONS 1

/*! \enum rvs_state_t
 * RVS states.
 */
typedef enum {
  RVS_STATE_INITIALIZED, /*!< RVS initialized state */
  RVS_STATE_UNINITIALIZED /*!< RVS uninitialized state */
} rvs_state_t;

/*! \struct rvs_session_t
 *  \brief RVS session parameters.
 */
typedef struct rvs_session_ {
  rvs_session_id_t id;/*!< Unique session id */
  rvs_session_state_t state;/*!< Current session state */
  rvs_session_callback callback;/*!< Session callback */
  rvs_session_property_t property;/*!< Session property */
} rvs_session_t;

#ifdef __cplusplus
}
#endif
