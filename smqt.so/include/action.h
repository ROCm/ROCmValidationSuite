/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
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
#ifndef SMQT_SO_INCLUDE_ACTION_H_
#define SMQT_SO_INCLUDE_ACTION_H_

#include <string>
#include "include/rvsactionbase.h"

/**
 * @class smqt_action
 * @ingroup SMQT
 *
 * @brief SMQT action implementation class
 *
 * Derives from rvs::actionbase and implements actual action functionality
 * in its run() method.
 *
 */
class smqt_action : public rvs::actionbase {
 public:
    smqt_action();
    virtual ~smqt_action();
    virtual int run(void);

 private:
    ulong  get_property(std::string);
    std::string pretty_print(ulong, uint16_t, std::string, std::string);
    bool get_all_common_config_keys();
    bool get_all_smqt_config_keys();
    std::string action_name;

 protected:
    //! specified device_id
    uint16_t dev_id;
    //! actual BAR1 size
    ulong bar1_size;
    //! actual BAR2 size
    ulong bar2_size;
    //! actual BAR4 size
    ulong bar4_size;
    //! actual BAR5 size
    ulong bar5_size;
    //! actual BAR1 address
    ulong bar1_base_addr;
    //! actual BAR2 address
    ulong bar2_base_addr;
    //! actual BAR4 address
    ulong bar4_base_addr;

#ifdef  RVS_UNIT_TEST

 protected:
  virtual void on_set_device_gpu_id() = 0;
  virtual void on_bar_data_read() = 0;
#endif
};

#endif  // SMQT_SO_INCLUDE_ACTION_H_
