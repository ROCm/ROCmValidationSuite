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

#include <pci/pci.h>
#include <linux/pci.h>
#include <vector>
#include <queue>
#include <string>

#include "gtest/gtest.h"

#include "include/pci_caps.h"
#include "include/rvs_unit_testing_defs.h"

using rvs::rvs_pci_read_word_return_value;
using rvs::rvs_pci_read_word;
using rvs::rvs_pci_read_long_return_value;
using rvs::rvs_pci_read_long;
using rvs::rvs_pci_get_param_return_value;
using rvs::rvs_readlink_return_value;
using rvs::rvs_readlink_buff_return_value;

class PcieCapsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dev = new pci_dev();

    test_cap[0] = new pci_cap();
    test_cap[1] = new pci_cap();

    // first cap
    test_cap[0]->id   = 17;
    test_cap[0]->type = 34;
    test_cap[0]->addr = 219;
    test_cap[0]->next = test_cap[1];
    // second cap
    test_cap[1]->id   = 25;
    test_cap[1]->type = 5;
    test_cap[1]->addr = 198;

    // fill up the struct
    test_dev->first_cap = test_cap[0];
    test_dev->bus = 9;
    test_dev->domain = 2;
    test_dev->dev = 7;
    test_dev->func = 8;
    test_dev->device_id = 54;
    test_dev->vendor_id = 98;

    test_access = new pci_access();
    test_dev->access = test_access;

    for (int k = 0; k < 6; k++) {
      test_dev->base_addr[k] = k;
      test_dev->size[k] = k;
    }
  }

  void TearDown() override {
    delete test_access;
    delete test_cap[0];
    delete test_cap[1];
    delete test_dev;
  }

  // pci device struct
  pci_dev* test_dev;

  // pci caps struct
  pci_cap* test_cap[2];

  // pci access struct
  pci_access* test_access;

  // utility function
  void get_num_bits(uint32_t input_num, uint32_t* num_ones,
                    int* first_one, uint32_t* max_value) {
    *num_ones = 0;
    *first_one = -1;
    for (uint i = 0; i < 32; i++) {
      if (((input_num >> i) & 0x1) == 1) {
        (*num_ones)++;
        if (*first_one == -1) {
          *first_one = i;
        }
      }
    }
    *max_value = (1 << *num_ones);
  }
};

TEST_F(PcieCapsTest, pcie_caps) {
  int return_value;
  char* buff;
  std::string exp_string;
  uint32_t num_ones;
  int first_one;
  uint32_t max_value;
  uint index;
  buff = new char[1024];

  get_num_bits(0xf, &num_ones, &first_one, &max_value);
  EXPECT_EQ(num_ones, 4u);
  EXPECT_EQ(first_one, 0);
  EXPECT_EQ(max_value, 16u);

  get_num_bits(0x3f0, &num_ones, &first_one, &max_value);
  EXPECT_EQ(num_ones, 6u);
  EXPECT_EQ(first_one, 4);
  EXPECT_EQ(max_value, 64u);

  rvs_pci_read_word_return_value = std::queue<u16>();
  rvs_pci_read_word_return_value.push(0x1);
  rvs_pci_read_word_return_value.push(0x2);
  rvs_pci_read_word_return_value.push(0x3);
  EXPECT_EQ(rvs_pci_read_word(test_dev, 0x0), 0x1u);
  EXPECT_EQ(rvs_pci_read_word(test_dev, 0x0), 0x2u);
  EXPECT_EQ(rvs_pci_read_word(test_dev, 0x0), 0x3u);
  EXPECT_EQ(rvs_pci_read_word(test_dev, 0x0), 0x3u);

  rvs_pci_read_long_return_value = std::queue<u32>();
  rvs_pci_read_long_return_value.push(0x1);
  rvs_pci_read_long_return_value.push(0x2);
  rvs_pci_read_long_return_value.push(0x3);
  EXPECT_EQ(rvs_pci_read_long(test_dev, 0x0), 0x1u);
  EXPECT_EQ(rvs_pci_read_long(test_dev, 0x0), 0x2u);
  EXPECT_EQ(rvs_pci_read_long(test_dev, 0x0), 0x3u);
  EXPECT_EQ(rvs_pci_read_long(test_dev, 0x0), 0x3u);

  // ---------------------------------------
  // pci_dev_find_cap_offset
  // ---------------------------------------
  // 1. null ptr
  test_dev->first_cap = NULL;
  return_value = pci_dev_find_cap_offset(test_dev, 0, 0);
  EXPECT_EQ(return_value, 0);
  test_dev->first_cap = test_cap[0];
  for (uint i = 0; i < 2; i++) {
    // 2. both correct
    return_value = pci_dev_find_cap_offset(test_dev, test_cap[i]->id,
                                           test_cap[i]->type);
    EXPECT_EQ(return_value, static_cast<int>(test_cap[i]->addr));
    // 3. one correct
    return_value = pci_dev_find_cap_offset(test_dev, test_cap[i]->id, 0);
    EXPECT_EQ(return_value, 0);
    return_value = pci_dev_find_cap_offset(test_dev, 0, test_cap[i]->type);
    EXPECT_EQ(return_value, 0);
  }
  // 4. none correct
  return_value = pci_dev_find_cap_offset(test_dev, 0, 0);
  EXPECT_EQ(return_value, 0);

  // ---------------------------------------
  // get_link_cap_max_speed
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_EXP;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_long_return_value = std::queue<u32>();
      rvs_pci_read_long_return_value.push(15);
      buff[0] = '\0';
      get_link_cap_max_speed(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        EXPECT_STREQ(buff, "Unknown speed");
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_long_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  get_num_bits(PCI_EXP_LNKCAP_SLS, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    buff[0] = '\0';
    rvs_pci_read_long_return_value = std::queue<u32>();
    rvs_pci_read_long_return_value.push(k << first_one);
    get_link_cap_max_speed(test_dev, buff);
    switch (k) {
    case 1:
      EXPECT_STREQ(buff, "2.5 GT/s");
      break;
    case 2:
      EXPECT_STREQ(buff, "5 GT/s");
      break;
    case 3:
      EXPECT_STREQ(buff, "8 GT/s");
      break;
    case 4:
      EXPECT_STREQ(buff, "16 GT/s");
      break;
    default:
      EXPECT_STREQ(buff, "Unknown speed");
      break;
    }
  }

  // ---------------------------------------
  // get_link_cap_max_width
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_EXP;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_long_return_value = std::queue<u32>();
      rvs_pci_read_long_return_value.push(15);
      buff[0] = '\0';
      get_link_cap_max_width(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_long_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  get_num_bits(PCI_EXP_LNKCAP_MLW, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_long_return_value = std::queue<u32>();
    rvs_pci_read_long_return_value.push(k << first_one);
    buff[0] = '\0';
    get_link_cap_max_width(test_dev, buff);
    exp_string = "x" + std::to_string(k);
    EXPECT_STREQ(buff, exp_string.c_str());
  }

  // ---------------------------------------
  // get_link_stat_cur_speed
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_EXP;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_word_return_value = std::queue<u16>();
      rvs_pci_read_word_return_value.push(15);
      buff[0] = '\0';
      get_link_stat_cur_speed(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        EXPECT_STREQ(buff, "Unknown speed");
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  get_num_bits(PCI_EXP_LNKSTA_CLS, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    buff[0] = '\0';
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    get_link_stat_cur_speed(test_dev, buff);
    switch (k) {
    case 1:
      EXPECT_STREQ(buff, "2.5 GT/s");
      break;
    case 2:
      EXPECT_STREQ(buff, "5 GT/s");
      break;
    case 3:
      EXPECT_STREQ(buff, "8 GT/s");
      break;
#ifdef PCI_EXP_LNKSTA_CLS_16_0GB
    case 4:
      EXPECT_STREQ(buff, "16 GT/s");
      break;
#endif
    default:
      EXPECT_STREQ(buff, "Unknown speed");
      break;
    }
  }

  // ---------------------------------------
  // get_link_stat_neg_width
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_EXP;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_word_return_value = std::queue<u16>();
      rvs_pci_read_word_return_value.push(15);
      buff[0] = '\0';
      get_link_stat_neg_width(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  get_num_bits(PCI_EXP_LNKSTA_NLW, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    buff[0] = '\0';
    get_link_stat_neg_width(test_dev, buff);
    exp_string = "x" + std::to_string(k);
    EXPECT_STREQ(buff, exp_string.c_str());
  }

  // ---------------------------------------
  // get_slot_pwr_limit_value
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_EXP;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_long_return_value = std::queue<u32>();
      rvs_pci_read_long_return_value.push(15);
      buff[0] = '\0';
      get_slot_pwr_limit_value(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_long_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  get_num_bits(PCI_EXP_SLTCAP_SPLV, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_long_return_value = std::queue<u32>();
    rvs_pci_read_long_return_value.push(k << first_one);
    buff[0] = '\0';
    get_slot_pwr_limit_value(test_dev, buff);
    if (k > 0xEF) {
      switch (k) {
      case 0xF0:
          exp_string = "250.000W";
          break;
      case 0xF1:
          exp_string = "270.000W";
          break;
      case 0xF2:
          exp_string = "300.000W";
          break;
      default:
          exp_string = "-1.000W";
      }
    } else {
      exp_string = std::to_string(k) + ".000W";
    }
    EXPECT_STREQ(buff, exp_string.c_str());
  }

  // ---------------------------------------
  // get_slot_physical_num
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_EXP;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_long_return_value = std::queue<u32>();
      rvs_pci_read_long_return_value.push(15);
      buff[0] = '\0';
      get_slot_physical_num(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_long_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  get_num_bits(PCI_EXP_SLTCAP_PSN, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_long_return_value = std::queue<u32>();
    rvs_pci_read_long_return_value.push(k << first_one);
    buff[0] = '\0';
    get_slot_physical_num(test_dev, buff);
    exp_string = "#" + std::to_string(k);
    EXPECT_STREQ(buff, exp_string.c_str());
  }

  // ---------------------------------------
  // get_pci_bus_id
  // ---------------------------------------
  get_pci_bus_id(test_dev, buff);
  exp_string = std::to_string(test_dev->bus);
  EXPECT_STREQ(buff, exp_string.c_str());
  // ---------------------------------------
  // get_device_id
  // ---------------------------------------
  get_device_id(test_dev, buff);
  exp_string = std::to_string(test_dev->device_id);
  EXPECT_STREQ(buff, exp_string.c_str());
  // ---------------------------------------
  // get_vendor_id
  // ---------------------------------------
  get_vendor_id(test_dev, buff);
  exp_string = std::to_string(test_dev->vendor_id);
  EXPECT_STREQ(buff, exp_string.c_str());

  for (uint p = 0; p < 8; p++) {
    // ---------------------------------------
    // get_pci_bus_id
    // ---------------------------------------
    test_dev->bus = p;
    get_pci_bus_id(test_dev, buff);
    exp_string = std::to_string(test_dev->bus);
    EXPECT_STREQ(buff, exp_string.c_str());
    // ---------------------------------------
    // get_device_id
    // ---------------------------------------
    test_dev->device_id = (p + 1) % 8;
    get_device_id(test_dev, buff);
    exp_string = std::to_string(test_dev->device_id);
    EXPECT_STREQ(buff, exp_string.c_str());
    // ---------------------------------------
    // get_vendor_id
    // ---------------------------------------
    test_dev->vendor_id = (p + 2) % 8;
    get_vendor_id(test_dev, buff);
    exp_string = std::to_string(test_dev->vendor_id);
    EXPECT_STREQ(buff, exp_string.c_str());
  }

  // ---------------------------------------
  // get_kernel_driver
  // ---------------------------------------
  // 1. method is not PCI_ACCESS_SYS_BUS_PCI
  buff[0] = '\0';
  exp_string = "";
  test_access->method = PCI_ACCESS_AUTO;
  get_kernel_driver(test_dev, buff);
  EXPECT_STREQ(buff, exp_string.c_str());
  // 2. method is PCI_ACCESS_SYS_BUS_PCI and base[0] = 0
  exp_string = "";
  test_access->method = PCI_ACCESS_SYS_BUS_PCI;
  rvs_pci_get_param_return_value = 0;
  get_kernel_driver(test_dev, buff);
  EXPECT_STREQ(buff, exp_string.c_str());
  // 3. method is PCI_ACCESS_SYS_BUS_PCI and base = 1
  exp_string = "";
  test_access->method = PCI_ACCESS_SYS_BUS_PCI;
  rvs_pci_get_param_return_value = const_cast<char*>("abc");
  rvs_readlink_return_value = -1;
  rvs_readlink_buff_return_value = const_cast<char*>("");
  get_kernel_driver(test_dev, buff);
  EXPECT_STREQ(buff, exp_string.c_str());
  // 4. method is PCI_ACCESS_SYS_BUS_PCI and base = 1 and n = 0
  exp_string = "";
  test_access->method = PCI_ACCESS_SYS_BUS_PCI;
  rvs_pci_get_param_return_value = const_cast<char*>("abc");
  rvs_readlink_return_value = 0;
  rvs_readlink_buff_return_value = const_cast<char*>("def");
  get_kernel_driver(test_dev, buff);
  EXPECT_STREQ(buff, exp_string.c_str());
  // 5. method is PCI_ACCESS_SYS_BUS_PCI and base = 1 and n > 0
  exp_string = "bla";
  test_access->method = PCI_ACCESS_SYS_BUS_PCI;
  rvs_pci_get_param_return_value = const_cast<char*>("abc");
  rvs_readlink_return_value = 10;
  rvs_readlink_buff_return_value = const_cast<char*>("def/bla");
  get_kernel_driver(test_dev, buff);
  EXPECT_STREQ(buff, exp_string.c_str());
  // 6. method is PCI_ACCESS_SYS_BUS_PCI and base = 1 and n > 0
  exp_string = "def";
  test_access->method = PCI_ACCESS_SYS_BUS_PCI;
  rvs_pci_get_param_return_value = const_cast<char*>("abc");
  rvs_readlink_return_value = 3;
  rvs_readlink_buff_return_value = const_cast<char*>("def/bla");
  get_kernel_driver(test_dev, buff);
  EXPECT_STREQ(buff, exp_string.c_str());

  // ---------------------------------------
  // get_dev_serial_num
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_EXT_CAP_ID_DSN;
      test_cap[(type_f+1)%2]->type = PCI_CAP_EXTENDED;
      rvs_pci_read_long_return_value = std::queue<u32>();
      rvs_pci_read_long_return_value.push(15);
      buff[0] = '\0';
      get_dev_serial_num(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_long_return_value
  test_cap[0]->id   = PCI_EXT_CAP_ID_DSN;
  test_cap[0]->type = PCI_CAP_EXTENDED;
  test_cap[1]->id   = PCI_EXT_CAP_ID_DSN;
  test_cap[1]->type = PCI_CAP_EXTENDED;
  index = 0;
  while (true) {
    uint k1, k2;
    char expect_string[1024];
    rvs_pci_read_long_return_value = std::queue<u32>();
    k1 = index;
    k2 = ~index;
    rvs_pci_read_long_return_value.push(k2);
    rvs_pci_read_long_return_value.push(k1);
    buff[0] = '\0';
    get_dev_serial_num(test_dev, buff);
    snprintf(expect_string, sizeof(expect_string),
        "%02x-%02x-%02x-%02x-%02x-%02x-%02x-%02x",
        k1 >> 24, (k1 >> 16) & 0xff, (k1 >> 8) & 0xff, k1 & 0xff,
        k2 >> 24, (k2 >> 16) & 0xff, (k2 >> 8) & 0xff, k2 & 0xff);
    EXPECT_STREQ(buff, expect_string);
    if (index > 0xffffffff - 0x01020304) {
      break;
    }
    index = index + 0x01020304;
  }

  // ---------------------------------------
  // get_pwr_budgeting
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_EXT_CAP_ID_PWR and PCI_CAP_EXTENDED are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_EXT_CAP_ID_PWR;
      test_cap[(type_f+1)%2]->type = PCI_CAP_EXTENDED;
      rvs_pci_read_long_return_value = std::queue<u32>();
      rvs_pci_read_long_return_value.push(15);
      buff[0] = '\0';
      get_pwr_budgeting(test_dev, 0x0, 0x0, 0x0, buff);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_long_return_value
  test_cap[0]->id   = PCI_EXT_CAP_ID_PWR;
  test_cap[0]->type = PCI_CAP_EXTENDED;
  test_cap[1]->id   = PCI_EXT_CAP_ID_PWR;
  test_cap[1]->type = PCI_CAP_EXTENDED;
  rvs_pci_read_word_return_value = std::queue<u16>();
  rvs_pci_read_word_return_value.push(0x0);
  buff[0] = '\0';
  get_pwr_budgeting(test_dev, 0x0, 0x0, 0x0, buff);
  EXPECT_STREQ(buff, "NOT SUPPORTED");
  // 3. valid Id / Type values and iterate rvs_pci_read_long_return_value
  test_cap[0]->id   = PCI_EXT_CAP_ID_PWR;
  test_cap[0]->type = PCI_CAP_EXTENDED;
  test_cap[1]->id   = PCI_EXT_CAP_ID_PWR;
  test_cap[1]->type = PCI_CAP_EXTENDED;
  for (int k = 1; k < 16; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k);
    rvs_pci_read_word_return_value.push(0x0);
    buff[0] = '\0';
    exp_string = std::to_string(k) + ".000W";
    get_pwr_budgeting(test_dev, 0x0, 0x0, 0x0, buff);
    EXPECT_STREQ(buff, exp_string.c_str());
  }

  // ---------------------------------------
  // get_pwr_curr_state
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_PM and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_PM;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_word_return_value = std::queue<u16>();
      rvs_pci_read_word_return_value.push(15);
      buff[0] = '\0';
      get_pwr_curr_state(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_PM;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_PM;
  test_cap[1]->type = PCI_CAP_NORMAL;
  get_num_bits(PCI_PM_CTRL_STATE_MASK, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    buff[0] = '\0';
    get_pwr_curr_state(test_dev, buff);
    exp_string = "D" + std::to_string(k);
    EXPECT_STREQ(buff, exp_string.c_str());
  }

  // ---------------------------------------
  // get_atomic_op_routing
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_EXP;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_word_return_value = std::queue<u16>();
      rvs_pci_read_word_return_value.push(15);
      buff[0] = '\0';
      get_atomic_op_routing(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  get_num_bits(PCI_EXP_FLAGS_VERS, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    rvs_pci_read_word_return_value.push((k % 2 == 0) ? 0x0 : 0x40);
    buff[0] = '\0';
    get_atomic_op_routing(test_dev, buff);
    if (k >= 2) {
      EXPECT_STREQ(buff, (k % 2 == 0) ? "FALSE" : "TRUE");
    } else {
      EXPECT_STREQ(buff, "NOT SUPPORTED");
    }
  }

  // ---------------------------------------
  // get_atomic_op_register_value
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_EXP;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_word_return_value = std::queue<u16>();
      rvs_pci_read_word_return_value.push(15);
      return_value = get_atomic_op_register_value(test_dev);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_EQ(return_value, -1);
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  for (int i = 0; i < 6; i++) {
    test_dev->base_addr[i] = PCI_BASE_ADDRESS_SPACE_IO;
  }
  get_num_bits(PCI_EXP_FLAGS_VERS, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    return_value = get_atomic_op_register_value(test_dev);
    EXPECT_EQ(return_value, -1);
  }
  // 3. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  for (int i = 0; i < 6; i++) {
    test_dev->base_addr[i] = i;
  }
  get_num_bits(PCI_EXP_FLAGS_VERS, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    rvs_pci_read_long_return_value = std::queue<u32>();
    rvs_pci_read_long_return_value.push(2*k);
    return_value = get_atomic_op_register_value(test_dev);
    if (k >= 2) {
      EXPECT_EQ(return_value, static_cast<int>(2*k));
    } else {
      EXPECT_EQ(return_value, -1);
    }
  }

  // ---------------------------------------
  // get_atomic_op_32_completer
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_EXP;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_word_return_value = std::queue<u16>();
      rvs_pci_read_word_return_value.push(15);
      buff[0] = '\0';
      get_atomic_op_32_completer(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  for (int i = 0; i < 6; i++) {
    test_dev->base_addr[i] = PCI_BASE_ADDRESS_SPACE_IO;
  }
  get_num_bits(PCI_EXP_FLAGS_VERS, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    buff[0] = '\0';
    get_atomic_op_32_completer(test_dev, buff);
    EXPECT_STREQ(buff, "NOT SUPPORTED");
  }
  // 3. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  for (int i = 0; i < 6; i++) {
    test_dev->base_addr[i] = i;
  }
  get_num_bits(PCI_EXP_FLAGS_VERS, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    rvs_pci_read_long_return_value = std::queue<u32>();
    rvs_pci_read_long_return_value.push(2*k);
    buff[0] = '\0';
    get_atomic_op_32_completer(test_dev, buff);
    if (k >= 2) {
      if (((2*k) & 0x80) == 1) {
        EXPECT_STREQ(buff, "TRUE");
      } else {
        EXPECT_STREQ(buff, "FALSE");
      }
    } else {
      EXPECT_STREQ(buff, "NOT SUPPORTED");
    }
  }

  // ---------------------------------------
  // get_atomic_op_64_completer
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_EXP;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_word_return_value = std::queue<u16>();
      rvs_pci_read_word_return_value.push(15);
      buff[0] = '\0';
      get_atomic_op_64_completer(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  for (int i = 0; i < 6; i++) {
    test_dev->base_addr[i] = PCI_BASE_ADDRESS_SPACE_IO;
  }
  get_num_bits(PCI_EXP_FLAGS_VERS, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    buff[0] = '\0';
    get_atomic_op_64_completer(test_dev, buff);
    EXPECT_STREQ(buff, "NOT SUPPORTED");
  }
  // 3. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  for (int i = 0; i < 6; i++) {
    test_dev->base_addr[i] = i;
  }
  get_num_bits(PCI_EXP_FLAGS_VERS, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    rvs_pci_read_long_return_value = std::queue<u32>();
    rvs_pci_read_long_return_value.push(2*k);
    buff[0] = '\0';
    get_atomic_op_64_completer(test_dev, buff);
    if (k >= 2) {
      if (((2*k) & 0x0100) == 1) {
        EXPECT_STREQ(buff, "TRUE");
      } else {
        EXPECT_STREQ(buff, "FALSE");
      }
    } else {
      EXPECT_STREQ(buff, "NOT SUPPORTED");
    }
  }

  // ---------------------------------------
  // get_atomic_op_128_CAS_completer
  // ---------------------------------------
  // 1. invalid Id / Type (PCI_CAP_ID_EXP and PCI_CAP_NORMAL are valid)
  for (uint id_f = 0; id_f < 2; id_f++) {
    for (uint type_f = 0; type_f < 2; type_f++) {
      test_cap[id_f]->id           = 0;
      test_cap[type_f]->type       = 0;
      test_cap[(id_f+1)%2]->id     = PCI_CAP_ID_EXP;
      test_cap[(type_f+1)%2]->type = PCI_CAP_NORMAL;
      rvs_pci_read_word_return_value = std::queue<u16>();
      rvs_pci_read_word_return_value.push(15);
      buff[0] = '\0';
      get_atomic_op_128_CAS_completer(test_dev, buff);

      if (id_f == type_f) {
        // other one is valid
        continue;
      } else {
        // none is valid
        EXPECT_STREQ(buff, "NOT SUPPORTED");
      }
    }
  }
  // 2. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  for (int i = 0; i < 6; i++) {
    test_dev->base_addr[i] = PCI_BASE_ADDRESS_SPACE_IO;
  }
  get_num_bits(PCI_EXP_FLAGS_VERS, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    buff[0] = '\0';
    get_atomic_op_128_CAS_completer(test_dev, buff);
    EXPECT_STREQ(buff, "NOT SUPPORTED");
  }
  // 3. valid Id / Type values and iterate rvs_pci_read_word_return_value
  test_cap[0]->id   = PCI_CAP_ID_EXP;
  test_cap[0]->type = PCI_CAP_NORMAL;
  test_cap[1]->id   = PCI_CAP_ID_EXP;
  test_cap[1]->type = PCI_CAP_NORMAL;
  for (int i = 0; i < 6; i++) {
    test_dev->base_addr[i] = i;
  }
  get_num_bits(PCI_EXP_FLAGS_VERS, &num_ones, &first_one, &max_value);
  for (uint k = 0; k < max_value; k++) {
    rvs_pci_read_word_return_value = std::queue<u16>();
    rvs_pci_read_word_return_value.push(k << first_one);
    rvs_pci_read_long_return_value = std::queue<u32>();
    rvs_pci_read_long_return_value.push(2*k);
    buff[0] = '\0';
    get_atomic_op_128_CAS_completer(test_dev, buff);
    if (k >= 2) {
      if (((2*k) & 0x0200) == 1) {
        EXPECT_STREQ(buff, "TRUE");
      } else {
        EXPECT_STREQ(buff, "FALSE");
      }
    } else {
      EXPECT_STREQ(buff, "NOT SUPPORTED");
    }
  }

  delete buff;
}

