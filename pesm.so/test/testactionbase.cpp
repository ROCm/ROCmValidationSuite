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

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/unitactionbase.h"

TEST(actionbase, run) {
  rvs::actionbase* p = new unitactionbase;

  EXPECT_EQ(p->run(), 3);
  delete p;
}

TEST(actionbase, boolttp) {
  std::string val;
  bool        bretval;
  int         intretval;
  bool        bval;

  rvs::actionbase* p = new unitactionbase;
  assert(p);

  p->property_set("bool1", "true");
  p->property_set("bool2", "false");
  bretval = p->has_property("bool1", &val);
  EXPECT_EQ(bretval, true);
  EXPECT_EQ(val, "true");

  intretval = p->property_get("bool1", &bval);
  EXPECT_EQ(bval, true);
  EXPECT_EQ(intretval, 0);

  intretval = p->property_get("bool2", &bval);
  EXPECT_EQ(bval, false);
  EXPECT_EQ(intretval, 0);

  delete p;
}

TEST(actionbase, boolttf) {
  std::string val;
  int         intretval;
  bool        bval;

  rvs::actionbase* p = new unitactionbase;
  assert(p);

  p->property_set("bool3", "xxx");
  p->property_set("bool4", "TRUE");
  p->property_set("bool5", "FALSE");

  intretval = p->property_get("bool3", &bval);
  EXPECT_EQ(intretval, 1);

  intretval = p->property_get("bool4", &bval);
  EXPECT_EQ(intretval, 1);

  intretval = p->property_get("bool5", &bval);
  EXPECT_EQ(intretval, 1);

  intretval = p->property_get("bool6", &bval);
  EXPECT_EQ(intretval, 2);

  delete p;
}

TEST(actionbase, uintttf) {
  std::string val;
  int         intretval;
  uint64_t    intval;

  rvs::actionbase* p = new unitactionbase;
  assert(p);

  p->property_set("uint1", "123456");
  p->property_set("uint2", "abcd");
  p->property_set("uint3", "-123");

  intretval = p->property_get_int("uint1", &intval);
  EXPECT_EQ(intretval, 0);
  EXPECT_EQ(intval, 123456u);

  intretval = p->property_get_int("uint2", &intval);
  EXPECT_EQ(intretval, 1);

  intretval = p->property_get_int("uint3", &intval);
  EXPECT_EQ(intretval, 1);

  intretval = p->property_get_int("uint4", &intval);
  EXPECT_EQ(intretval, 2);

  delete p;
}

TEST(actionbase, devicetf) {
  std::string val;
  int         intretval;
  bool b_all;
  std::vector<uint16_t> dev;

  unitactionbase* p = new unitactionbase;
  assert(p);

  p->property_set("device", "26720");

  intretval = p->property_get_device();
  EXPECT_EQ(intretval, 0);
  p->test_get_device_all(&dev, &b_all);
  EXPECT_EQ(dev[0], 26720u);

  p->test_erase_property("device");
  p->property_set("device", "123 456");
  intretval = p->property_get_device();
  p->test_get_device_all(&dev, &b_all);
  EXPECT_EQ(intretval, 0);
  EXPECT_EQ(dev[0], 123u);
  EXPECT_EQ(dev[1], 456u);
  EXPECT_EQ(dev.size(), 2u);

  p->test_erase_property("device");
  p->property_set("device", "all");
  intretval = p->property_get_device();
  p->test_get_device_all(&dev, &b_all);
  EXPECT_EQ(intretval, 0);
  EXPECT_TRUE(b_all);

  p->test_erase_property("device");
  p->property_set("device", "abcd");
  intretval = p->property_get_device();
  p->test_get_device_all(&dev, &b_all);
  EXPECT_EQ(intretval, 1);
  EXPECT_FALSE(b_all);
  EXPECT_EQ(dev.size(), 0u);

  delete p;
}
