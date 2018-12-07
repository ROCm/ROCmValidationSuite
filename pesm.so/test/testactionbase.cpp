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
