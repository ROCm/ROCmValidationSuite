/********************************************************************************
 *
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "include/rvslognodebase.h"
#include "include/rvslognodestring.h"
#include "include/rvs_unit_testing_defs.h"

class ext_lognode : public rvs::LogNodeString {
 public:
  std::string get_Name() {
    return Name;
  }

  LogNodeBase* get_Parent() {
    return const_cast<LogNodeBase*>(Parent);
  }

  rvs::eLN get_Type() {
    return Type;
  }

  std::string get_String() {
    return Value;
  }
};

class LogNodeStringTest : public ::testing::Test {
 protected:
  void SetUp() override {
    level_1 = new rvs::LogNodeString("level_1", "10", nullptr);

    level_2[0] = new rvs::LogNodeString("level_2_0", "20", level_1);
    level_2[1] = new rvs::LogNodeString("level_2_1", "21", level_1);
    level_2[2] = new rvs::LogNodeString("level_2_2", "22", level_1);

    level_3[0] = new rvs::LogNodeString("level_3_0", "30", level_2[0]);
    level_3[1] = new rvs::LogNodeString("level_3_1", "31", level_2[0]);
    level_3[2] = new rvs::LogNodeString("level_3_2", "32", level_2[1]);
    level_3[3] = new rvs::LogNodeString("level_3_3", "33", level_2[2]);
    level_3[4] = new rvs::LogNodeString("level_3_4", "34", level_2[2]);

    level_4[0] = new rvs::LogNodeString("level_4_0", "40", level_3[2]);
    level_4[1] = new rvs::LogNodeString("level_4_1", "41", level_3[4]);

    empty = new rvs::LogNodeString("empty", "-1", nullptr);
  }

  void TearDown() override {
    delete level_1;
    for (int i = 0; i < 3; i++) {
      delete level_2[i];
    }
    for (int i = 0; i < 5; i++) {
      delete level_3[i];
    }
    for (int i = 0; i < 2; i++) {
      delete level_4[i];
    }
    delete empty;
  }

  // tree structure of nodes
  rvs::LogNodeString* level_1;
  rvs::LogNodeString* level_2[3];
  rvs::LogNodeString* level_3[5];
  rvs::LogNodeString* level_4[2];
  // empty node
  rvs::LogNodeString* empty;
};

TEST_F(LogNodeStringTest, log_node_base) {
  std::string json_string;
  std::string exp_string;
  ext_lognode* node;
  ext_lognode* tmp_node;

  // ----------------------------------------
  // check empty
  // ----------------------------------------
  node = static_cast<ext_lognode*>(empty);
  EXPECT_STREQ(node->get_Name().c_str(), "empty");
  exp_string = "-1";
  EXPECT_STREQ(node->get_String().c_str(), exp_string.c_str());
  tmp_node = static_cast<ext_lognode*>(node->get_Parent());
  EXPECT_EQ(tmp_node, nullptr);
  json_string = node->ToJson("Test ");
  EXPECT_STREQ(json_string.c_str(), "\nTest \"empty\" : \"-1\"");

  // ----------------------------------------
  // check name and type for each node
  // ----------------------------------------
  node = static_cast<ext_lognode*>(level_1);
  EXPECT_STREQ(node->get_Name().c_str(), "level_1");
  EXPECT_EQ(node->get_Type(), rvs::eLN::String);
  exp_string = "10";
  EXPECT_STREQ(node->get_String().c_str(), exp_string.c_str());
  for (uint i = 0; i < 3; i++) {
    exp_string = "level_2_" + std::to_string(i);
    node = static_cast<ext_lognode*>(level_2[i]);
    EXPECT_STREQ(node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(node->get_Type(), rvs::eLN::String);
    exp_string = "2" + std::to_string(i);
    EXPECT_STREQ(node->get_String().c_str(), exp_string.c_str());
  }
  for (uint i = 0; i < 5; i++) {
    exp_string = "level_3_" + std::to_string(i);
    node = static_cast<ext_lognode*>(level_3[i]);
    EXPECT_STREQ(node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(node->get_Type(), rvs::eLN::String);
    exp_string = "3" + std::to_string(i);
    EXPECT_STREQ(node->get_String().c_str(), exp_string.c_str());
  }
  for (uint i = 0; i < 2; i++) {
    exp_string = "level_4_" + std::to_string(i);
    node = static_cast<ext_lognode*>(level_4[i]);
    EXPECT_STREQ(node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(node->get_Type(), rvs::eLN::String);
    exp_string = "4" + std::to_string(i);
    EXPECT_STREQ(node->get_String().c_str(), exp_string.c_str());
  }

  // ----------------------------------------
  // check parent nodes
  // ----------------------------------------
  // 1 parent
  node = static_cast<ext_lognode*>(level_1);
  tmp_node = static_cast<ext_lognode*>(node->get_Parent());
  EXPECT_EQ(tmp_node, nullptr);
  // 2 parents
  for (uint i = 0; i < 3; i++) {
    node = static_cast<ext_lognode*>(level_2[i]);
    tmp_node = static_cast<ext_lognode*>(node->get_Parent());
    exp_string = "level_1";
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::String);
  }
  // 3_0, 3_1 parents
  for (uint i = 0; i < 2; i++) {
    node = static_cast<ext_lognode*>(level_3[i]);
    tmp_node = static_cast<ext_lognode*>(node->get_Parent());
    exp_string = "level_2_0";
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::String);
  }
  // 3_2 parents
  for (uint i = 2; i < 3; i++) {
    node = static_cast<ext_lognode*>(level_3[i]);
    tmp_node = static_cast<ext_lognode*>(node->get_Parent());
    exp_string = "level_2_1";
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::String);
  }
  // 3_3, 3_4 parents
  for (uint i = 3; i < 5; i++) {
    node = static_cast<ext_lognode*>(level_3[i]);
    tmp_node = static_cast<ext_lognode*>(node->get_Parent());
    exp_string = "level_2_2";
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::String);
  }

  // ----------------------------------------
  // check json
  // ----------------------------------------
  json_string = level_1->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_1\" : \"10\"");

  json_string = level_2[0]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_2_0\" : \"20\"");
  json_string = level_2[1]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_2_1\" : \"21\"");
  json_string = level_2[2]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_2_2\" : \"22\"");

  json_string = level_3[0]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_3_0\" : \"30\"");
  json_string = level_3[1]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_3_1\" : \"31\"");
  json_string = level_3[2]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_3_2\" : \"32\"");
  json_string = level_3[3]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_3_3\" : \"33\"");
  json_string = level_3[4]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_3_4\" : \"34\"");

  json_string = level_4[0]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_4_0\" : \"40\"");
  json_string = level_4[1]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_4_1\" : \"41\"");
}

