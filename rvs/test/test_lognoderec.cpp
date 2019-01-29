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

#include "include/rvslognodebase.h"
#include "include/rvslognoderec.h"
#include "include/rvs_unit_testing_defs.h"

class ext_lognode : public rvs::LogNodeRec {
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

  int get_Level() {
    return Level;
  }

  int get_sec() {
    return sec;
  }

  int get_usec() {
    return usec;
  }
};

class LogNodeRecTest : public ::testing::Test {
 protected:
  void SetUp() override {
    level_1 = new rvs::LogNodeRec("level_1", 10, 1, 0, nullptr);

    level_2[0] = new rvs::LogNodeRec("level_2_0", 20, 2, 0, level_1);
    level_2[1] = new rvs::LogNodeRec("level_2_1", 21, 2, 1, level_1);
    level_2[2] = new rvs::LogNodeRec("level_2_2", 22, 2, 2, level_1);

    level_3[0] = new rvs::LogNodeRec("level_3_0", 30, 3, 0, level_2[0]);
    level_3[1] = new rvs::LogNodeRec("level_3_1", 31, 3, 1, level_2[0]);
    level_3[2] = new rvs::LogNodeRec("level_3_2", 32, 3, 2, level_2[1]);
    level_3[3] = new rvs::LogNodeRec("level_3_3", 33, 3, 3, level_2[2]);
    level_3[4] = new rvs::LogNodeRec("level_3_4", 34, 3, 4, level_2[2]);

    level_4[0] = new rvs::LogNodeRec("level_4_0", 40, 4, 0, level_3[2]);
    level_4[1] = new rvs::LogNodeRec("level_4_1", 41, 4, 1, level_3[4]);

    empty = new rvs::LogNodeRec("empty", -1, -1, -1, nullptr);
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
  rvs::LogNodeRec* level_1;
  rvs::LogNodeRec* level_2[3];
  rvs::LogNodeRec* level_3[5];
  rvs::LogNodeRec* level_4[2];
  // empty node
  rvs::LogNodeRec* empty;
};

TEST_F(LogNodeRecTest, log_node_base) {
  std::string json_string;
  std::string exp_string;
  ext_lognode* node;
  ext_lognode* tmp_node;

  // ----------------------------------------
  // check empty
  // ----------------------------------------
  node = static_cast<ext_lognode*>(empty);
  EXPECT_STREQ(node->get_Name().c_str(), "empty");
  EXPECT_EQ(node->get_Level(), static_cast<int>(-1));
  EXPECT_EQ(node->get_sec(), static_cast<int>(-1));
  EXPECT_EQ(node->get_usec(), static_cast<int>(-1));
  tmp_node = static_cast<ext_lognode*>(node->get_Parent());
  EXPECT_EQ(tmp_node, nullptr);
  json_string = node->ToJson("T ");
  EXPECT_STREQ(json_string.c_str(),
      "\nT {\nT   \"loglevel\" : -1,\nT   \"time\" : \"    -1.-1    \",\nT }");

  // ----------------------------------------
  // check name and type for each node
  // ----------------------------------------
  node = static_cast<ext_lognode*>(level_1);
  EXPECT_STREQ(node->get_Name().c_str(), "level_1");
  EXPECT_EQ(node->get_Type(), rvs::eLN::Record);
  EXPECT_EQ(node->get_Level(), static_cast<int>(10));
  EXPECT_EQ(node->get_sec(), static_cast<int>(1));
  EXPECT_EQ(node->get_usec(), static_cast<int>(0));
  for (uint i = 0; i < 3; i++) {
    exp_string = "level_2_" + std::to_string(i);
    node = static_cast<ext_lognode*>(level_2[i]);
    EXPECT_STREQ(node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(node->get_Type(), rvs::eLN::Record);
    EXPECT_EQ(node->get_Level(), static_cast<int>(20+i));
    EXPECT_EQ(node->get_sec(), static_cast<int>(2));
    EXPECT_EQ(node->get_usec(), static_cast<int>(i));
  }
  for (uint i = 0; i < 5; i++) {
    exp_string = "level_3_" + std::to_string(i);
    node = static_cast<ext_lognode*>(level_3[i]);
    EXPECT_STREQ(node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(node->get_Type(), rvs::eLN::Record);
    EXPECT_EQ(node->get_Level(), static_cast<int>(30+i));
    EXPECT_EQ(node->get_sec(), static_cast<int>(3));
    EXPECT_EQ(node->get_usec(), static_cast<int>(i));
  }
  for (uint i = 0; i < 2; i++) {
    exp_string = "level_4_" + std::to_string(i);
    node = static_cast<ext_lognode*>(level_4[i]);
    EXPECT_STREQ(node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(node->get_Type(), rvs::eLN::Record);
    EXPECT_EQ(node->get_Level(), static_cast<int>(40+i));
    EXPECT_EQ(node->get_sec(), static_cast<int>(4));
    EXPECT_EQ(node->get_usec(), static_cast<int>(i));
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
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::Record);
  }
  // 3_0, 3_1 parents
  for (uint i = 0; i < 2; i++) {
    node = static_cast<ext_lognode*>(level_3[i]);
    tmp_node = static_cast<ext_lognode*>(node->get_Parent());
    exp_string = "level_2_0";
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::Record);
  }
  // 3_2 parents
  for (uint i = 2; i < 3; i++) {
    node = static_cast<ext_lognode*>(level_3[i]);
    tmp_node = static_cast<ext_lognode*>(node->get_Parent());
    exp_string = "level_2_1";
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::Record);
  }
  // 3_3, 3_4 parents
  for (uint i = 3; i < 5; i++) {
    node = static_cast<ext_lognode*>(level_3[i]);
    tmp_node = static_cast<ext_lognode*>(node->get_Parent());
    exp_string = "level_2_2";
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::Record);
  }

  // ----------------------------------------
  // check json
  // ----------------------------------------
  json_string = level_1->ToJson();
  EXPECT_STREQ(json_string.c_str(),
               "\n{\n  \"loglevel\" : 10,\n  \"time\" : \"     1.0     \",\n}");

  json_string = level_2[0]->ToJson();
  EXPECT_STREQ(json_string.c_str(),
               "\n{\n  \"loglevel\" : 20,\n  \"time\" : \"     2.0     \",\n}");
  json_string = level_2[1]->ToJson();
  EXPECT_STREQ(json_string.c_str(),
               "\n{\n  \"loglevel\" : 21,\n  \"time\" : \"     2.1     \",\n}");
  json_string = level_2[2]->ToJson();
  EXPECT_STREQ(json_string.c_str(),
               "\n{\n  \"loglevel\" : 22,\n  \"time\" : \"     2.2     \",\n}");

  json_string = level_3[0]->ToJson();
  EXPECT_STREQ(json_string.c_str(),
               "\n{\n  \"loglevel\" : 30,\n  \"time\" : \"     3.0     \",\n}");
  json_string = level_3[1]->ToJson();
  EXPECT_STREQ(json_string.c_str(),
               "\n{\n  \"loglevel\" : 31,\n  \"time\" : \"     3.1     \",\n}");
  json_string = level_3[2]->ToJson();
  EXPECT_STREQ(json_string.c_str(),
               "\n{\n  \"loglevel\" : 32,\n  \"time\" : \"     3.2     \",\n}");
  json_string = level_3[3]->ToJson();
  EXPECT_STREQ(json_string.c_str(),
               "\n{\n  \"loglevel\" : 33,\n  \"time\" : \"     3.3     \",\n}");
  json_string = level_3[4]->ToJson();
  EXPECT_STREQ(json_string.c_str(),
               "\n{\n  \"loglevel\" : 34,\n  \"time\" : \"     3.4     \",\n}");

  json_string = level_4[0]->ToJson();
  EXPECT_STREQ(json_string.c_str(),
               "\n{\n  \"loglevel\" : 40,\n  \"time\" : \"     4.0     \",\n}");
  json_string = level_4[1]->ToJson();
  EXPECT_STREQ(json_string.c_str(),
               "\n{\n  \"loglevel\" : 41,\n  \"time\" : \"     4.1     \",\n}");
}

