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
#include "include/rvslognode.h"
#include "include/rvs_unit_testing_defs.h"

class ext_lognode : public rvs::LogNode {
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
};

class LogNodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    level_1 = new rvs::LogNode("level_1", nullptr);

    level_2[0] = new rvs::LogNode("level_2_0", level_1);
    level_2[1] = new rvs::LogNode("level_2_1", level_1);
    level_2[2] = new rvs::LogNode("level_2_2", level_1);

    level_3[0] = new rvs::LogNode("level_3_0", level_2[0]);
    level_3[1] = new rvs::LogNode("level_3_1", level_2[0]);
    level_3[2] = new rvs::LogNode("level_3_2", level_2[1]);
    level_3[3] = new rvs::LogNode("level_3_3", level_2[2]);
    level_3[4] = new rvs::LogNode("level_3_4", level_2[2]);

    level_4[0] = new rvs::LogNode("level_4_0", level_3[2]);
    level_4[1] = new rvs::LogNode("level_4_1", level_3[4]);

    empty = new rvs::LogNode("empty", nullptr);

    level_1->Add(level_2[0]);
    level_1->Add(level_2[1]);
    level_1->Add(level_2[2]);

    level_2[0]->Add(level_3[0]);
    level_2[0]->Add(level_3[1]);

    level_2[1]->Add(level_3[2]);

    level_2[2]->Add(level_3[3]);
    level_2[2]->Add(level_3[4]);

    level_3[2]->Add(level_4[0]);
    level_3[4]->Add(level_4[1]);
  }

  void TearDown() override {
    delete level_1;
    delete empty;
  }

  // tree structure of nodes
  rvs::LogNode* level_1;
  rvs::LogNode* level_2[3];
  rvs::LogNode* level_3[5];
  rvs::LogNode* level_4[2];
  // empty node
  rvs::LogNode* empty;
};

TEST_F(LogNodeTest, log_node_base) {
  std::string json_string;
  std::string exp_string;
  ext_lognode* node;
  ext_lognode* tmp_node;

  // ----------------------------------------
  // check empty
  // ----------------------------------------
  node = static_cast<ext_lognode*>(empty);
  EXPECT_STREQ(node->get_Name().c_str(), "empty");
  EXPECT_EQ(node->Child.size(), 0u);
  tmp_node = static_cast<ext_lognode*>(node->get_Parent());
  EXPECT_EQ(tmp_node, nullptr);
  json_string = node->ToJson("Test ");
  EXPECT_STREQ(json_string.c_str(), "\nTest \"empty\" : {\nTest }");

  // ----------------------------------------
  // check name and type for each node
  // ----------------------------------------
  node = static_cast<ext_lognode*>(level_1);
  EXPECT_STREQ(node->get_Name().c_str(), "level_1");
  EXPECT_EQ(node->get_Type(), rvs::eLN::List);
  for (uint i = 0; i < 3; i++) {
    exp_string = "level_2_" + std::to_string(i);
    node = static_cast<ext_lognode*>(level_2[i]);
    EXPECT_STREQ(node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(node->get_Type(), rvs::eLN::List);
  }
  for (uint i = 0; i < 5; i++) {
    exp_string = "level_3_" + std::to_string(i);
    node = static_cast<ext_lognode*>(level_3[i]);
    EXPECT_STREQ(node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(node->get_Type(), rvs::eLN::List);
  }
  for (uint i = 0; i < 2; i++) {
    exp_string = "level_4_" + std::to_string(i);
    node = static_cast<ext_lognode*>(level_4[i]);
    EXPECT_STREQ(node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(node->get_Type(), rvs::eLN::List);
  }

  // ----------------------------------------
  // check child nodes
  // ----------------------------------------
  // 1 has 2_0, 2_1, 2_2
  node = static_cast<ext_lognode*>(level_1);
  for (uint i = 0; i < 3; i++) {
    exp_string = "level_2_" + std::to_string(i);
    tmp_node = static_cast<ext_lognode*>(node->Child[i]);
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::List);
  }
  // 2_0 has 3_0, 3_1
  node = static_cast<ext_lognode*>(level_2[0]);
  for (uint i = 0; i < 2; i++) {
    exp_string = "level_3_" + std::to_string(i);
    tmp_node = static_cast<ext_lognode*>(node->Child[i]);
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::List);
  }
  // 2_1 has 3_2
  node = static_cast<ext_lognode*>(level_2[1]);
  for (uint i = 0; i < 1; i++) {
    exp_string = "level_3_" + std::to_string(i+2);
    tmp_node = static_cast<ext_lognode*>(node->Child[i]);
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::List);
  }
  // 2_2 has 3_3, 3_4
  node = static_cast<ext_lognode*>(level_2[2]);
  for (uint i = 0; i < 2; i++) {
    exp_string = "level_3_" + std::to_string(i+3);
    tmp_node = static_cast<ext_lognode*>(node->Child[i]);
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::List);
  }
  // 3_2 has 4_0
  node = static_cast<ext_lognode*>(level_3[2]);
  for (uint i = 0; i < 1; i++) {
    exp_string = "level_4_" + std::to_string(i);
    tmp_node = static_cast<ext_lognode*>(node->Child[i]);
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::List);
  }
  // 3_4 has 4_1
  node = static_cast<ext_lognode*>(level_3[4]);
  for (uint i = 0; i < 1; i++) {
    exp_string = "level_4_" + std::to_string(i+1);
    tmp_node = static_cast<ext_lognode*>(node->Child[i]);
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::List);
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
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::List);
  }
  // 3_0, 3_1 parents
  for (uint i = 0; i < 2; i++) {
    node = static_cast<ext_lognode*>(level_3[i]);
    tmp_node = static_cast<ext_lognode*>(node->get_Parent());
    exp_string = "level_2_0";
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::List);
  }
  // 3_2 parents
  for (uint i = 2; i < 3; i++) {
    node = static_cast<ext_lognode*>(level_3[i]);
    tmp_node = static_cast<ext_lognode*>(node->get_Parent());
    exp_string = "level_2_1";
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::List);
  }
  // 3_3, 3_4 parents
  for (uint i = 3; i < 5; i++) {
    node = static_cast<ext_lognode*>(level_3[i]);
    tmp_node = static_cast<ext_lognode*>(node->get_Parent());
    exp_string = "level_2_2";
    EXPECT_STREQ(tmp_node->get_Name().c_str(), exp_string.c_str());
    EXPECT_EQ(tmp_node->get_Type(), rvs::eLN::List);
  }

  // ----------------------------------------
  // check json
  // ----------------------------------------
  json_string = level_1->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_1\" : {\n  \"level_2_0\" : {\n"
  "    \"level_3_0\" : {\n    },\n    \"level_3_1\" : {\n    }\n  },\n"
  "  \"level_2_1\" : {\n    \"level_3_2\" : {\n      \"level_4_0\" : {\n      }"
  "\n    }\n  },\n  \"level_2_2\" : {\n    \"level_3_3\" : {\n    },\n"
  "    \"level_3_4\" : {\n      \"level_4_1\" : {\n      }\n    }\n  }\n}");

  json_string = level_2[0]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_2_0\" : {\n  \"level_3_0\" : {\n"
                                           "  },\n  \"level_3_1\" : {\n  }\n}");
  json_string = level_2[1]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_2_1\" : {\n  \"level_3_2\" : {\n"
                                        "    \"level_4_0\" : {\n    }\n  }\n}");
  json_string = level_2[2]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_2_2\" : {\n  \"level_3_3\" : {\n"
             "  },\n  \"level_3_4\" : {\n    \"level_4_1\" : {\n    }\n  }\n}");

  json_string = level_3[0]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_3_0\" : {\n}");
  json_string = level_3[1]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_3_1\" : {\n}");
  json_string = level_3[2]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_3_2\" : {\n  \"level_4_0\" : {\n"
                                                                      "  }\n}");
  json_string = level_3[3]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_3_3\" : {\n}");
  json_string = level_3[4]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_3_4\" : {\n  \"level_4_1\" : {\n"
                                                                      "  }\n}");

  json_string = level_4[0]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_4_0\" : {\n}");
  json_string = level_4[1]->ToJson();
  EXPECT_STREQ(json_string.c_str(), "\n\"level_4_1\" : {\n}");
}

