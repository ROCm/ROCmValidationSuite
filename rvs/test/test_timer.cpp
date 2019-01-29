/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without result_idtriction, including without limitation the rights to
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

#include "gtest/gtest.h"

#include "include/rvs_unit_testing_defs.h"
#include "test/ext_timer.h"

class test_action {
 public:
  void clear_all() {
    clear_run();
    clear_final();
  }
  // -------------------------------------------
  // Running
  // -------------------------------------------
  // indicator whether function is done
  int action_run_done = 0;
  // clear action
  void clear_run() {
    action_run_done = 0;
  }
  // calback
  void test_run_increment(void) {
    action_run_done++;
  }
  // -------------------------------------------
  // Final
  // -------------------------------------------
  // indicator whether function is done
  int action_final_done = 0;
  // clear action
  void clear_final() {
    action_final_done = 0;
  }
  // calback
  void test_final_increment(void) {
    action_final_done++;
  }
};

class TimerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    t_act1 = new test_action();
    t_act2 = new test_action();
  }

  void TearDown() override {
  }

  ext_timer<test_action> *timer1;
  ext_timer<test_action> *timer2;
  test_action *t_act1;
  test_action *t_act2;
};

TEST_F(TimerTest, timer) {
  EXPECT_EQ(t_act1->action_run_done, 0);
  EXPECT_EQ(t_act1->action_final_done, 0);
  EXPECT_EQ(t_act2->action_run_done, 0);
  EXPECT_EQ(t_act2->action_final_done, 0);

  ext_timer<test_action> timer1(&test_action::test_final_increment, t_act1);
  ext_timer<test_action> timer2(&test_action::test_run_increment, t_act2);

  // --------------------------------
  // 1. check tick once timer
  // --------------------------------
  // 1.1 run and stop before tick
  t_act1->clear_all();
  timer1.start(50, true);
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  EXPECT_EQ(t_act1->action_run_done, 0);
  EXPECT_EQ(t_act1->action_final_done, 0);
  EXPECT_EQ(timer1.get_brun(), true);
  EXPECT_EQ(timer1.get_brunonce(), true);
  EXPECT_EQ(timer1.get_timeset(), 50);
  timer1.stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(t_act1->action_run_done, 0);
  EXPECT_EQ(t_act1->action_final_done, 0);
  EXPECT_EQ(timer1.get_brun(), false);
  EXPECT_EQ(timer1.get_brunonce(), true);
  EXPECT_EQ(timer1.get_timeset(), 50);

  // 1.2 run and stop after tick
  t_act1->clear_all();
  timer1.start(50, true);
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  EXPECT_EQ(t_act1->action_run_done, 0);
  EXPECT_EQ(t_act1->action_final_done, 0);
  EXPECT_EQ(timer1.get_brun(), true);
  EXPECT_EQ(timer1.get_brunonce(), true);
  EXPECT_EQ(timer1.get_timeset(), 50);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(t_act1->action_run_done, 0);
  EXPECT_EQ(t_act1->action_final_done, 1);
  EXPECT_EQ(timer1.get_brun(), false);
  EXPECT_EQ(timer1.get_brunonce(), true);
  EXPECT_EQ(timer1.get_timeset(), 50);
  timer1.stop();

  // 1.3 run for 10s and check calback
  t_act1->clear_all();
  timer1.start(10000, true);
  std::this_thread::sleep_for(std::chrono::milliseconds(10005));
  EXPECT_EQ(t_act1->action_final_done, 1);
  EXPECT_EQ(timer1.get_brun(), false);
  EXPECT_EQ(timer1.get_brunonce(), true);
  timer1.stop();

  // 1.4 restart timer
  t_act1->clear_all();
  timer1.start(1000, true);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  EXPECT_EQ(t_act1->action_final_done, 0);
  EXPECT_EQ(timer1.get_brun(), true);
  EXPECT_EQ(timer1.get_brunonce(), true);
  timer1.start(100, true);
  std::this_thread::sleep_for(std::chrono::milliseconds(105));
  EXPECT_EQ(t_act1->action_final_done, 1);
  EXPECT_EQ(timer1.get_brun(), false);
  EXPECT_EQ(timer1.get_brunonce(), true);
  timer1.stop();

  // --------------------------------
  // 2. check tick continuous timer
  // --------------------------------
  // 1.1 run and stop before tick
  t_act2->clear_all();
  timer2.start(50, false);
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  EXPECT_EQ(t_act2->action_run_done, 0);
  EXPECT_EQ(t_act2->action_final_done, 0);
  EXPECT_EQ(timer2.get_brun(), true);
  EXPECT_EQ(timer2.get_brunonce(), false);
  EXPECT_EQ(timer2.get_timeset(), 50);
  timer2.stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(t_act2->action_run_done, 0);
  EXPECT_EQ(t_act2->action_final_done, 0);
  EXPECT_EQ(timer2.get_brun(), false);
  EXPECT_EQ(timer2.get_brunonce(), false);
  EXPECT_EQ(timer2.get_timeset(), 50);

  // 1.2 run and stop after tick
  t_act2->clear_all();
  timer2.start(100, false);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(t_act2->action_run_done, 0);
  EXPECT_EQ(t_act2->action_final_done, 0);
  EXPECT_EQ(timer2.get_brun(), true);
  EXPECT_EQ(timer2.get_brunonce(), false);
  EXPECT_EQ(timer2.get_timeset(), 100);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(t_act2->action_run_done, 1);
  EXPECT_EQ(t_act2->action_final_done, 0);
  EXPECT_EQ(timer2.get_brun(), true);
  EXPECT_EQ(timer2.get_brunonce(), false);
  EXPECT_EQ(timer2.get_timeset(), 100);
  timer2.stop();

  // 1.3 run for 10s and check calback
  t_act2->clear_all();
  timer2.start(10000, false);
  std::this_thread::sleep_for(std::chrono::milliseconds(10005));
  EXPECT_EQ(t_act2->action_run_done, 1);
  EXPECT_EQ(timer2.get_brun(), true);
  EXPECT_EQ(timer2.get_brunonce(), false);
  timer2.stop();

  // 1.4 restart timer
  t_act2->clear_all();
  timer2.start(1000, false);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  EXPECT_EQ(t_act2->action_run_done, 0);
  EXPECT_EQ(timer2.get_brun(), true);
  EXPECT_EQ(timer2.get_brunonce(), false);
  timer2.start(100, false);
  std::this_thread::sleep_for(std::chrono::milliseconds(205));
  EXPECT_EQ(t_act2->action_run_done, 2);
  EXPECT_EQ(timer2.get_brun(), true);
  EXPECT_EQ(timer2.get_brunonce(), false);
  timer2.stop();

  // 1.5 run periodicaly for 1s and check calback
  t_act2->clear_all();
  timer2.start(10000, false);
  EXPECT_EQ(timer2.get_brunonce(), false);
  for (int i = 0; i < 2; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10005));
    EXPECT_EQ(t_act2->action_run_done, i + 1);
    EXPECT_EQ(timer2.get_brun(), true);
  }
  timer2.stop();
}
