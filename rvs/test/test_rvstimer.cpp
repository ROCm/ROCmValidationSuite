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

#include <thread>

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
  // check remaining time interval
  EXPECT_LE(timer1.get_timeleft(), 25+10);
  EXPECT_GE(timer1.get_timeleft(), 25-10);
  EXPECT_EQ(timer1.get_timeset(), 50);
  timer1.stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(t_act1->action_run_done, 0);
  EXPECT_EQ(t_act1->action_final_done, 0);
  EXPECT_EQ(timer1.get_brun(), false);
  EXPECT_EQ(timer1.get_brunonce(), true);
  EXPECT_LE(timer1.get_timeleft(), 25+10);
  EXPECT_GE(timer1.get_timeleft(), 25-10);
  EXPECT_EQ(timer1.get_timeset(), 50);

  // 1.2 run and stop after tick
  t_act1->clear_all();
  timer1.start(50, true);
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  EXPECT_EQ(t_act1->action_run_done, 0);
  EXPECT_EQ(t_act1->action_final_done, 0);
  EXPECT_EQ(timer1.get_brun(), true);
  EXPECT_EQ(timer1.get_brunonce(), true);
  // check remaining time interval
  EXPECT_LE(timer1.get_timeleft(), 25+10);
  EXPECT_GE(timer1.get_timeleft(), 25-10);
  EXPECT_EQ(timer1.get_timeset(), 50);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(t_act1->action_run_done, 0);
  EXPECT_EQ(t_act1->action_final_done, 1);
  EXPECT_EQ(timer1.get_brun(), false);
  EXPECT_EQ(timer1.get_brunonce(), true);
  // check remaining time interval
  EXPECT_EQ(timer1.get_timeleft(), 0);
  EXPECT_EQ(timer1.get_timeset(), 50);
  timer1.stop();

  // 1.3 run for 1s and check calback
  t_act1->clear_all();
  timer1.start(1000, true);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  EXPECT_LE(timer1.get_timeleft(), 100);
  EXPECT_GE(timer1.get_timeleft(), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(t_act1->action_final_done, 1);
  EXPECT_EQ(timer1.get_brun(), false);
  EXPECT_EQ(timer1.get_brunonce(), true);
  // check remaining time interval
  EXPECT_EQ(timer1.get_timeleft(), 0);
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
  // check remaining time interval
  EXPECT_LE(timer2.get_timeleft(), 30);
  EXPECT_GE(timer2.get_timeleft(), 20);
  EXPECT_EQ(timer2.get_timeset(), 50);
  timer2.stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(t_act2->action_run_done, 0);
  EXPECT_EQ(t_act2->action_final_done, 0);
  EXPECT_EQ(timer2.get_brun(), false);
  EXPECT_EQ(timer2.get_brunonce(), false);
  EXPECT_EQ(timer2.get_timeleft(), 50); // stop() will reinitialize timeleft
  EXPECT_EQ(timer2.get_timeset(), 50);

  // 1.2 run and stop after tick
  t_act2->clear_all();
  timer2.start(100, false);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(t_act2->action_run_done, 0);
  EXPECT_EQ(t_act2->action_final_done, 0);
  EXPECT_EQ(timer2.get_brun(), true);
  EXPECT_EQ(timer2.get_brunonce(), false);
  // check remaining time interval
  EXPECT_LE(timer2.get_timeleft(), 50+15);
  EXPECT_GE(timer2.get_timeleft(), 50-15);
  EXPECT_EQ(timer2.get_timeset(), 100);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(t_act2->action_run_done, 1);
  EXPECT_EQ(t_act2->action_final_done, 0);
  EXPECT_EQ(timer2.get_brun(), true);
  EXPECT_EQ(timer2.get_brunonce(), false);
  // check remaining time interval
  // check remaining time interval
  EXPECT_LE(timer2.get_timeleft(), 50+15);
  EXPECT_GE(timer2.get_timeleft(), 50-15);
  EXPECT_EQ(timer2.get_timeset(), 100);
  timer2.stop();

  // 1.3 run for 1s and check calback
  t_act2->clear_all();
  timer2.start(1000, false);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  EXPECT_LE(timer2.get_timeleft(), 100);
  EXPECT_GE(timer2.get_timeleft(), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(t_act2->action_run_done, 1);
  EXPECT_EQ(timer2.get_brun(), true);
  EXPECT_EQ(timer2.get_brunonce(), false);
  // check remaining time interval
  EXPECT_LE(timer2.get_timeleft(), 1000);
  EXPECT_GE(timer2.get_timeleft(), 850);
  timer2.stop();

}
