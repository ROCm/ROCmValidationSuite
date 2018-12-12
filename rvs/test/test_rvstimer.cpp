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
  EXPECT_LE(timer1.get_timeleft(), 30);
  EXPECT_GE(timer1.get_timeleft(), 20);
  EXPECT_EQ(timer1.get_timeset(), 50);
  timer1.stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(t_act1->action_run_done, 0);
  EXPECT_EQ(t_act1->action_final_done, 0);
  EXPECT_EQ(timer1.get_brun(), false);
  EXPECT_EQ(timer1.get_brunonce(), true);
  EXPECT_LE(timer1.get_timeleft(), 30);
  EXPECT_GE(timer1.get_timeleft(), 20);
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
  EXPECT_LE(timer1.get_timeleft(), 30);
  EXPECT_GE(timer1.get_timeleft(), 20);
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

  EXPECT_EQ(1, 1);
}

/*
  //! true for the duration of timer activity
  bool        get_brun;
  //! true if timer is to fire only once
  bool        get_brunonce;
  //! time left to next fire (ms)
  int         get_timeleft;
  //! timer interval (ms)
  int         get_timeset;

./pqt.so/src/action_run.cpp:109:  // define timers
./pqt.so/src/action_run.cpp:110:  rvs::timer<pqt_action> timer_running(&pqt_action::do_running_average, this);
./pqt.so/src/action_run.cpp:111:  rvs::timer<pqt_action> timer_final(&pqt_action::do_final_average, this);
./pqt.so/src/action_run.cpp:121:    // start timers
./pqt.so/src/action_run.cpp:124:      timer_final.start(property_duration, true);  // ticks only once
./pqt.so/src/action_run.cpp:129:      timer_running.start(property_log_interval);        // ticks continuously
./pqt.so/src/action_run.cpp:143:    timer_running.stop();
./pqt.so/src/action_run.cpp:144:    timer_final.stop();
*/