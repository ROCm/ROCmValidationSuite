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

#include <vector>

#include "gtest/gtest.h"

#include "include/rvsthreadbase.h"
#include "include/rvs_unit_testing_defs.h"

class ext_thread : public rvs::ThreadBase {
 public:
  virtual ~ext_thread() {
  }
  // indicator for thread finished
  int finished = 0;
  // duration of the thread
  int wait_ms = 1000;
  // run override
  void run() {
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    finished = 1;
  }
  void clear() {
    finished = 0;
  }
};

class ThreadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    t1 = new ext_thread();
    t2 = new ext_thread();
  }

  void TearDown() override {
    delete t1;
    delete t2;
  }

  // threads
  ext_thread* t1;
  ext_thread* t2;
};

TEST_F(ThreadTest, threadbase) {
  // 1. start and join
  t1->clear();
  t2->clear();
  EXPECT_EQ(t1->finished, 0);
  EXPECT_EQ(t2->finished, 0);

  t1->wait_ms = 10;
  t1->start();

  t2->wait_ms = 5;
  t2->start();

  t1->join();
  t2->join();

  EXPECT_EQ(t1->finished, 1);
  EXPECT_EQ(t2->finished, 1);

  // 2. start and join/detach
  t1->clear();
  t2->clear();
  EXPECT_EQ(t1->finished, 0);
  EXPECT_EQ(t2->finished, 0);

  t1->wait_ms = 5;
  t1->start();
  t1->detach();

  t2->wait_ms = 10;
  t2->start();

  t2->join();

  EXPECT_EQ(t1->finished, 1);
  EXPECT_EQ(t2->finished, 1);

  // 3. start and detach with sleep
  t1->clear();
  t2->clear();
  EXPECT_EQ(t1->finished, 0);
  EXPECT_EQ(t2->finished, 0);

  t1->wait_ms = 10;
  t1->start();
  t1->detach();

  t2->wait_ms = 5;
  t2->start();
  t2->sleep(15);

  t2->join();

  EXPECT_EQ(t1->finished, 1);
  EXPECT_EQ(t2->finished, 1);

  // 4. detach
  t1->clear();
  t2->clear();
  EXPECT_EQ(t1->finished, 0);
  EXPECT_EQ(t2->finished, 0);

  t1->wait_ms = 10;
  t1->start();
  t1->detach();

  t2->wait_ms = 20;
  t2->start();
  t2->detach();

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(t1->finished, 0);
  EXPECT_EQ(t2->finished, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(t1->finished, 1);
  EXPECT_EQ(t2->finished, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(t1->finished, 1);
  EXPECT_EQ(t2->finished, 1);

  // 5. sleep
  t1->clear();
  t2->clear();
  EXPECT_EQ(t1->finished, 0);
  EXPECT_EQ(t2->finished, 0);

  t1->wait_ms = 10;
  t1->start();
  t1->sleep(10);
  t1->detach();

  t2->wait_ms = 20;
  t2->start();
  t2->sleep(10);
  t2->detach();

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(t1->finished, 1);
  EXPECT_EQ(t2->finished, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(t1->finished, 1);
  EXPECT_EQ(t2->finished, 1);
}

