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
#include "include/rvsthreadbase.h"

#include <chrono>

//! Default constructor.
rvs::ThreadBase::ThreadBase() : t() {
}

//! Default destructor.
rvs::ThreadBase::~ThreadBase() {
}

/**
 *  \brief Internal thread function
 *
 * Used to construct std::thread object. Calls virtual run()
 * to perform actual payload work.
 *
 */
void rvs::ThreadBase::runinternal() {
  run();
}

/**
 *  \brief Starts the thread.
 *
 * Creates std::thread object passing runinternal() as thread function.
 *
 */
void rvs::ThreadBase::start() {
  t = std::thread(&rvs::ThreadBase::runinternal, this);
}

/**
 *  \brief Performs detach() on the underlaying std::thread object.
 *
 */
void rvs::ThreadBase::detach() {
  t.detach();
}

/**
 *  \brief Performs join() on the underlaying std::thread object.
 *
 */
void rvs::ThreadBase::join() {
  // wait a bit to make sure thread has exited
  try {
    if (t.joinable())
      t.join();
  }
  catch(...) {
  }
}

/**
 * @brief Pauses current thread for the given time period
 *
 * @param ms Sleep time in milliseconds.
 *
 * */
void rvs::ThreadBase::sleep(const unsigned int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
