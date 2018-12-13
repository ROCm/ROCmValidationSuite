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
#ifndef INCLUDE_RVSTIMER_H_
#define INCLUDE_RVSTIMER_H_

#include <chrono>

#include "include/rvsthreadbase.h"

namespace rvs {

/**
 * @class timer
 * @ingroup RVS
 *
 * @brief Timer utility template class
 *
 * Provides for easy implementation of timer based scenarios.
 * It accepts parameter T which is a class which member function will
 * be called upon expiration of timer interval.
 *
 * Timer resolution is 1ms
 *
 */

template<class T>
class timer : public ThreadBase {
 public:
  //! helper typedef to simplify member declaration of callback function
  typedef void (T::*timerfunc_t)();

/**
 * @brief Constructor
 * @param cbFunc Member function in calss T. it has to be of type "void f()"";
 * @param cbArg pointer to instance of class T method of which will be called
 * back
 *
 */
  timer(timerfunc_t cbFunc, T* cbArg) {
    cbfunc = cbFunc;
    cbarg = cbArg;
  }

  //! Default destructor
  virtual ~timer() {
    stop();
  }

  /**
  * @brief Start timer
  *
  * @param Interval Timer interval in ms
  * @param RunOnce 'true' if timer is to fire only once
  *
  * */
  void start(int Interval, bool RunOnce = false) {
    brunonce = RunOnce;
    timeset = Interval;
    end_time = std::chrono::system_clock::now() +
               std::chrono::milliseconds(timeset);
    // recalculate the end time and only if thread is not started start it
    if (brun == false) {
      brun = true;
      rvs::ThreadBase::start();
    }
  }


/**
 * @brief Stop timer
 *
 * Sets brun member to FALSE thus signaling end of processing.
 * Then it waits for std::thread to exit before returning.
 *
 * */
  void stop() {
    // signal thread to exit
    brun = false;

    // wait a bit to make sure thread has exited
    std::this_thread::yield();

    try {
      if (t.joinable())
        t.join();
    } catch(...) {
    }
  }


/**
 * @brief Timer internal thread function (called from ThreadBase)
 *
 * */
  virtual void run() {
    do {
      std::chrono::time_point<std::chrono::system_clock> curr_time;
      // wait for time to ellsapse (or for timer to be stopped)
      while (brun) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        curr_time = std::chrono::system_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>
                                         (curr_time - end_time).count() >= 0) {
          break;
        }
      }

      // if timer is not stopped, call the callback function
      if (brun) {
        (cbarg->*cbfunc)();
      }

      if (brunonce) {
        brun = false;
      } else {
        end_time = std::chrono::system_clock::now() +
                   std::chrono::milliseconds(timeset);
      }
    } while (brun);
  }

 protected:
  //! true for the duration of timer activity
  bool        brun;
  //! true if timer is to fire only once
  bool        brunonce;
  //! timer interval (ms)
  int         timeset;
  //! call back function
  timerfunc_t cbfunc;
  //! ptr to instance of a class to be called-back through cbfunc.
  T*          cbarg;
  //! time when timer will expire
  std::chrono::time_point<std::chrono::system_clock> end_time;
};

}  // namespace rvs

#endif  // INCLUDE_RVSTIMER_H_
