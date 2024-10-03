/* OpenCL runtime library: time measurement utility functions

   Copyright (c) 2024 Michal Babej / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "pocl_timing.h"

#include <chrono>
#include <ctime>

uint64_t pocl_gettimemono_ns() {
  auto TP = std::chrono::high_resolution_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(TP).count();
}

uint64_t pocl_gettimer_resolution() {
  auto DP = std::chrono::high_resolution_clock::duration(1);
  return std::chrono::duration_cast<std::chrono::nanoseconds>(DP).count();
}

int pocl_gettimereal(int *year, int *mon, int *day,
                     int *hour, int *min,
                     int *sec, int *nanosec) {
  // TODO C++20 makes this simpler
  struct tm t;
  struct timespec timespec;
  time_t sec_input;

  std::time_t TS = std::time(nullptr);
  struct std::tm *TM = std::gmtime(&TS);
  *year = TM->tm_year + 1900;
  *mon = TM->tm_mon + 1;
  *day = TM->tm_mday;
  *hour = TM->tm_hour;
  *min = TM->tm_min;
  *sec = TM->tm_sec;
  return 0;
}
