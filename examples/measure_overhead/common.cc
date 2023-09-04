/* Helper for printing statistics of a set of measurements

   Copyright (c) 2019 pocl developers

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

#include "common.hh"
#include <algorithm>
#include <iostream>

void print_measurements(const std::string &title,
                        const std::vector<double> &times, int indent) {
  if (times.empty()) {
    std::cout << "\t\tNo Data" << std::endl;
    return;
  }
  double sum_time = 0;
  double min_time = times[0];
  double max_time = times[0];
  for (double time : times) {
    sum_time += time;
    min_time = std::min(time, min_time);
    max_time = std::max(time, max_time);
  }
  sum_time /= times.size();

  std::vector<double> sorted_times(times);
  std::sort(sorted_times.begin(), sorted_times.end());

  std::string ind(indent, '\t');
  std::cout << ind << title << std::endl
            << ind << "\taverage: " << sum_time << " µs" << std::endl
            << ind << "\tmin: " << min_time << " µs" << std::endl
            << ind << "\tmax: " << max_time << " µs" << std::endl
            << ind << "\t99th percentile: "
            << sorted_times[sorted_times.size() * 99 / 100] << " µs"
            << std::endl;
}
