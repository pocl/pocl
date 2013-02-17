#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright (c) 2013 Pekka Jääskeläinen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# Creates a barchart out from a results file produced by benchmark.py
# 
# requires matplotlib http://matplotlib.org
import sys
import os
import os.path

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from pylab import *
except:
    print "Matplotlib and numpy are required!"
    sys.exit(1)


class BenchmarkResult(object):
    def __init__(self, runtime):
        self.kernel_run_time = runtime

class BenchmarkCase(object):
    def __init__(self, case_description):
        self.desc = case_description
        self.results = {}

    def add_result(self, platform, res):
        self.results[platform] = res
        
class BenchmarkResults(object):
    def __init__(self, title_, platforms_):
        self.cpu = None
        self.platforms = ['pocl'] + sorted(list(set(platforms_) - set(['pocl'])))
        self.cases = []
        self.title = title_

    def bar_chart(self, output_fname):
        """Following the example from
        http://matplotlib.org/examples/pylab_examples/barchart_demo.html
        to produce a basic bar chart. """        

        width = 0.35 # the width of the bars
        all_bars_in_a_group = width * len(self.platforms) + width
        ind = np.arange(len(self.cases)) * all_bars_in_a_group
        plt.subplot(111)

        colors = ['1.0','0.5', '0.0']
        i = 0
        rects = []
        # Add as many bars as there are platforms
        for platform in self.platforms:
            results = [x.results[platform] for x in self.cases]
            rects1 = plt.bar(ind+width*i, results, width, color=colors[i])
            i += 1
            rects.append(rects1)

        plt.ylabel('Runtime (s)')
        plt.title(self.title)
        benchmark_names = [x.desc + '  ' for x in self.cases]
        locs, labels = plt.xticks(ind + width, benchmark_names, ha='left')
        setp(labels, 'rotation', 'vertical')
        plt.legend([x[0] for x in rects], self.platforms)
        plt.savefig(output_fname, bbox_inches="tight")

    def _get_benchmark_case(self, case_desc):
        for case in self.cases:
            if case.desc == case_desc: return case
        case = BenchmarkCase(case_desc)
        self.cases.append(case)
        return case

    def parse(self, fn, all_platforms, platform_start):
        f = open(fn, 'r')
        found_cpu = False
        found_case = False
        # How many platforms were in this result set.
        platform_count = 0
        platforms = all_platforms[platform_start:]
        for line in f.readlines():
            if not found_cpu:     
                splat = line.split(':')

                if len(splat) and splat[0].strip() == 'CPU':
                    cpu = splat[1].strip()
                    found_cpu = True
                continue
        
            if not found_case:
                if line.startswith('case'):
                    found_case = True
                continue

            # Now parsing the cases
            splat = line.split()
            case = self._get_benchmark_case(splat[0])
            i = 0
            for number in splat[1:-1]:
                try:
                    float(number)
                    case.add_result(platforms[i], float(number))
                except:
                    case.add_result(platforms[i], 0.0)                
                i += 1
            platform_count = max(i, platform_count)
            
        return platform_count

if __name__ == "__main__":
    result_files = []
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            result_files.append(arg)

    input_fname = result_files[0]
    i = len(result_files) + 1
    title = sys.argv[i]
    platforms = sys.argv[i+1:]
    results = BenchmarkResults(title, platforms)

    consumed_platforms = 0
    for ifn in result_files:
        consumed_platforms += results.parse(ifn, platforms, consumed_platforms)

    output_fname = ".".join(input_fname.split(".")[0:-1]) + ".eps"
    results.bar_chart(output_fname)


    
