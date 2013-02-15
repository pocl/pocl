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
        self.case_desc = case_description
        self.results = {}

    def add_result(self, platform, res):
        self.results[platform] = res
        
class BenchmarkResults(object):
    def __init__(self, title_, *platforms_):
        self.cpu = None
        self.platforms = platforms_
        self.cases = []
        self.title = title_

    def bar_chart(self, output_fname):
        """Following the example from
        http://matplotlib.org/examples/pylab_examples/barchart_demo.html
        to produce a basic bar chart. """        

        ind = np.arange(len(self.cases)) # the ex locations for the groups
        width = 0.35 # the width of the bars
        plt.subplot(111)

        colors = 'kwbrg'
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
        benchmark_names = [x.case_desc + '  ' for x in self.cases]
        locs, labels = plt.xticks(ind+(width*i)/2, benchmark_names, ha='left')
        setp(labels, 'rotation', 'vertical')
        plt.legend([x[0] for x in rects], self.platforms)
        plt.savefig(output_fname, bbox_inches="tight")

def parse_results(fn, results):
    f = open(fn, 'r')
    found_cpu = False
    found_case = False
    for line in f.readlines():
        if not found_cpu:     
            splat = line.split(':')

            if len(splat) and splat[0].strip() == 'CPU':
                results.cpu = splat[1].strip()
                found_cpu = True
            continue
        
        if not found_case:
            if line.startswith('case'):
                found_case = True
            continue

        # Now parsing the cases
        splat = line.split()
        case = BenchmarkCase(splat[0])
        i = 0
        for number in splat[1:-1]:
            try:
                float(number)
                case.add_result(results.platforms[i], float(number))
            except:
                case.add_result(results.platforms[i], 0.0)                
            i += 1

        results.cases.append(case)
    return results

if __name__ == "__main__":
    input_fname = sys.argv[2]
    results = BenchmarkResults(sys.argv[1], sys.argv[3], sys.argv[4])
    results = parse_results(input_fname, results)

    output_fname = ".".join(input_fname.split(".")[0:-1]) + ".eps"
    results.bar_chart(output_fname)


    
