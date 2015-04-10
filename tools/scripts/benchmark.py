#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright (c) 2012-2014 Pekka Jääskeläinen
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
# Measures the performance of several test cases and compares it to
# the given (vendor) OpenCL implementation.
#
# Requires the ocl-icd as the icd loader. It uses its OCL_ICD_VENDORS env
# to limit the choice of the OpenCL implementation to only the benchmarked 
# one.
#
# Run this from the pocl top source directory. For example:
#
# tools/scripts/benchmark.py /etc/OpenCL/vendors_intel_only
#
# This runs the test suite with pocl and with the ICD found in the given
# directory. It assumes the directory contains only one ICD file. In this
# case it could be the Intel's OpenCL. If you run the benchmark
# without the parameter, it measures only pocl execution times.

import sys
import os
import tempfile
import signal
import time
import datetime
import platform
import optparse
import re

from optparse import OptionParser

from subprocess import Popen, PIPE

# With POCL we can reuse the old compilation results by
# leaving the kernel compiler temp directories and rerunning
# the same OpenCL app. This simulates a compiler cache in pocl.
# At least NVIDIA OpenCL has a ccache in their implementation.
POCL_EXCLUDE_COMPILATION_TIME = True

# How many times each case is repeated. The best result is picked from these.
REPEAT_COUNT = 5
POCL_SRC_ROOT_PATH = os.getcwd()

AMD_TEST_ROOT_DIR = "examples/AMD/AMD-APP-SDK-v2.8-RC-lnx64/samples/opencl/cl/app"
HALIDE_TEST_ROOT_DIR = "examples/Halide/Halide/"
CLOVERLEAF_TEST_ROOT_DIR = "examples/CloverLeaf/CloverLeaf_OpenCL/"

AMD_BIN_DIR = "/build/debug/x86_64/"

AMD2_9_CL_TEST_ROOT_DIR = "examples/AMDSDK2.9/AMD-APP-SDK-v2.9-RC-lnx64/samples/opencl/cl"

AMD2_9_CPP_CL_TEST_ROOT_DIR = "examples/AMDSDK2.9/AMD-APP-SDK-v2.9-RC-lnx64/samples/opencl/cpp_cl"

AMD2_9_BIN_DIR = "/bin/x86_64/Release/"

def run_cmd(command, inputStream = ""):
    """
    Runs the given process until it exits or a given time out is reached.

    Returns a tuple: (bool:timeout, str:stdout, str:stderr, int:exitcode)
    """
    timeoutSecs = 3600
    timePassed = 0.0
    increment = 0.01

    stderrFD, errFile = tempfile.mkstemp()
    stdoutFD, outFile = tempfile.mkstemp()

    process =  Popen(command, shell=True, stdin=PIPE, stdout=stdoutFD, 
                     stderr=stderrFD, close_fds=False)

    if process == None:
        print "Could not create process"
        sys.exit(1)

    try:
        if inputStream != "":
            for line in inputStream:
                process.stdin.write(line)
                process.stdin.flush()

        while True:
            status = process.poll()
            if status != None:
                # Process terminated succesfully.
                stdoutSize = os.lseek(stdoutFD, 0, 2)
                stderrSize = os.lseek(stderrFD, 0, 2)

                os.lseek(stdoutFD, 0, 0)
                os.lseek(stderrFD, 0, 0)

                stdoutContents = os.read(stdoutFD, stdoutSize)
                stderrContents = os.read(stderrFD, stderrSize)

                os.close(stdoutFD)
                os.remove(outFile)
                os.close(stderrFD)
                os.remove(errFile)
                return (False, stdoutContents, stderrContents, process.returncode)

            if timePassed < timeoutSecs:
                time.sleep(increment)
                timePassed = timePassed + increment
            else:
                # time out, kill the process.
                stdoutSize = os.lseek(stdoutFD, 0, 2)
                stderrSize = os.lseek(stderrFD, 0, 2)

                os.lseek(stdoutFD, 0, 0)
                os.lseek(stderrFD, 0, 0)

                stdoutContents = os.read(stdoutFD, stdoutSize)
                stderrContents = os.read(stderrFD, stderrSize)

                os.close(stdoutFD)
                os.remove(outFile)
                os.close(stderrFD)
                os.remove(errFile)
                os.kill(process.pid, signal.SIGTSTP)
                return (True, stdoutContents, stderrContents, process.returncode)
    except Exception, e:
        # if something threw exception (e.g. ctrl-c)
        print e
        os.kill(process.pid, signal.SIGTSTP)
        try:
            # time out, kill the process.
            # time out, kill the process.
            stdoutSize = os.lseek(stdoutFD, 0, 2)
            stderrSize = os.lseek(stderrFD, 0, 2)

            os.lseek(stdoutFD, 0, 0)
            os.lseek(stderrFD, 0, 0)

            stdoutContents = os.read(stdoutFD, stdoutSize)
            stderrContents = os.read(stderrFD, stderrSize)

            os.close(stdoutFD)
            os.remove(outFile)
            os.close(stderrFD)
            os.remove(errFile)
            os.kill(process.pid, signal.SIGTSTP)                
        except:
            pass

        return (False, stdoutContents, stderrContents, process.returncode)


class BenchmarkCase(object):
    def __init__(self, name, wg_method=None):
        self.name = name
        self.wg_method = None

    # Returns the execution time in seconds.
    def execution_time(self):
        pass
    
    def repeat(self, times):
        """Executes the benchmark case given number of times and returns the
        best result."""
        best = None
        if POCL_EXCLUDE_COMPILATION_TIME:
            temp_dir = tempfile.mkdtemp(suffix=self.name)
            os.environ['POCL_LEAVE_TEMP_DIRS'] = '1'
            os.environ['POCL_TEMP_DIR'] = temp_dir

        if self.wg_method is not None:
            os.environ['POCL_WORK_GROUP_METHOD'] = self.wg_method

        for t in range(times):
            result = self.run()
            if best is None or result.kernel_run_time < best.kernel_run_time:
                best = result

        return best            

    def get_kernel_runtime(self):
        pass

class BenchmarkResult(object):
    def __init__(self, kernel_run_time):
        self.kernel_run_time = kernel_run_time


class HalideBenchmarkCase(BenchmarkCase):
    def __init__(self, name, app_dir, command):
        super(HalideBenchmarkCase, self).__init__(name)
        self.stdout = ""
        self.test_app_dir = app_dir
        self.command = command    

    def run(self):
        directory = os.path.join(POCL_SRC_ROOT_PATH, HALIDE_TEST_ROOT_DIR, 
                                 "apps", self.test_app_dir)
        os.chdir(directory)

        halide_lib_dir = os.path.join(POCL_SRC_ROOT_PATH, HALIDE_TEST_ROOT_DIR, "bin")

        os.environ['HL_TARGET'] = 'opencl'
        if halide_lib_dir not in os.environ['LD_LIBRARY_PATH']:
            os.environ['LD_LIBRARY_PATH'] = halide_lib_dir + ":" + \
                                            os.environ['LD_LIBRARY_PATH']

        start_time = time.clock()
        timeout, self.stdout, self.stderr, rc = run_cmd(self.command)
        if timeout or rc != 0:
            sys.stderr.write("\nFAIL (cmd: %s in dir: %s rc: %d).\n" % \
                             ( self.command, directory, rc) )
            sys.stderr.write("stderr:")
            sys.stderr.write(self.stderr)
            sys.stderr.write("\nstdout:")
            sys.stderr.write(self.stdout)
            sys.stderr.write("\n")
            sys.exit(1)
        time_spent = time.clock() - start_time
        
        result = BenchmarkResult(time_spent)
        return result

class CloverLeafBenchmarkCase(BenchmarkCase):
    def __init__(self, name, command="./clover_leaf"):
        super(CloverLeafBenchmarkCase, self).__init__(name)
        self.stdout = ""
        self.command = command    

    def run(self):
        directory = os.path.join(POCL_SRC_ROOT_PATH, CLOVERLEAF_TEST_ROOT_DIR)
        os.chdir(directory)

        timeout, self.stdout, self.stderr, rc = run_cmd(self.command)
        if timeout or rc != 0 or 'Wall clock' not in self.stderr:
            sys.stderr.write("\nFAIL (cmd: %s in dir: %s rc: %d).\n" % \
                             ( self.command, directory, rc) )
            sys.stderr.write("stderr:")
            sys.stderr.write(self.stderr)
            sys.stderr.write("\nstdout:")
            sys.stderr.write(self.stdout)
            sys.stderr.write("\n")
            sys.exit(1)

        final_wall_clock = self.stderr.splitlines()[-2]
        time_spent = float(re.search(r'Wall\sclock\s+(\d\.\d+)', final_wall_clock).group(1))

        result = BenchmarkResult(time_spent)
        return result


class KernelGrindBenchmarkCase(BenchmarkCase):
    def __init__(self, name, command, root_dir=AMD_TEST_ROOT_DIR, 
                 bin_dir=AMD_BIN_DIR, wg_method="auto"):
        super(KernelGrindBenchmarkCase, self).__init__(name, wg_method)
        self.stdout = ""
        self.test_root_dir = root_dir
        self.test_bin_dir = bin_dir
        self.command = command    

    def get_kernel_runtime(self, stdout):
        lines = stdout.split("\n")
        s = 0
        for line in lines:
            if "Execution time" in line:
                s = (line.split(':')[1])
                break

#        print lines[i]
#        print lines[i].split()

        return float(s)

    def run(self):
        directory = POCL_SRC_ROOT_PATH + "/" + self.test_root_dir
        os.chdir(directory)
        # Iterate 10 times to amortize the kernel compilation time.
        # In a real application, the kernel compilation overheads can be excluded 
        # (to some extent, at least) by using the binary API of OpenCL, so it 
        # should be realistic to exclude it.
        cmd = "." + self.test_bin_dir + self.command 
        timeout, self.stdout, self.stderr, rc = run_cmd(cmd)
        if timeout or rc != 0:
            sys.stderr.write("\nFAIL (cmd: %s in dir: %s rc: %d).\n" % \
                             ( cmd, directory, rc) )
            sys.stderr.write("stderr:")
            sys.stderr.write(self.stderr)
            sys.stderr.write("\nstdout:")
            sys.stderr.write(self.stdout)
            sys.stderr.write("\n")
            sys.exit(1)
        
        result = BenchmarkResult(self.get_kernel_runtime(self.stdout))
        return result

class AMDBenchmarkCase(BenchmarkCase):
    def __init__(self, name, command, root_dir=AMD_TEST_ROOT_DIR, 
                 bin_dir=AMD_BIN_DIR, wg_method="auto"):
        super(AMDBenchmarkCase, self).__init__(name, wg_method)
        self.stdout = ""
        self.test_root_dir = root_dir
        self.test_bin_dir = bin_dir
        self.command = command    

    def get_kernel_runtime(self, stdout):
        lines = stdout.split("\n")
        i = 0
        time_column = 1
        for line in lines:
            i += 1
            if "Time" in line:

                columns = line.split()
                time_column = 0
                for col in columns:
                    if "Time" in col:
                        break
                    time_column += 1
                break

#        print lines[i]
#        print lines[i].split()

        return float(lines[i].split()[time_column])

    def run(self):
        directory = POCL_SRC_ROOT_PATH + "/" + self.test_root_dir + "/" +self.name
        os.chdir(directory)
        # Iterate 10 times to amortize the kernel compilation time.
        # In a real application, the kernel compilation overheads can be excluded 
        # (to some extent, at least) by using the binary API of OpenCL, so it 
        # should be realistic to exclude it.
        cmd = "." + self.test_bin_dir + self.command + " -i 10"
        timeout, self.stdout, self.stderr, rc = run_cmd(cmd)
        if timeout or rc != 0:
            sys.stderr.write("\nFAIL (cmd: %s in dir: %s rc: %d).\n" % \
                             ( cmd, directory, rc) )
            sys.stderr.write("stderr:")
            sys.stderr.write(self.stderr)
            sys.stderr.write("\nstdout:")
            sys.stderr.write(self.stdout)
            sys.stderr.write("\n")
            sys.exit(1)
        
        result = BenchmarkResult(self.get_kernel_runtime(self.stdout))
        return result


class AMDBenchmarkCase(BenchmarkCase):
    def __init__(self, name, command, root_dir=AMD_TEST_ROOT_DIR, 
                 bin_dir=AMD_BIN_DIR, wg_method="auto"):
        super(AMDBenchmarkCase, self).__init__(name, wg_method)
        self.stdout = ""
        self.test_root_dir = root_dir
        self.test_bin_dir = bin_dir
        self.command = command    

    def get_kernel_runtime(self, stdout):
        lines = stdout.split("\n")
        i = 0
        time_column = 1
        for line in lines:
            i += 1
            if "Time" in line:

                columns = line.split()
                time_column = 0
                for col in columns:
                    if "Time" in col:
                        break
                    time_column += 1
                break

#        print lines[i]
#        print lines[i].split()

        return float(lines[i].split()[time_column])

    def run(self):
        directory = POCL_SRC_ROOT_PATH + "/" + self.test_root_dir + "/" +self.name
        os.chdir(directory)
        # Iterate 10 times to amortize the kernel compilation time.
        # In a real application, the kernel compilation overheads can be excluded 
        # (to some extent, at least) by using the binary API of OpenCL, so it 
        # should be realistic to exclude it.
        cmd = "." + self.test_bin_dir + self.command + " -i 10"
        timeout, self.stdout, self.stderr, rc = run_cmd(cmd)
        if timeout or rc != 0:
            sys.stderr.write("\nFAIL (cmd: %s in dir: %s rc: %d).\n" % \
                             ( cmd, directory, rc) )
            sys.stderr.write("stderr:")
            sys.stderr.write(self.stderr)
            sys.stderr.write("\nstdout:")
            sys.stderr.write(self.stdout)
            sys.stderr.write("\n")
            sys.exit(1)
        
        result = BenchmarkResult(self.get_kernel_runtime(self.stdout))
        return result



class AMDBenchmarkCaseTransferOverlap(AMDBenchmarkCase):
    def __init__(self, name, command, root_dir, bin_dir, wg_method="auto"):
        super(AMDBenchmarkCaseTransferOverlap, self).__init__(name, command, root_dir, bin_dir, wg_method)
        
    def get_kernel_runtime(self, stdout):
        lines = stdout.split("\n")
        s = 0
        for line in lines:
            if "Complete test time" in line:
                s = (line.split(':')[1]).split()[0]
                break

        return float(s)

class AMDBenchmarkCaseConcurrentKernel(AMDBenchmarkCase):
    def __init__(self, name, command, root_dir, bin_dir, wg_method="auto"):
        super(AMDBenchmarkCaseConcurrentKernel, self).__init__(name, command, root_dir, bin_dir, wg_method)

    def get_kernel_runtime(self, stdout):
        lines = stdout.split("\n")
        i = 0
        s = 0
        for line in lines:
            if "Size(Bytes)" in line:
                if self.name == "ConcurrentKernel_Con":
                    s = lines[i+2].split("|")[4]
                if self.name == "ConcurrentKernel_Seq":
                    s = lines[i+2].split("|")[3]
                break
            i += 1

        return float(s)

    def run(self):
        directory = POCL_SRC_ROOT_PATH + "/" + self.test_root_dir + "/" + "ConcurrentKernel"
        os.chdir(directory)
        # Iterate 10 times to amortize the kernel compilation time.
        # In a real application, the kernel compilation overheads can be excluded 
        # (to some extent, at least) by using the binary API of OpenCL, so it 
        # should be realistic to exclude it.
        cmd = "." + self.test_bin_dir + self.command + " -i 10"
        timeout, self.stdout, self.stderr, rc = run_cmd(cmd)
        if timeout or rc != 0:
            sys.stderr.write("\nFAIL (cmd: %s in dir: %s rc: %d).\n" % \
                             ( cmd, directory, rc) )
            sys.stderr.write("stderr:")
            sys.stderr.write(self.stderr)
            sys.stderr.write("\nstdout:")
            sys.stderr.write(self.stdout)
            sys.stderr.write("\n")
            sys.exit(1)
        
        result = BenchmarkResult(self.get_kernel_runtime(self.stdout))
        return result

class AMDBenchmarkCaseAsyncDataTransfer(AMDBenchmarkCase):
    def __init__(self, name, command, root_dir, bin_dir, wg_method="auto"):
        super(AMDBenchmarkCaseAsyncDataTransfer, self).__init__(name, command, root_dir, bin_dir, wg_method)

    def get_kernel_runtime(self, stdout):
        lines = stdout.split("\n")
        i = 0
        s = 0
        for line in lines:
            if "Size(Bytes)" in line:
                if self.name == "AsyncDataTransfer_Sync":
                    s = lines[i+2].split("|")[3]
                if self.name == "AsyncDataTransfer_Async":
                    s = lines[i+2].split("|")[4]
                break
            i += 1

        return float(s)

    def run(self):
        directory = POCL_SRC_ROOT_PATH + "/" + self.test_root_dir + "/" + "AsyncDataTransfer"
        os.chdir(directory)
        # Iterate 10 times to amortize the kernel compilation time.
        # In a real application, the kernel compilation overheads can be excluded 
        # (to some extent, at least) by using the binary API of OpenCL, so it 
        # should be realistic to exclude it.
        cmd = "." + self.test_bin_dir + self.command + " -i 10"
        timeout, self.stdout, self.stderr, rc = run_cmd(cmd)
        if timeout or rc != 0:
            sys.stderr.write("\nFAIL (cmd: %s in dir: %s rc: %d).\n" % \
                             ( cmd, directory, rc) )
            sys.stderr.write("stderr:")
            sys.stderr.write(self.stderr)
            sys.stderr.write("\nstdout:")
            sys.stderr.write(self.stdout)
            sys.stderr.write("\n")
            sys.exit(1)
        
        result = BenchmarkResult(self.get_kernel_runtime(self.stdout))
        return result


class AMDBenchmarkCaseKernelLauch(AMDBenchmarkCase):
    def __init__(self, name, command, root_dir, bin_dir, wg_method="auto"):
        super(AMDBenchmarkCaseKernelLauch, self).__init__(name, command, root_dir, bin_dir, wg_method)
        
    def get_kernel_runtime(self, stdout):
        lines = stdout.split("\n")
        s = 0
        for line in lines:
            if "clEnqueueNDRangeKernel()" in line:
                s = (line.split(':')[1]).split()[0]
                break
        #scale up to get reasonable result
        return float(s) * 100000

class EinsteinToolkitCase(BenchmarkCase):
    def __init__(self, name):
        super(EinsteinToolkitCase, self).__init__(name)
        self.stdout = ""
        self.test_root_dir = "examples/EinsteinToolkit"

    def get_kernel_runtime(self, stdout):
        lines = stdout.split("\n")
        for line in lines:
            if "Total elapsed time:" in line:
                return float(line.split(": ")[1].replace(" sec", ""))
        assert False

    def run(self):
        os.chdir(POCL_SRC_ROOT_PATH + "/" + self.test_root_dir)
        timeout, self.stdout, self.stderr, rc = run_cmd("./EinsteinToolkit")
        if timeout or rc != 0:
            sys.stderr.write(self.name + " FAIL (rc %d).\n" % rc)
            sys.exit(1)
        
        result = BenchmarkResult(self.get_kernel_runtime(self.stdout))
        return result


pocl_ocl_dir = POCL_SRC_ROOT_PATH + "/ocl-vendors"

# Benchmarks that exercise the kernel. As big input sets as possible.
amd_benchmarks = \
    [AMDBenchmarkCase("AESEncryptDecrypt", "AESEncryptDecrypt -t -q"),
     AMDBenchmarkCase("BitonicSort", "BitonicSort -q -t -x 1048576"),
     AMDBenchmarkCase("BinarySearch", "BinarySearch -q -t -x 5242880000"),
     AMDBenchmarkCase("BinomialOption", "BinomialOption -q -t -x 10000"),
     AMDBenchmarkCase("BlackScholes", "BlackScholes -q -t -x 16777216"),
     AMDBenchmarkCase("DCT", "DCT -q -t -x 4000 -y 4000"),
     AMDBenchmarkCase("FastWalshTransform", "FastWalshTransform -q -t -x 134217728"),
     AMDBenchmarkCase("FloydWarshall", "FloydWarshall -q -t -x 512"),
     AMDBenchmarkCase("Histogram", "Histogram -t -x 15000 -y 15000 -q"),
     AMDBenchmarkCase("Mandelbrot", "Mandelbrot -t -x 8192 -y 8192 -q"),
     AMDBenchmarkCase("MatrixTranspose", "MatrixTranspose -t -x 12288 -y 12288 -q"),
     #This gives garbage execution times for some reason
     AMDBenchmarkCase("MatrixMultiplication", "MatrixMultiplication -q -t -x 1024 -y 1024 -z 2048"),
     AMDBenchmarkCase("NBody", "NBody -t -x 19968 -q"),
     AMDBenchmarkCase("QuasiRandomSequence", "QuasiRandomSequence -q -t -y 10200 -x 10000"),
     AMDBenchmarkCase("RadixSort", "RadixSort -q -t -x 65536"),
     AMDBenchmarkCase("Reduction", "Reduction -q -t -x 400000000"),
     AMDBenchmarkCase("SimpleConvolution", "SimpleConvolution -q -t -x 512000"),
     #AMD SDK 2.9 benchmarks
     AMDBenchmarkCaseAsyncDataTransfer("AsyncDataTransfer_Sync", 
                                       "POCL_WORK_GROUP_METHOD=loopvec AsyncDataTransfer -t", 
                                       AMD2_9_CPP_CL_TEST_ROOT_DIR, 
                                       AMD2_9_BIN_DIR),
     AMDBenchmarkCaseAsyncDataTransfer("AsyncDataTransfer_Async", 
                                       "POCL_WORK_GROUP_METHOD=loopvec AsyncDataTransfer -t", 
                                       AMD2_9_CPP_CL_TEST_ROOT_DIR, 
                                       AMD2_9_BIN_DIR),
     AMDBenchmarkCaseKernelLauch("KernelLaunch", "KernelLaunch", 
                      AMD2_9_CL_TEST_ROOT_DIR, 
                      AMD2_9_BIN_DIR),
     AMDBenchmarkCaseTransferOverlap("TransferOverlap", "TransferOverlap -t", 
                      AMD2_9_CL_TEST_ROOT_DIR, AMD2_9_BIN_DIR),
     AMDBenchmarkCaseConcurrentKernel("ConcurrentKernel_Con", 
                                      "ConcurrentKernel -t",
                                      AMD2_9_CPP_CL_TEST_ROOT_DIR, 
                                      AMD2_9_BIN_DIR),
     AMDBenchmarkCaseConcurrentKernel("ConcurrentKernel_Seq", 
                                      "ConcurrentKernel -t",
                                      AMD2_9_CPP_CL_TEST_ROOT_DIR, 
                                      AMD2_9_BIN_DIR),
     KernelGrindBenchmarkCase("kernel_grind", 
                              "kernel_grind -wi 1 -wig 1 -i 20000 -dual", 
                              "/tests/benchmarks", "/"),
     KernelGrindBenchmarkCase("kernel_grind_in_order",
                              "kernel_grind_in_order -wi 1 -wig 1 -i 20000 -dual",
                              "/tests/benchmarks", "/"),
     KernelGrindBenchmarkCase("kernel_grind_no_deps",
                              "kernel_grind_no_dependencies -wi 1 -wig 1 -i 20000 -dual",
                              "/tests/benchmarks", "/"),

     KernelGrindBenchmarkCase("kernel_grind_r_only_buf",
                              "kernel_grind_r_only_buffer -wi 1 -wig 1 -i 20000",
                              "/tests/benchmarks", "/")]

    
# Benchmarks tuned for low memory targers. E.g., smaller input sets.
amd_benchmarks_lowmem = \
    [AMDBenchmarkCase("AESEncryptDecrypt", "AESEncryptDecrypt -t -q"),
     AMDBenchmarkCase("BitonicSort", "BitonicSort -q -t -x 1048576"),
     AMDBenchmarkCase("BinarySearch", "BinarySearch -q -t -x 8388608"),
     AMDBenchmarkCase("BinomialOption", "BinomialOption -q -t -x 10000"),
     AMDBenchmarkCase("BlackScholes", "BlackScholes -q -t -x 1694304"),
     AMDBenchmarkCase("DCT", "DCT -q -t -x 2000 -y 2000"),
     AMDBenchmarkCase("FastWalshTransform", "FastWalshTransform -q -t -x 1054432"),
     AMDBenchmarkCase("FloydWarshall", "FloydWarshall -q -t -x 512"),
     AMDBenchmarkCase("Histogram", "Histogram -t -x 1500 -y 1500 -q"),
     AMDBenchmarkCase("Mandelbrot", "Mandelbrot -t -x 8192 -y 8192 -q"),
     AMDBenchmarkCase("MatrixTranspose", "MatrixTranspose -t -x 1024 -y 1024 -q"),
     #This gives garbage execution times for some reason
     #AMDBenchmarkCase("MatrixMultiplication", "MatrixMultiplication -q -t -x 1024 -y 1024 -z 2048"),
     AMDBenchmarkCase("NBody", "NBody -t -x 19968 -q"),
     AMDBenchmarkCase("QuasiRandomSequence", "QuasiRandomSequence -q -t -y 1000 -x 1000"),
     #This is marked as XFAIL, but seems to work randomly when run here
     #AMDBenchmarkCase("RadixSort", "RadixSort -q -t -x 65536"),
     AMDBenchmarkCase("Reduction", "Reduction -q -t -x 5000000"),
     AMDBenchmarkCase("SimpleConvolution", "SimpleConvolution -q -t -x 128000")]

halide_benchmarks = \
  [HalideBenchmarkCase("Bilateral Grid", "bilateral_grid", 
                       "./filter ../images/gray.png out.png 0.1"),
   HalideBenchmarkCase("Interpolate", "interpolate", 
                       "./interpolate ../images/rgba.png out.png"),
   HalideBenchmarkCase("Local Laplacian", "local_laplacian", 
                       "./process ../images/rgb.png 8 1 1 out.png")]

halide_benchmarks_4k = \
  [HalideBenchmarkCase("Bilateral Grid", "bilateral_grid", 
                       "./filter ../../../buhd4k.png out.png 0.1"),
   HalideBenchmarkCase("Interpolate", "interpolate", 
                       "./interpolate ../../../buhd4k.png out.png"),
   HalideBenchmarkCase("Local Laplacian", "local_laplacian", 
                       "./process ../../../buhd4k.png 8 1 1 out.png")]

cloverleaf_benchmarks = \
    [CloverLeafBenchmarkCase("CloverLeaf")]

benchmark_sets = (("amdlowmem", amd_benchmarks_lowmem), 
                  ("amd", amd_benchmarks),
                  ("cloverleaf", cloverleaf_benchmarks),
                  ("halide", halide_benchmarks), 
                  ("halide4k", halide_benchmarks_4k), 
                  ("tlp", cloverleaf_benchmarks + halide_benchmarks_4k))


def print_environment_info():
    timeout, llvm_version, stderr, rc = run_cmd("llvm-config --version")

    llvm_version = llvm_version.strip()

    cpumodel = ""
    if os.path.exists("/proc/cpuinfo"):
        lines = open("/proc/cpuinfo").readlines()
        for line in lines:
            if "model name" in line:
                cpumodel = line.split(":")[1].strip()
                break
        #the PowerPC cpuinfo lacks "model name"
        if cpumodel == "":
            for line in lines:
                if "cpu" in line:
                    cpumodel = line.split(":")[1].strip()
                    break 

    sys.stdout.write("date: " + datetime.datetime.now().strftime("%Y-%m-%d") + "\n")
    sys.stdout.write("LLVM: " + llvm_version + "\n");
    sys.stdout.write(" CPU: " + cpumodel + "\n\n")


if __name__ == "__main__":

    parser = optparse.OptionParser('usage: %prog [options]')
    parser.add_option('--ocl_dir', type="string", metavar='dir', default="", 
                        help='Directory that contains comparison OCL .icd file')
    parser.add_option('-o', type="string", metavar='log file', dest='logfile', default="",
                        help='Write log to this file, instead of stdout')
    parser.add_option('--set', action="append", dest='benchmarks', 
                      choices=[x[0] for x in benchmark_sets],
                      help='Select the benchmark sets to run.')
    parser.add_option('--filter', type="string", dest="filter", default="",
                      help="Include only tests with the given string in the name.")

    args, free_args = parser.parse_args()
 
    #vendor_ocl_dir = sys.argv[1] if len(sys.argv) == 2 else None     
    vendor_ocl_dir = args.ocl_dir if args.ocl_dir != "" else None

    if args.logfile != "":
      sys.stdout = open(args.logfile, 'w')

    results = []

    print_environment_info()

    colwidths = (25, 8, 8, 8)

    sys.stdout.write("case".ljust(colwidths[0]))
    sys.stdout.write("pocl".ljust(colwidths[1]))
    if vendor_ocl_dir is not None:
        sys.stdout.write("vendor".ljust(colwidths[2]))
        #sys.stdout.write("pocl perf.".ljust(colwidths[3]))
    sys.stdout.write("\n")
    sys.stdout.flush()

    benchmarks = []
    for benchmark_set in benchmark_sets:
        name, benchmark_list = benchmark_set
        if name in args.benchmarks:
            benchmarks += benchmark_list 

    if args.filter != "":
        new_benchmarks = [x for x in benchmarks if args.filter in x.name]
        benchmarks = new_benchmarks

    for case in benchmarks:
        sys.stdout.write(case.name.ljust(colwidths[0]))
        sys.stdout.flush()

        os.environ['OCL_ICD_VENDORS'] = pocl_ocl_dir
        os.environ['POCL_BUILDING'] = '1'

        result_pocl = case.repeat(REPEAT_COUNT)
        sys.stdout.write(("%.3f" % result_pocl.kernel_run_time).ljust(colwidths[1]))
        sys.stdout.flush()

        if vendor_ocl_dir is None:
            results.append((case, result_pocl))
            sys.stdout.write("\n")
            continue

        os.environ['OCL_ICD_VENDORS'] = vendor_ocl_dir

        result_vendor = case.repeat(REPEAT_COUNT)

        sys.stdout.write(("%.3f" % result_vendor.kernel_run_time).ljust(colwidths[2]))
        sys.stdout.flush()                         

        speedup = result_vendor.kernel_run_time / result_pocl.kernel_run_time 

        sys.stdout.write(("%.2fx" % speedup).ljust(colwidths[3]))
        sys.stdout.write("\n")
        sys.stdout.flush()
        
        results.append((case, result_pocl, result_vendor))        
