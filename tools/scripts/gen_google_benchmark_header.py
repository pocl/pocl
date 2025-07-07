#!/usr/bin/python3

# generates a JSON file with google-benchmark-like results

# input: a file with one JSON hash per line,
# output: JSON header + input file + closing braces

# google benchmark JSON format example:

# {
#  "context": {
#    "date": "2015/03/17-18:40:25",
#    "num_cpus": 40,
#    "mhz_per_cpu": 2801,
#    "cpu_scaling_enabled": false,
#    "build_type": "debug"
#  },
#  "benchmarks": [
#    {
#      "name": "BM_SetInsert/1024/1",
#      "iterations": 94877,
#      "real_time": 29275,
#      "cpu_time": 29836,
#      "bytes_per_second": 134066,
#      "items_per_second": 33516
#    },
#    ...
#    ]
# }

import sys
import subprocess
import argparse
import multiprocessing
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='gen_google_benchmark_json',
                        description='generates a google benchmark JSON file with input JSON data')
    parser.add_argument('-i', '--input', required=True, help='Path to input JSON file')
    parser.add_argument('-o', '--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()

    content = ""
    with open(args.input) as my_file:
        content = my_file.read()

    input = content.strip()
    if input[-1] == ',':
        input = input[0:-1]

    ncpus = multiprocessing.cpu_count()
    now = datetime.datetime.now()
    f = open(args.output, 'w')

    header = """
{{
  "context": {{
    "date": "{}",
    "num_cpus": {},
    "mhz_per_cpu": 2000,
    "cpu_scaling_enabled": false,
    "build_type": "debug"
  }},
  "benchmarks": [
"""
    footer = """
  ]
}"""
    f.write(header.format(str(now), ncpus))
    f.write(input)
    f.write(footer)
