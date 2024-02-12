#!/usr/bin/env python3

# A python script to parse text traces generated with the LTTng tracer
# on pocl. For usage run './babel_parse.py -h'

import re
import json
import argparse


# different regexes that parse the arguments of a lttng trace

pocl_msg_regex = re.compile(
    "^{ msg_id = (?P<msg_id>0x[0-9A-Fa-f]+), event_id = (?P<event_id>0x[0-9A-Fa-f]+), local_did = (?P<local_did>0x[0-9A-Fa-f]+), remote_did = (?P<remote_did>0x[0-9A-Fa-f]+), type = (?P<type>0x[0-9A-Fa-f]+), status = (?P<status>0x[0-9A-Fa-f]+) }$")

pocl_free_object_regex = re.compile(
    "^{ context_id = (?P<context_id>0x[0-9A-Fa-f]+), (buffer|kernel|queue|program|)_id = (?P<id>0x[0-9A-Fa-f]+) }$")

pocl_kernel_create_regex = re.compile(
    "^{ context_id = (?P<context_id>0x[0-9A-Fa-f]+), kernel_id = (?P<id>0x[0-9A-Fa-f]+), kernel_name = \"(?P<kernel_name>\S+)\" }$")

pocl_buffer_regex = re.compile(
    "^{ event_id = (?P<event_id>0x[0-9A-Fa-f]+), evt_status = (?P<evt_status>0x[0-9A-Fa-f]+), dev_id = (?P<dev_id>0x[0-9A-Fa-f]+), queue_id = (?P<queue_id>0x[0-9A-Fa-f]+), obj_id = (?P<obj_id>0x[0-9A-Fa-f]+) }$")

pocl_ndrange_kernel_regex = re.compile(
    "^{ event_id = (?P<event_id>0x[0-9A-Fa-f]+), evt_status = (?P<evt_status>0x[0-9A-Fa-f]+), dev_id = (?P<dev_id>0x[0-9A-Fa-f]+), queue_id = (?P<queue_id>0x[0-9A-Fa-f]+), kernel_id = (?P<kernel_id>0x[0-9A-Fa-f]+), kernel_name = \"(?P<kernel_name>\S+)\" }$")

pocl_copy_regex = re.compile(
    "^{ event_id = (?P<event_id>0x[0-9A-Fa-f]+), evt_status = (?P<evt_status>0x[0-9A-Fa-f]+), dev_id = (?P<dev_id>0x[0-9A-Fa-f]+), queue_id = (?P<queue_id>0x[0-9A-Fa-f]+), src_id = (?P<src_id>0x[0-9A-Fa-f]+), dest_id = (?P<dest_id>0x[0-9A-Fa-f]+) }$")


def pocl_parse_status_and_return(dict, event_status):
    """
    A function that parses the event status and creates
    dictionaries with different phases set based on the status
    @param dict The dictionary with most values already set
    @param event_status the cl status of the event
    @return a list of dictionaries with their phase set
    """
    if event_status == 0:
        # if the status is cl_finish, also add a line to
        # indicate the enqueue bar is done and exit
        dict_copy = dict.copy()
        dict_copy["ph"] = "e"

        dict["name"] = "running"
        dict["ph"] = "e"
        return [dict, dict_copy]
    elif event_status == 1:
        dict_copy = dict.copy()
        dict_copy["name"] = "running"
        dict_copy["ph"] = "b"

        dict["name"] = "submitted"
        dict["ph"] = "e"
        return [dict, dict_copy]
    elif event_status == 2:
        dict_copy = dict.copy()
        dict_copy["name"] = "submitted"
        dict_copy["ph"] = "b"

        dict["name"] = "queued"
        dict["ph"] = "e"
        return [dict, dict_copy]
    else:
        # add an overshadowing bar for the entire
        # execution
        dict_copy = dict.copy()
        dict_copy["ph"] = "b"

        dict["name"] = "queued"
        dict["ph"] = "b"
        return [dict_copy, dict]


def set_dict_values(dict, inp_dict):
    """
    function that sets the common values of cl commands
    @param dict the dictionary to be filled
    @param inp_dict the dictionary with common values
    that are copied to the dict
    """
    dict["cat"] = "event"
    dict["id"] = int(inp_dict["event_id"], 16)
    dict["pid"] = int(inp_dict["dev_id"], 16)
    dict["args"] = inp_dict


def parse_pocl_trace_data(trace_type, string, dict):
    """
    function to parse trace specific data.
    @param trace_type: string with the name of the trace
    @param string: string to be parsed
    @param dict: resulting dictionary with relevant key value pairs
    """

    if (trace_type == "msg_received" or
            trace_type == "msg_sent"):
        res = pocl_msg_regex.match(string)
        if res is None:
            print("could not parse msg_received trace point, possibly the type has changed?")
            return None

        res_dict = res.groupdict()

        dict["cat"] = "event"
        dict["id"] = int(res_dict["event_id"], 16)

        dict["pid"] = int(res_dict["local_did"], 16)
        status = int(res_dict["status"], 16)

        # set a different phase type depending on
        # the message type and status
        if trace_type == "msg_received":
            if status == 0:
                dict["ph"] = "b"
            elif status == 1:
                dict["ph"] = "n"
            elif status == 2:
                dict["ph"] = "e"
            else:
                print("Unknown msg_received status")
                return None
        else:
            status = int(res_dict["status"], 16)
            if status == 0:
                dict["ph"] = "b"
            elif status == 1:
                dict["ph"] = "e"
            else:
                print("Unknown msg_received status")
                return None

        dict["args"] = res_dict
        return [dict]

    elif (trace_type == "free_buffer" or
          trace_type == "free_queue" or
          trace_type == "free_program" or
          trace_type == "create_buffer" or
          trace_type == "create_queue" or
          trace_type == "build_program" or
          trace_type == "create_program"):
        res = pocl_free_object_regex.match(string)

        if res is None:
            print("could not parse pocl trace point: " + string)
            return None

        res_dict = res.groupdict()
        dict["cat"] = "event"
        dict["id"] = int(res_dict["id"], 16)
        dict["pid"] = int(res_dict["context_id"], 16)
        dict["ph"] = "i"
        dict["args"] = res_dict
        return [dict]

    elif (trace_type == "free_kernel" or
          trace_type == "create_kernel"):
        res = pocl_kernel_create_regex.match(string)

        if res is None:
            print("could not parse pocl trace point: " + string)
            return None

        res_dict = res.groupdict()
        dict["cat"] = "event"
        dict["name"] = "kernel:" + res_dict["kernel_name"]
        dict["id"] = int(res_dict["id"], 16)
        dict["pid"] = int(res_dict["context_id"], 16)
        dict["ph"] = "i"
        dict["args"] = res_dict
        return [dict]

    elif (trace_type == "read_buffer" or
          trace_type == "write_buffer" or
          trace_type == "fill_buffer" or
          trace_type == "read_buffer_rect" or
          trace_type == "write_buffer_rect" or
          trace_type == "read_image_rect" or
          trace_type == "write_image_rect" or
          trace_type == "fill_image" or
          trace_type == "map_image" or
          trace_type == "map_buffer" or
          trace_type == "unmap_memobj"
    ):

        res = pocl_buffer_regex.match(string)
        if res is None:
            print("could not parse buffer trace point: " + string)
            return None

        res_dict = res.groupdict()
        set_dict_values(dict, res_dict)
        event_status = int(res_dict["evt_status"], 16)

        return pocl_parse_status_and_return(dict, event_status)

    elif trace_type == "ndrange_kernel":

        res = pocl_ndrange_kernel_regex.match(string)
        if res is None:
            print("could not parse ndrange kernel trace point: " + string)
            return None

        res_dict = res.groupdict()
        set_dict_values(dict, res_dict)
        dict["name"] = res_dict["kernel_name"]

        event_status = int(res_dict["evt_status"], 16)

        return pocl_parse_status_and_return(dict, event_status)

    elif (trace_type == "copy_image_rect" or
          trace_type == "copy_buffer" or
          trace_type == "copy_buffer_rect" or
          trace_type == "copy_image2buf" or
          trace_type == "copy_buf2image"):

        res = pocl_copy_regex.match(string)
        if res is None:
            print("could not parse buffer trace point: " + string)
            return None

        res_dict = res.groupdict()
        set_dict_values(dict, res_dict)
        event_status = int(res_dict["evt_status"], 16)
        return pocl_parse_status_and_return(dict, event_status)

    else:
        print("pocl_trace: unknown message type: " + trace_type)
        return None


pocld_msg_regex = re.compile(
    "^{ msg_id = (?P<msg_id>0x[0-9A-Fa-f]+), dev_id = (?P<dev_id>0x[0-9A-Fa-f]+), queue_id = (?P<queue_id>0x[0-9A-Fa-f]+), type = (?P<type>[0-9]+) }$")

pocld_buffer_regex = re.compile(
    "^{ msg_id = (?P<msg_id>0x[0-9A-Fa-f]+), dev_id = (?P<dev_id>0x[0-9A-Fa-f]+), queue_id = (?P<queue_id>0x[0-9A-Fa-f]+), buffer_id = (?P<buffer_id>0x[0-9A-Fa-f]+), (write_|read_|)size = (?P<size>0x[0-9A-Fa-f]+), event_status = (?P<event_status>[0-9]+) }$")

pocld_ndrange_kernel_regex = re.compile(
    "^{ msg_id = (?P<msg_id>0x[0-9A-Fa-f]+), dev_id = (?P<dev_id>0x[0-9A-Fa-f]+), queue_id = (?P<queue_id>0x[0-9A-Fa-f]+), kernel_id = (?P<kernel_id>0x[0-9A-Fa-f]+), event_status = (?P<event_status>[0-9]+) }$")

pocld_free_regex = re.compile(
    "^{ msg_id = (?P<msg_id>0x[0-9A-Fa-f]+), dev_id = (?P<dev_id>0x[0-9A-Fa-f]+), (buffer|kernel|queue|program|)_id = (?P<id>0x[0-9A-Fa-f]+) }$")


def pocld_parse_event_status(status):
    """
    Convert the status number to an appropriate phase
    letter.
    @param status: pocl status integer
    @return: char representing the phase
    """
    # for some reason pocld can have the running status
    # set to either 1 or 2
    if status == 1 or status == 2:
        return "b"
    # corresponds to cl_finish
    elif status == 3:
        return "e"
    else:
        print("unknown event status: {}".format(status))
        return "e"


def parse_pocld_trace_data(trace_type, string, dict):
    """
    function to parse trace specific data.
    @param trace_type: string with the name of the trace
    @param string: string to be parsed
    @param dict: resulting dictionary with relevant key value pairs
    """

    # prefix used to indicate that the device is not local
    pocld_pid_prefix = 1000

    if (trace_type == "msg_received" or
            trace_type == "msg_sent"):
        res = pocld_msg_regex.match(string)

        if res is None:
            print("could not parse pocld msg_received trace point")
            return None

        res_dict = res.groupdict()
        dict["cat"] = "msg"
        dict["id"] = int(res_dict["msg_id"], 16)
        dict["pid"] = int(res_dict["dev_id"], 16) + pocld_pid_prefix
        dict["ph"] = "n"
        dict["args"] = res_dict
        return [dict]

    elif (trace_type == "write_buffer" or
          trace_type == "read_buffer" or
          trace_type == "fill_buffer"):
        res = pocld_buffer_regex.match(string)

        if res is None:
            print("could not parse pocld write_buffer trace point")
            return None

        res_dict = res.groupdict()
        dict["cat"] = "event"
        dict["id"] = int(res_dict["msg_id"], 16)
        dict["pid"] = int(res_dict["dev_id"], 16) + pocld_pid_prefix
        dict["ph"] = pocld_parse_event_status(int(res_dict["event_status"]))
        dict["args"] = res_dict
        return [dict]

    elif (trace_type == "ndrange_kernel"):
        res = pocld_ndrange_kernel_regex.match(string)

        if res is None:
            print("could not parse pocld write_buffer trace point")
            return None

        res_dict = res.groupdict()
        dict["cat"] = "event"
        dict["id"] = int(res_dict["msg_id"], 16)
        dict["pid"] = int(res_dict["dev_id"], 16) + pocld_pid_prefix
        dict["ph"] = pocld_parse_event_status(int(res_dict["event_status"]))
        dict["args"] = res_dict
        return [dict]

    elif (trace_type == "free_buffer" or
          trace_type == "free_kernel" or
          trace_type == "free_queue" or
          trace_type == "free_program" or
          trace_type == "create_buffer" or
          trace_type == "create_kernel" or
          trace_type == "create_queue" or
          trace_type == "build_program"):
        res = pocld_free_regex.match(string)

        if res is None:
            print("could not parse pocld trace point: " + string)
            return None

        res_dict = res.groupdict()
        dict["cat"] = "event"
        dict["id"] = int(res_dict["msg_id"], 16)
        dict["pid"] = int(res_dict["dev_id"], 16) + pocld_pid_prefix
        dict["ph"] = "i"
        dict["args"] = res_dict
        return [dict]

    else:
        print("pocld_trace: unknown message type: " + trace_type)
        return None


line_regex = re.compile(
    "^\[(?P<seconds>\d+)\.(?P<nanos>\d+)\]\s+\(\+[0-9?]+\.[0-9?]+\)\s+(?P<hostname>[\S]+)\s+(?P<tracer_name>.*):(?P<trace_type>.*): { cpu_id = \d+ },\s(?P<trace_data>.*)$")
# used to store the earliest timestamp recorded
# which is used as the starting point
start_time = 0


def parse_line(line):
    """
    take a raw line that has been parsed by babelparse
    and extract all the relevant information from it
    @param line: raw line from babelparse
    @return: a dictionary with required key value pairs
    for the chrome trace format
    """
    global start_time
    regex_result = line_regex.match(line)
    if regex_result:
        regex_dict = regex_result.groupdict()
        if not (regex_dict["tracer_name"] == "pocl_trace" or
                regex_dict["tracer_name"] == "pocld_trace"):
            print("other tracer are not supported yet")
            return None

        ts = int(regex_dict["seconds"]) * 1_000_000_000 + int(regex_dict["nanos"]) - start_time
        if start_time == 0:
            start_time = ts
            ts = 0

        # ts divided by 1000 since chrome profiler expects time microseconds
        ret_dict = {"ts": ts / 1000, "name": regex_dict["trace_type"]}
        if regex_dict["tracer_name"] == "pocl_trace":
            parsed_lines = parse_pocl_trace_data(regex_dict["trace_type"], regex_dict["trace_data"], ret_dict)
        else:
            parsed_lines = parse_pocld_trace_data(regex_dict["trace_type"], regex_dict["trace_data"], ret_dict)

        if parsed_lines:
            return parsed_lines
        return None
    else:
        print("could not parse line, unknown format")
        print(line)
        return None


def main():
    # parse the command line arguments
    parser = argparse.ArgumentParser(description="Parse LTTng text trace files.")
    parser.add_argument("input_file", metavar="input_file", nargs="?",
                        default="trace.txt",
                        help="the name of the input file that has already been parsed with babel parse.")
    parser.add_argument("--out", "-o", metavar="output_file",
                        default="trace.json", nargs="?",
                        help="the name of the output file. Default: trace.json")

    args = vars(parser.parse_args())

    input_file = open(args.get("input_file"), "rt")
    output_file = open(args.get("out"), "wt")
    parse_count = 0
    try:
        output_file.write("[ \n")

        # read the first line, but don't start it with a ','
        first_line = input_file.readline()
        result = parse_line(first_line)
        if result:
            for res in result:
                output_file.write(json.dumps(res) + "\n")
        parse_count = 1
        for line in input_file:
            result = parse_line(line)
            if result:
                for res in result:
                    output_file.write("," + json.dumps(res) + "\n")
                parse_count += 1

        output_file.write("] \n")
    finally:
        input_file.close()
        output_file.close()
        print("parsed {} lines, exiting".format(parse_count))


if __name__ == "__main__":
    main()
