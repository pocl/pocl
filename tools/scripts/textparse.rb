# converts output from PoCL's text tracer to JSON for Chromium tracing
#
# idea from:
# https://www.gamasutra.com/view/news/176420/Indepth_Using_Chrometracing_to_view_your_inline_profiling_data.php

require 'tempfile'
require 'json'
require 'getoptlong'

opts = GetoptLong.new(
  [ '--help', '-h', GetoptLong::NO_ARGUMENT ],
  [ '--output', '-o', GetoptLong::REQUIRED_ARGUMENT ],
  [ '--start', '-s', GetoptLong::OPTIONAL_ARGUMENT ],
  [ '--end', '-e', GetoptLong::OPTIONAL_ARGUMENT ]
)

slice_start = nil
slice_end = nil
in_f = nil
out_f = nil

opts.each do |opt, arg|
  case opt
    when '--help'
      puts "USAGE: $0 [-s START] [-e END] -o output_file input_file"
      puts "START and END are optional timestamps (seconds.fraction) for slicing out a portion of trace. Trace begins at 0.0"
    when '--output'
      out_f = File.open(arg, "wb")
    when '--start'
      slice_start = (arg.to_f * 1000000).to_i
    when '--end'
      slice_end = (arg.to_f * 1000000).to_i
  end
end

puts "Input files:"
puts ARGV.inspect

if (ARGV.size == 0)
  puts "No input files specified"
  exit 1
end

if out_f.nil?
  puts "Output not specified"
  exit 1
end

if slice_start and slice_end
  puts "START: #{slice_start} usec END: #{slice_end} usec"
end

EVENTS = Hash.new
DEPS_FROM = Array.new
DEPS_TO = Array.new

NANOS_IN_SEC = 1_000_000_000

# 109287478675743 | EV ID 1 | DEV 1 | CQ 4 | ndrange_kernel | queued | KERNEL ID 6 | name=ML_BSSN_CL_RHS1
COMMAND_REGEXP = /^(?<timestamp>\d+) \| EV ID (?<event_id>\d+) \| DEV (?<dev_id>\d+) \| CQ (?<queue_id>\d+) \| (?<command_type>\w+) \| (?<evt_status>\w+) \|(?<remainder>.*)/

# DEP | EV ID 1 -> EV ID 2
DEP_REGEXP = /DEP \| EV ID (?<src_id>\d+) -> EV ID (?<dst_id>\d+)/

# KERNEL ID 6 | name=ML_BSSN_CL_RHS1
NDR_REGEXP = /KERNEL ID (?<kernel_id>\d+) \| name=(?<kernel_name>[\.\w]+)/

# MEM ID 34 | size=2744000 | host_ptr=0x5574f7c0b990
# unmap_mem_object | complete | MEM ID 21
MEM_ID_REGEXP = /MEM ID (?<mem_id>\d+)/

# MEM ID FROM %" PRIu64 " | MEM ID TO %" PRIu64 "
COPY_REGEXP = /MEM ID FROM (?<src_id>\d+) \| MEM ID TO (?<dst_id>\d+)/

# MEM ID %" PRIu64 " | FROM DEV %" PRIu64 " | TO DEV %" PRIu64 "
MIGR_REGEXP = /MEM 0 ID (?<mem_id>\d+) \| FROM DEV (?<src_id>\w+) \| TO DEV (?<dst_id>\w+)/
NOP_REGEXP = /MEM 0 ID (?<mem_id>\d+) \| NOP MIGRATION DEV/

def parsecommand(timestamp, match)

    event_id = match[:event_id].to_i
    dev_id = match[:dev_id].to_i
    queue_id = match[:queue_id].to_i

    if event_id.nil? or dev_id.nil? or queue_id.nil?
      raise StandardError.new "unknown event 1"
    end

    cmd_type = match[:command_type]
    event_status = match[:evt_status]
    remainder = match[:remainder]

    event_numeric_status = nil
    case event_status
      when "queued"
        event_numeric_status = 0
      when "submitted"
        event_numeric_status = 1
      when "running"
        event_numeric_status = 2
      when "complete"
        event_numeric_status = 3
      else
        # map errors (<0) to complete
        event_numeric_status = 3
    end

    meta = Hash.new

    case cmd_type
      when "ndrange_kernel"
        match = NDR_REGEXP.match remainder
        raise StandardError.new "parse error NDR" unless match
        meta[:kernel_id] = match[:kernel_id]
        meta[:kernel_name] = match[:kernel_name]
      when "read_buffer", "write_buffer", "fill_buffer", "map_buffer", "unmap_mem_object", "read_buffer_rect", "write_buffer_rect"
        match = MEM_ID_REGEXP.match remainder
        raise StandardError.new "parse error MEM" unless match
        meta[:mem_id] = match[:mem_id]
      when "copy_buffer", "copy_buffer_rect"
        match = COPY_REGEXP.match remainder
        raise StandardError.new "parse error COPY" unless match
        meta[:src_id] = match[:src_id]
        meta[:dst_id] = match[:dst_id]
      when "migrate_mem_objects"
        match = NOP_REGEXP.match remainder
        if match
        meta[:mem_id] = match[:mem_id]
        meta[:type] = "nop migration"
        else
        match = MIGR_REGEXP.match remainder
        raise StandardError.new "parse error MIGR" unless match
        meta[:mem_id] = match[:mem_id]
        meta[:src_id] = match[:src_id]
        meta[:dst_id] = match[:dst_id]
        end
      when "barrier", "svm_memcpy"
        meta[:type] = "empty"
      else
        puts "unknown command type: #{cmd_type}"
    end

    h = EVENTS[event_id]
    if h.nil?
      h = Hash.new
      EVENTS[event_id] = h
    end

    if h[:dev_id].nil?
      h[:event_id] = event_id
      h[:dev_id] = dev_id
      h[:queue_id] = queue_id
      # for NDRange commands, put the kernel name in the "type"
      if cmd_type == 'ndrange_kernel'
        h[:type] = meta[:kernel_name].to_sym
      else
        h[:type] = cmd_type.to_sym
      end

      h[:stat] = Array.new(4)
      h[:meta] = meta
    end

    h[:stat][event_numeric_status] = timestamp

end

def parsedeps(dep_match)

    src_id = dep_match[:src_id].to_i
    dst_id = dep_match[:dst_id].to_i

    h = EVENTS[dst_id]
    if h.nil?
      h = Hash.new
      EVENTS[dst_id] = h
    end

    if h[:deps].nil?
      h[:deps] = Array.new
    end

    h[:deps] << src_id

    DEPS_FROM << src_id
    DEPS_TO << dst_id

end

##############################################################################

lines = 0
starting_timestamp = nil

ARGV.each do |file|

    puts "Parsing #{file}"

    in_f = File.open(file, "rb")

    in_f.each_line do |line|

      line.strip!
      next if line.empty?

      cmd_match = COMMAND_REGEXP.match line
      dep_match = DEP_REGEXP.match line

      if cmd_match

        ts = cmd_match[:timestamp].to_i
        secs = ts / NANOS_IN_SEC
        nsecs = ts % NANOS_IN_SEC

        if starting_timestamp.nil?
          starting_timestamp = ts
        end

        cmd_timestamp = ts - starting_timestamp
        parsecommand(cmd_timestamp, cmd_match)

      elsif dep_match

        parsedeps(dep_match)

      else

        puts "Line did not match anything: \n"
        puts line
      end

      #break if lines > 2000
      lines += 1

    end

    puts "Parsed lines: #{lines}"

    in_f.close

end

#############################################################

=begin

    'cat' – the category for this event. Useful when doing larger grouping (eg "UnitUpdates")
    'name' – the name of this event (eg 'PathingUpdate')
    'pid' – the processor ID that spawned this event
    'tid' – the thread ID that spawned this event
    'ts' – the processor time stamp at the time this event was created
    'ph' – the phase or type of this event
    'args' – any programmatic metadata that's attached to this event

 cat = pocl / pold TODO something more useful ?
 name = cmdtype
 pid = device
 tid = queue
 ts  = timestamp
 ph = status
 args = all metadata

=end

event_json = Hash.new
event_json['cat'] = 'pocl'

out_f.puts '['

first_done = false

###############################################
###############################################

event_ids = EVENTS.keys
event_ids.sort!

event_ids.each do |event_id|
#  puts EVENTS[event_id].inspect

  e = EVENTS[event_id]

  stat = e[:stat]

  # skip broken events
  next if e[:dev_id].nil? or e[:queue_id].nil? or e[:event_id].nil?

  # skip events with no timing or incomplete timing
  next if stat.nil? or stat[3].nil?

  if slice_start and slice_end
#    puts "ST: #{(stat[0] / 1000)} END: #{(stat[3] / 1000)}"
    next if (stat[0] / 1000) < slice_start
    next if (stat[3] / 1000) > slice_end
  end

  event_json['id'] = e[:event_id]
  event_json['cat'] = "pocl"
  event_json['name'] = e[:type]
  event_json['pid'] = e[:dev_id]
  event_json['tid'] = e[:queue_id]
  event_json['args'] = e[:meta]

  ts = stat[2]
  event_json['ph'] = 'B'
  event_json['ts'] = ts / 1000
  out_f.puts ',' if first_done
  first_done = true
  out_f.puts event_json.to_json

  #ts = stat[1]
  #event_json['ph'] = 'I'
  #event_json['ts'] = ts / 1000
  #out_f.puts ',' if first_done
  #out_f.puts event_json.to_json

  #ts = stat[2]
  #event_json['ph'] = 'I'
  #event_json['ts'] = ts / 1000
  #out_f.puts ',' if first_done
  #out_f.puts event_json.to_json

  ts = stat[3]
  event_json['ph'] = 'E'
  event_json['ts'] = ts / 1000
  out_f.puts ',' if first_done
  out_f.puts event_json.to_json

end

=begin
dep_count = DEPS_FROM.size
(0...dep_count).each do |i|

  src_id = DEPS_FROM[i]
  dst_id = DEPS_TO[i]

#  puts "SRC DST ID: "
#  puts src_id.inspect
#  puts dst_id.inspect

#  event_json['pid'] = e[:dev_id]
#  event_json['tid'] = e[:queue_id]

  event_json = Hash.new

  event_json['id'] = src_id
  event_json['cat'] = "pocl"
  event_json['ph'] = 's'
  event_json['pid'] = EVENTS[src_id][:dev_id]
  event_json['tid'] = EVENTS[src_id][:queue_id]
#  ts = EVENTS[src_id][:stat][3]
#  event_json['ts'] = ts / 1000

  out_f.puts ','
  out_f.puts event_json.to_json

  event_json = Hash.new

  event_json['id'] = dst_id
  event_json['cat'] = "pocl"
  event_json['ph'] = 'f'
  event_json['pid'] = EVENTS[dst_id][:dev_id]
  event_json['tid'] = EVENTS[dst_id][:queue_id]
#  ts = EVENTS[dst_id][:stat][0]
#  event_json['ts'] = ts / 1000

  out_f.puts ','
  out_f.puts event_json.to_json

end
=end

###############################################

out_f.puts ']'
out_f.close

############################################

=begin
  host_que = stat[1] - stat[0]
  dev_que = stat[2] - stat[1]
  on_dev = stat[3] - stat[2]

  stat = Hash.new
  stat[:start] = stat[0]
  stat[:host_que] = host_que
  stat[:dev_que] = dev_que
  stat[:on_dev] = on_dev

  e[:stat] = stat

#  puts e.inspect
#  puts "EVENT #{event_id}: HQ #{host_que} DQ #{dev_que} D #{on_dev}"

=end
