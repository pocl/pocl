# converts babeltrace output to JSON for Chromium tracing
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
SYNC_EVENTS = Array.new
BUILDS = Hash.new

RECV = Hash.new
SENT = Hash.new

# (?<cmd_type>ndrange_kernel): { cpu_id = \d+ }, { event_id = (?<event_id>0x\h+), evt_status = (?<event_status>0x\h+), dev_id = (?<dev_id>0x\h+), queue_id = (?<queue_id>0x\h+), kernel_id = (?<kernel_id>0x\h+), kernel_name = "(?<kernel_name>.*)"
# ^(?<cmd_type>\w+): { cpu_id = \d+ }, { event_id = (?<event_id>0x\h+), evt_status = (?<event_status>0x\h+), dev_id = (?<dev_id>0x\h+), queue_id = (?<queue_id>0x\h+), obj_id = (?<obj_id>0x\h+)
# ^(?<cmd_type>\w+): { cpu_id = \d+ }, { event_id = (?<event_id>0x\h+), evt_status = (?<event_status>0x\h+), dev_id = (?<dev_id>0x\h+), queue_id = (?<queue_id>0x\h+), src_id = (?<src_id>0x\h+), dst_id = (?<dst_id>0x\h+)

# sync 1
# ^(?<cmd_type>\w+): { cpu_id = \d+ }, { context_id = (?<context_id>0x\h+), (queue|buffer|program|kernel|image)_id = (?<obj_id>0x\h+)

REGEXES = [
  /(?<cmd_type>ndrange_kernel): { cpu_id = \d+ }, { event_id = (?<event_id>0x\h+), evt_status = (?<event_status>0x\h+), dev_id = (?<dev_id>0x\h+), queue_id = (?<queue_id>0x\h+), kernel_id = (?<kernel_id>0x\h+), kernel_name = "(?<kernel_name>.*)"/,
  /^(?<cmd_type>\w+): { cpu_id = \d+ }, { event_id = (?<event_id>0x\h+), evt_status = (?<event_status>0x\h+), dev_id = (?<dev_id>0x\h+), queue_id = (?<queue_id>0x\h+), src_id = (?<src_id>0x\h+), dst_id = (?<dst_id>0x\h+)/,
  /^(?<cmd_type>\w+): { cpu_id = \d+ }, { event_id = (?<event_id>0x\h+), evt_status = (?<event_status>0x\h+), dev_id = (?<dev_id>0x\h+), queue_id = (?<queue_id>0x\h+), obj_id = (?<obj_id>0x\h+)/
]

# ^(?<cmd_type>msg_\w+): { cpu_id = \d+ }, { msg_id = (?<msg_id>0x\h+), event_id = (?<event_id>0x\h+), dev_id = (?<dev_id>0x\h+), queue_id = (?<queue_id>0x\h+)


def parsecommand(epoch_start, seconds, nanos, cmd)

  nanos = nanos.to_i
  seconds = seconds.to_i
  timestamp = (seconds - epoch_start)*1_000_000_000 + nanos

  match = nil

  # synchronous commands
  if /^build_program: { cpu_id = \d+ }, { context_id = (?<context_id>0x\h+), program_id = (?<obj_id>0x\h+)/ =~ cmd
    build_id = obj_id.to_i(16)
    h = BUILDS.delete build_id
    if h.nil?
      h = Hash.new
      h[:type] = :build_program
      h[:ctx] = context_id.to_i(16)
      h[:obj_id] = build_id
      h[:start] = timestamp
      BUILDS[build_id] = h
    else
      h[:endt] = timestamp
      SYNC_EVENTS << h
    end
    return
  end

  if /^(?<cmd_type>\w+): { cpu_id = \d+ }, { context_id = (?<context_id>0x\h+), (queue|buffer|program|image)_id = (?<obj_id>0x\h+)/ =~ cmd
    h = Hash.new
    h[:type] = cmd_type
    h[:ctx] = context_id.to_i(16)
    h[:obj_id] = obj_id.to_i(16)
    h[:ts] = timestamp
    SYNC_EVENTS << h
    return
  end

  if /^(?<cmd_type>\w+): { cpu_id = \d+ }, { context_id = (?<context_id>0x\h+), kernel_id = (?<kernel_id>0x\h+), kernel_name = "(?<kernel_name>.*)"/ =~ cmd
    h = Hash.new
    h[:type] = cmd_type
    h[:ctx] = context_id.to_i(16)
    h[:obj_id] = kernel_id.to_i(16)
    h[:ts] = timestamp
    h[:kernel_name] = kernel_name
    SYNC_EVENTS << h
    return
  end

  # network commands - sent/received
  match = /^(?<cmd_type>msg_\w+): { cpu_id = \d+ }, { msg_id = (?<msg_id>0x\h+), event_id = (?<event_id>0x\h+), dev_id = (?<dev_id>0x\h+), queue_id = (?<queue_id>0x\h+)/.match(cmd)
  if (match)
    event_id = match[:event_id].to_i(16)
    msg_id = match[:msg_id].to_i(16)
    dev_id = match[:dev_id].to_i(16) + 1
    queue_id = match[:queue_id].to_i(16)

    if match[:cmd_type] == "msg_received"
      RECV[event_id] = { msg_id: msg_id, dev_id: dev_id, queue_id: queue_id, timestamp: timestamp }
    else
      SENT[event_id] = { msg_id: msg_id, dev_id: dev_id, queue_id: queue_id, timestamp: timestamp }
    end

    return
  end

  REGEXES.each do |reg|
    match = reg.match cmd
    break if match
  end

  if match.nil?
    puts "UNKNOWN COMMAND: #{cmd}"
  else
#    puts "COMMAND: #{cmd}"
    event_id = match[:event_id].to_i(16)
    event_status = match[:event_status].to_i(16)
    cmd_type = match[:cmd_type]

    dev_id = match[:dev_id].to_i(16) + 1
    queue_id = match[:queue_id].to_i(16)

    meta = Hash.new
    n = match.names
    meta[:obj_id] = match[:obj_id].to_i(16) if n.include? "obj_id"
    meta[:src_id] = match[:src_id].to_i(16) if n.include? "src_id"
    meta[:dst_id] = match[:dst_id].to_i(16) if n.include? "dst_id"
    meta[:kernel_id] = match[:kernel_id].to_i(16) if n.include? "kernel_id"
    meta[:kernel_name] = match[:kernel_name] if n.include? "kernel_name"

#    puts "EP: #{epoch_start} EID: #{event_id} SEC: #{seconds} NANOS: #{nanos} TS: #{timestamp}"

    eid = event_id
    est = 3 - event_status
    est = 0 if est < 0

    if EVENTS[eid].nil?
      h = Hash.new
      h[:stat] = Array.new(4)
      # for NDRange commands, put the kernel name in the "type"
      if cmd_type == 'ndrange_kernel'
        h[:type] = meta[:kernel_name].to_sym
      else
        h[:type] = cmd_type.to_sym
      end
      h[:stat][est] = timestamp

      h[:dev_id] = dev_id if dev_id
      h[:queue_id] = queue_id if queue_id

      h[:meta] = meta
      EVENTS[eid] = h
    else
      h = EVENTS[eid]
      h[:stat][est] = timestamp
    end

  end

end

##############################################################################

lines = 0
epoch = nil

ARGV.each do |file|

puts "Parsing #{file}"

in_f = File.open(file, "rb")

in_f.each_line do |line|

  line.strip!
  next if line.empty?

  if /^\[(?<seconds>\d+)\.(?<nanos>\d+)\]\s+\(\+[0-9?]+\.[0-9?]+\)\s+(?<hostname>[\S]+)\s+pocl_trace:(?<command>.*)/ =~ line

    if epoch.nil?
      epoch = seconds.to_i
    end

    parsecommand(epoch, seconds, nanos, command)

  else
    puts "Line did not 1st match: \n"
    puts line
  end

#   break if lines > 2000

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

SYNC_EVENTS.each do |ev|

  event_json['name'] = ev.delete :type
  event_json['pid'] = 0
  event_json['tid'] = ev.delete :ctx

  if event_json['name'] == :build_program
    ev_start = ev.delete :start
    ev_end = ev.delete :endt
  else
    ev_start = ev.delete :ts
    ev_end = ev_start + 100000
  end

  if slice_start and slice_end
    next if (ev_start / 1000) < slice_start
    next if (ev_end / 1000) > slice_end
  end

  event_json['args'] = ev

  event_json['ph'] = 'B'
  event_json['ts'] = ev_start / 1000

  out_f.puts ',' if first_done
  first_done = true
  out_f.puts event_json.to_json

  event_json['ph'] = 'E'
  event_json['ts'] = ev_end / 1000

  out_f.puts ',' if first_done
  out_f.puts event_json.to_json

end

###############################################

event_ids = EVENTS.keys
event_ids.sort!

event_ids.each do |event_id|
#  puts EVENTS[event_id].inspect

  e = EVENTS[event_id]

  if RECV[event_id]
    e[:recv] = RECV[event_id][:timestamp]
    e[:recv_msg] = RECV[event_id][:msg_id]
  end

  if SENT[event_id]
    e[:sent] = SENT[event_id][:timestamp]
    e[:sent_msg] = SENT[event_id][:msg_id]
  end

  stat = e[:stat]

  # skip events with no timing or incomplete timing
  next if stat.nil? or stat[3].nil?

  if slice_start and slice_end
#    puts "ST: #{(stat[0] / 1000)} END: #{(stat[3] / 1000)}"
    next if (stat[0] / 1000) < slice_start
    next if (stat[3] / 1000) > slice_end
  end

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

  ts = stat[1]
  event_json['ph'] = 'I'
  event_json['ts'] = ts / 1000
  out_f.puts ',' if first_done
  out_f.puts event_json.to_json

  ts = stat[2]
  event_json['ph'] = 'I'
  event_json['ts'] = ts / 1000
  out_f.puts ',' if first_done
  out_f.puts event_json.to_json

  ts = stat[3]
  event_json['ph'] = 'E'
  event_json['ts'] = ts / 1000
  out_f.puts ',' if first_done
  out_f.puts event_json.to_json

end

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
