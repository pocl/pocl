#/usr/bin/ruby

# input:
# find all gtest executables in Glow build directory (e.g. OpenCLBackendCorrectnessTest,
# OpenCLGradCheckTest and so on); put them into /tmp/tests; then:
# for f in `cat /tmp/tests`; do echo "########## $f"; ./$f  --gtest_list_tests ; done >/tmp/test_list_input
# ... then run this script with /tmp/test_list_input as only argument

file = ARGV[0]

lines = File.readlines(file)

# puts lines.inspect

def process(lines, tests)
  executable = lines.shift
  raise "unknown executable" unless executable.start_with? "######"
  executable.delete! "########## "
  executable.strip!
  while lines[0] and !lines[0].start_with?("#####") do
    part1 = lines.shift
    part1.strip!
    raise "part1 wrong" if part1.start_with? '  '
    while lines[0] and lines[0].start_with?('  ') do
      part2 = lines.shift
      part2.strip!
      if part2.end_with? '# GetParam() = "OpenCL"'
        part2 = part2[0..-26]
      end
      if part2.end_with? '# GetParam() = ("OpenCL")'
        part2 = part2[0..-28]
      end
      test = %Q{
      add_test(NAME "glow/#{executable}_#{part1}#{part2}"
               COMMAND "${TS_BUILDDIR}/tests/#{executable}" "--gtest_filter=#{part1}#{part2}")
}
      puts test
      tests << "    glow/#{executable}_#{part1}#{part2}\n"
    end
  end
end

tests = []
while !lines.empty? do
  puts "\n"
  puts lines[0]
  raise "unknown executable" unless lines[0].start_with? "######"
  process(lines, tests)
end

puts %Q{
  set_tests_properties(
#{tests.join}
  PROPERTIES
    LABELS "glow")
}
