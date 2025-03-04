# adapted code from:
# https://github.com/vector-of-bool/cmrc/blob/master/CMakeRC.cmake
# using MIT license.
#
# This command generates a C header file with the content of another file
# as a C string (const char*)
#
# similar to the tool xxd, however xxd is problematic for 2 reasons,
# 1) another dependency 2) the option to choose the name of the C string
# is not available in all version of xxd (e.g. ubuntu 22.04 doesn't have it)

if(NOT name)
  message(FATAL_ERROR "name is required argument")
endif()
if(NOT input)
  message(FATAL_ERROR "input is required argument")
endif()
if(NOT output)
  message(FATAL_ERROR "output is required argument")
endif()

# Read in the digits
file(READ "${input}" bytes HEX)
# Format each pair into a character literal. Heuristics seem to favor doing
# the conversion in groups of five for fastest conversion
string(REGEX REPLACE "(..)(..)(..)(..)(..)"
  "\n          '\\\\x\\1','\\\\x\\2','\\\\x\\3','\\\\x\\4','\\\\x\\5',"
  chars "${bytes}")
# Since we did this in groups, we have some leftovers to clean up
string(LENGTH "${bytes}" n_bytes2)
math(EXPR n_bytes "${n_bytes2} / 2")
math(EXPR remainder "${n_bytes} % 5") # '5' is the grouping count from above
set(cleanup_re "$")
set(cleanup_sub )
while(remainder)
  set(cleanup_re "(..)${cleanup_re}")
  set(cleanup_sub "'\\\\x\\${remainder}',${cleanup_sub}")
  math(EXPR remainder "${remainder} - 1")
endwhile()
if(NOT cleanup_re STREQUAL "$")
  string(REGEX REPLACE "${cleanup_re}" "${cleanup_sub}" chars "${chars}")
endif()
string(CONFIGURE [[
  const char @name@[] = { @chars@ 0 };
  const unsigned @name@_len = @n_bytes@;]] code)
file(WRITE "${output}" "${code}")
