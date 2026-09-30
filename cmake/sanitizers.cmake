# Sanitizers work with GCC and Clang as the host compiler.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
  option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
  option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
  option(ENABLE_UBSAN "Enable UBSanitizer" OFF)
  option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
endif()

# Unfortunately the way CMake tests work, if they're given
# a pass/fail expression, they don't check for exit status.
# This was causing some false negatives with ASan (test was
# returning with 1, but CMake reported it as pass because
# the pass expression was present in output).
if(ENABLE_ASAN OR ENABLE_TSAN OR ENABLE_UBSAN OR ENABLE_LSAN)
  set(ENABLE_ANYSAN 1)
endif()

if(ENABLE_ANYSAN)
  if(NOT (CMAKE_C_COMPILER_ID STREQUAL "GNU"
          OR CMAKE_C_COMPILER_ID STREQUAL "Clang"))
    message(FATAL_ERROR "Sanitizers require GCC or Clang as the host compiler")
  endif()
  if(NOT CMAKE_C_COMPILER_ID STREQUAL CMAKE_CXX_COMPILER_ID)
    message(FATAL_ERROR "Sanitizers need the same C and C++ compiler, got "
                        "${CMAKE_C_COMPILER_ID} and ${CMAKE_CXX_COMPILER_ID}")
  endif()
  if(ENABLE_TSAN AND (ENABLE_ASAN OR ENABLE_LSAN))
    message(FATAL_ERROR "ThreadSanitizer cannot be combined with "
                        "AddressSanitizer or LeakSanitizer")
  endif()
endif()


set(SANITIZER_OPTIONS "")

if(ENABLE_ASAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=address" "-fsanitize-recover=address")
endif()

if(ENABLE_LSAN)
  if(ENABLE_ASAN)
    message(STATUS "LeakSanitizer is part of AddressSanitizer")
  else()
    list(APPEND SANITIZER_OPTIONS "-fsanitize=leak")
  endif()
endif()

if(ENABLE_TSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=thread")
endif()

if(ENABLE_UBSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=undefined")
  # Clang's -fsanitize=function incorrectly crashes at kernel launch. GCC does not
  # have the flag so nothing occurs
  check_c_compiler_flag("-fno-sanitize=function" HAVE_FNO_SANITIZE_FUNCTION)
  if(HAVE_FNO_SANITIZE_FUNCTION)
    list(APPEND SANITIZER_OPTIONS "-fno-sanitize=function")
  endif()
endif()

if(SANITIZER_OPTIONS)
  list(APPEND SANITIZER_OPTIONS "-fno-omit-frame-pointer")
  add_compile_options(${SANITIZER_OPTIONS})
  add_link_options(${SANITIZER_OPTIONS})
  string(JOIN " " SANITIZER_FLAGS_STR ${SANITIZER_OPTIONS})
endif()
