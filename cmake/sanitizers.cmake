# currently only works with gcc as host compiler
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_C_COMPILER_VERSION VERSION_GREATER "4.7.99")
    option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
    option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
  else()
    set(ENABLE_ASAN OFF CACHE INTERNAL "Enable AddressSanitizer")
    set(ENABLE_TSAN OFF CACHE INTERNAL "Enable ThreadSanitizer")
  endif()

  if (CMAKE_C_COMPILER_VERSION VERSION_GREATER "4.8.99")
    option(ENABLE_UBSAN "Enable UBSanitizer" OFF)
  else()
    set(ENABLE_UBSAN OFF CACHE INTERNAL "Enable UBSanitizer")
  endif()

  if (CMAKE_C_COMPILER_VERSION VERSION_GREATER "5.0.99")
    option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
  else()
    set(ENABLE_LSAN OFF CACHE INTERNAL "Enable LeakSanitizer")
  endif()

else()
  set(ENABLE_ASAN OFF CACHE INTERNAL "Enable AddressSanitizer")
  set(ENABLE_TSAN OFF CACHE INTERNAL "Enable ThreadSanitizer")
endif()


set(SANITIZER_OPTIONS "")

if(ENABLE_ASAN)
  if("${CMAKE_C_COMPILER_VERSION}" VERSION_LESS "6.0.0")
    list(APPEND SANITIZER_OPTIONS "-fsanitize=address")
  else()
    list(APPEND SANITIZER_OPTIONS "-fsanitize=address" "-fsanitize-recover=address")
  endif()
  list(APPEND SANITIZER_LIBS "asan")
endif()

if(ENABLE_LSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=leak")
  list(APPEND SANITIZER_LIBS "lsan")
endif()

if(ENABLE_TSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=thread")
  list(APPEND SANITIZER_LIBS "tsan")
endif()

if(ENABLE_UBSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=undefined")
  list(APPEND SANITIZER_LIBS "ubsan")
endif()

if(SANITIZER_OPTIONS)
  list(APPEND SANITIZER_OPTIONS "-fno-omit-frame-pointer")
  add_compile_options(${SANITIZER_OPTIONS})
endif()


# Unfortunately the way CMake tests work, if they're given
# a pass/fail expression, they don't check for exit status.
# This was causing some false negatives with ASan (test was
# returning with 1, but CMake reported it as pass because
# the pass expression was present in output).
if(ENABLE_ASAN OR ENABLE_TSAN OR ENABLE_UBSAN OR ENABLE_LSAN)
  set(ENABLE_ANYSAN 1)
endif()
