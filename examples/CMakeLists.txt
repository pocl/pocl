if(NOT DEFINED ENABLED_TESTSUITES)
  message(STATUS "Disabling all testsuites")
else()

  include(ExternalProject)

  if("${ENABLED_TESTSUITES}" STREQUAL "all")
    message(STATUS "Enabling all testsuites")
    set(ENABLED_TESTSUITES ${ALL_TESTSUITES})
  endif()

  foreach(TESTSUITE IN LISTS ENABLED_TESTSUITES)
    if(ALL_TESTSUITES MATCHES ${TESTSUITE})
      message(STATUS "Enabling testsuite ${TESTSUITE}")
      add_subdirectory("${TESTSUITE}")
    else()
      message(WARNING "Unknown testsuite ${TESTSUITE} requested")
    endif()
  endforeach()

endif()
set(ENABLED_TESTSUITES ${ENABLED_TESTSUITES} PARENT_SCOPE)


add_subdirectory("example1")
add_subdirectory("example1-spir32")
add_subdirectory("example1-spir64")
add_subdirectory("example2")
add_subdirectory("example2a")
add_subdirectory("standalone")

