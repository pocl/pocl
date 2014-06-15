#~ AT_SETUP([Trigonometric functions])
#~ AT_DATA([expout],
#~ [OK
#~ ])
#~ AT_CHECK([$abs_top_builddir/examples/trig/trig], 0, expout)
#~ AT_CLEANUP

#AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include -DSRCDIR='"$(abs_srcdir)"'
add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

# trig_CFLAGS = @OPENCL_CFLAGS@
add_compile_options(${OPENCL_CFLAGS})


add_executable("trig" trig.c trig.cl trig_exec.c)

# trig_LDADD = @OPENCL_LIBS@ -lm ../../lib/poclu/libpoclu.la
target_link_libraries("trig" ${POCLU_LINK_OPTIONS})

add_test("examples/trig" "trig")

set_tests_properties(
  PROPERTIES
    COST 3.0
    PASS_REGULAR_EXPRESSION "OK\n"
    PROCESSORS 1
    LABELS "OpenCL_Spec"
    DEPENDS "pocl_version_check")
