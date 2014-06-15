#~ AT_SETUP([Scalar wave equation])
#~ AT_KEYWORDS([scalarwave])
#~ #this is a check for if doubles are available
#~ AT_SKIP_IF([grep DISABLE_LONG $abs_top_builddir/Makefile])
#~ AT_DATA([expout],
#~ [Initial condition: t=0
#~ Time step 1: t=0.03125
#~ Time step 2: t=0.0625
#~ Time step 3: t=0.09375
#~ Time step 4: t=0.125
#~ phi(0       ,0       ,0       ) = 0
#~ phi(0.0625  ,0.0625  ,0.0625  ) = 0.0057744
#~ phi(0.125   ,0.125   ,0.125   ) = 0.0435832
#~ phi(0.1875  ,0.1875  ,0.1875  ) = 0.133357
#~ phi(0.25    ,0.25    ,0.25    ) = 0.274951
#~ phi(0.3125  ,0.3125  ,0.3125  ) = 0.447032
#~ phi(0.375   ,0.375   ,0.375   ) = 0.613262
#~ phi(0.4375  ,0.4375  ,0.4375  ) = 0.733705
#~ phi(0.5     ,0.5     ,0.5     ) = 0.777678
#~ phi(0.5625  ,0.5625  ,0.5625  ) = 0.733705
#~ phi(0.625   ,0.625   ,0.625   ) = 0.613262
#~ phi(0.6875  ,0.6875  ,0.6875  ) = 0.447032
#~ phi(0.75    ,0.75    ,0.75    ) = 0.274951
#~ phi(0.8125  ,0.8125  ,0.8125  ) = 0.133357
#~ phi(0.875   ,0.875   ,0.875   ) = 0.0435832
#~ phi(0.9375  ,0.9375  ,0.9375  ) = 0.0057744
#~ phi(1       ,1       ,1       ) = 0
#~ Done.
#~ ])
#~ AT_CHECK([$abs_top_builddir/examples/scalarwave/scalarwave], 0, expout)
#~ AT_CLEANUP



#AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include -DSRCDIR='"$(abs_srcdir)"'
add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

#scalarwave_CFLAGS = -std=c99 @OPENCL_CFLAGS@
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 ${OPENCL_CFLAGS}")

add_executable("scalarwave" scalarwave.c scalarwave.cl)

#scalarwave_LDADD = -lm @OPENCL_LIBS@ ../../lib/poclu/libpoclu.la
target_link_libraries("scalarwave" ${POCLU_LINK_OPTIONS})

add_test("full_applications/scalarwave" "scalarwave")

set_tests_properties( "full_applications/scalarwave"
  PROPERTIES
    COST 3.0
    PASS_REGULAR_EXPRESSION "Done.\n"
    PROCESSORS 1
    LABELS "FullApplications"
    DEPENDS "pocl_version_check")
