prefix=@CMAKE_INSTALL_PREFIX@
exec_prefix=@POCL_INSTALL_PUBLIC_BINDIR@
libdir=@POCL_INSTALL_PUBLIC_LIBDIR@
includedir=@POCL_INSTALL_PUBLIC_HEADER_DIR@

Name: Portable Computing Language
Description: Portable Computing Language
Version: @POCL_VERSION@
Libs: -L${libdir} -lpocl -I@POCL_INSTALL_PRIVATE_HEADER_DIR@ @LD_FLAGS_BIN@
Cflags: -I${includedir}

