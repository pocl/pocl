
*****************************
Release Notes for PoCL 5.0
*****************************

=============================
Major new features
=============================

~~~~~~~~~~~~~
Remote Driver
~~~~~~~~~~~~~

PoCL now has a new backend (called 'remote') for offloading OpenCL commands
across a network to one or more servers that are running the (also newly
added) 'pocld' daemon. See the `announcement <http://portablecl.org/remote-backend.html>`
and the `documentation <http://portablecl.org/docs/html/remote.html>` for details.

=============================
Bugfixes and minor features
=============================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver (partial) OpenCL 3.0 support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA driver has gained some new features:

* program scope variables
* OpenCL 2.x/3.x atomics
* initial subgroup support (only intel_sub_group_shuffle, intel_sub_group_shuffle_xor,
  get_sub_group_local_id, sub_group_barrier, sub_group_ballot are supported)
* enable FP16 & generic address space support (with SPIR-V input)


================
Acknowledgements
================

Customized Parallel Computing (CPC) research group of Tampere University,
Finland leads the development of PoCL on the side and for the needs of
their research projects. This project has received funding from the ECSEL
Joint Undertaking (JU) under grant agreement No 783162 (FitOptiVis), European
Union's Horizon 2020 research and innovation programme under Grant Agreement
No 871738 (CPSoSaware), Academy of Finland (decision #331344) and Business
Finland's AISA Veturi project. The financial support is very much appreciated
-- it keeps this open source project going!
