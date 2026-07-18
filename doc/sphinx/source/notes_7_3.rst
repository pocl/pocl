**************************
Release Notes for PoCL 7.3
**************************

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Make event synchronisation (clFinish and clWaitEvents) thread safe with non
  threaded queue handling
* Remove threaded queue handling and POCL_CUDA_DISABLE_QUEUE_THREADS environment 
  variable
