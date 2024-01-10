
**************************
Release Notes for PoCL 5.1
**************************

========================
Driver-specific features
========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote: Basis for the coarse-grain SVM support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CG SVM support works only if the client manages to mmap() the
device-side allocated SVM pool to the same address as in the
server-side. This is a work-in-progress, but is usable for testing
client apps and libraries that require CG SVM as it seems to work
often enough.

===================================
Deprecation/feature removal notices
===================================
