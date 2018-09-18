<%!
        sub_page = "pocl and the Matrix-2000 co-processor"
%>

<%inherit file="basic_page.makt" />

<p><i>
The Matrix-2000 architecture is designed to replace the Intel Xeon Phi accelerators of the TianHe-2 supercomputer. While the Matrix-2000 accelerator provides the potential for higher performance, its potential can only be realized if the software can make effective use of it. To this end, the Compiler Lab from <a href="http://www.nudt.edu.cn/index_eng.htm">National University of Defense Technology (NUDT)</a> provides an efficient OpenCL implementation for such an architecture.</p>

<p>Overall, our OpenCL implementation is built on top of LLVM v5.0 and reuses code pieces from POCL as well. Thanks very much for the excellent work in POCL. When starting to develop our own OpenCL implementation, we have selected POCL with the CUDA support as the starting point. Based on this, we have customized POCL to adapt to the Matrix-2000 architecture. In particular, we have rewritten the device-side runtime scheduler, as well as reimplementing all the OpenCL APIs based on the Xeon-Matrix2000 driver. To unlock the hardware potential, our device runtime uses a push-based task dispatching strategy and the performance of the kernel atomics is improved significantly. This new implementation has led to a lot of coding work and thus we give it a new name "MOCL", which is detailed in our article <a href="https://jianbinfang.github.io/files/2018-03-15-mocl.pdf">Zhang et al. MOCL: an efficient OpenCL implementation for the matrix-2000 architecture.</a> CF 2018: 26-35".</p>

<p>This MOCL framework has been deployed on the TH-2A system and is readily available to the public. In the near future, we plan to upstream our improved runtime scheduler to POCL, so as to provide users with more options. Again, we thank the very nice work in POCL for this community. We sincerely thank Dr. Pekka J&auml;&auml;skel&auml;inen from Tampere University of Technology, Dr. James Price from Bristol University, and other POCL contributors, for their hands-on help.</p>

<p>Regards,<br/>
Dr. Jianbin Fang<br/>
2018-09-18<br/>
</p>
</i>

<p>As always, if you have found any other interesting use cases for pocl code base, <a href="http://portablecl.org/discussion.html">please let us know</a>, we'll be more than happy to post them to pocl web page.</p>
