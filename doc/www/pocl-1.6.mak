<%!
        sub_page = "Portable Computing Language (pocl) v1.6 released"
%>
<%inherit file="basic_page.makt" />

<h1>October 9, 2020: pocl v1.6 released</h2>

<h2>Release Highlights</h2>

<h3>Improved CUDA performance and features</h3>

In pocl v1.6, CUDA backend gained several performance improvements.
One optimization was the use of static CUDA memory blocks for OpenCL's constant __local
blocks. Previous version of pocl one dynamic shared CUDA memory block for
OpenCL's constant __local blocks and __local function arguments which resulted in
poor SASS code generation due to a pointer aliasing issue. pocl 1.6 fixes this to use
static CUDA memory blocks when there's no __local function arguments in the OpenCL
kernel. Changing the inlining thresholds in LLVM, also resuted in better PTX code
generation. Another improvement was to use 32 bit addresses when accessing CUDA shared
memory as 32-bit addresses are more than enough to address the CUDA shared memory space.

<a href="https://github.com/vetter/shoc/wiki">SHOC</a> benchmarks shows that these optimizations
resulted in much better performance for FFT and GEMM benchmarks compared to last benchmark
run given <a href="http://portablecl.org/cuda-backend.html">here</a>. There are still a few
more SHOC benchmarks where pocl does not reach the same level of performance as the
NVIDIA OpenCL driver we welcome any contributions to improve them.

<img src="img/pocl-nvidia-SHOC-October20.png" border="0" style="vertical-align: middle;" />

Additional features added in this release include support for more special functions
including frexp, tgamma, ldexp, modf and remquo. clEnqueueFillBuffer functionality for
CUDA backend was also partially implemented for this release.

<h3>PowerPC support</h3>

Pocl v1.6 brings back support for PowerPC 8/9 with test suite passing fully on the pthread
device and the CUDA device test suite pass rate is the same as the pass rate for CUDA
on an x86_64 machine. Pocl fills the gap of running OpenCL codes on PowerPC machines
as NVIDIA does not provide an OpenCL backend for PowerPC machines and IBM's OpenCL CPU
implementation is deprecated in favour of pocl. This was tested on a PowerPC node with a
Tesla V100 on Lawrence Livermore National Laboratory's Lassen supercomputer.

<h3>Improved packaging support</h3>

In previous pocl releases, distributing a pocl binary built with various devices support
required that the build machine and the host machine have the same support for the devices.
With pocl v1.6, pocl can be built with as many device support as possible on the build machine
and when transferred to the host machine pocl will check for device support at runtime.
This has enabled the conda package manager to distribute pocl binary packages with CUDA
support to be distributed for Linux-x86_64 and Linux-ppc64le. Pre-built packages of pocl are
available via <a href="https://github.com/conda-forge/pocl-feedstock">conda package manager</a>
for Linux-x86_64, Linux-ppc64le, Linux-aarch64 and Darwin-x86_64.

<p>A more detailed changelog <a href="http://portablecl.org/downloads/CHANGES">here</a>.

<h2>Acknowledgements</h2>

<p>
Part of Isuru Fernando's work on CUDA enhancements was supported by a grant from the
National Science Foundation and Isuru would like to thank Matt Wala, Nick Christensen,
and, Andreas Klöckner from the Scientific Computing group at University of Illinois
at Urbana-Champaign for their assistance.

</p>

<p><a href="download.html">Download</a>.</p>

