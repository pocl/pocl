<%!
        sub_page = "Portable Computing Language (pocl) v1.6 released"
%>
<%inherit file="basic_page.makt" />

<h1>October 9, 2020: pocl v1.6 released</h2>

<h2>Release Highlights</h2>

<h3>Improved CUDA performance and features</h3>

In pocl v1.6, the CUDA backend gained several performance improvements.
Benchmarks using <a href="https://github.com/vetter/shoc/wiki">SHOC</a> benchmarks
(now <a href="https://github.com/pocl/pocl/pull/847">continually tested</a>
show that these optimizations resulted in much better performance,
particularly for benchmarks involving local memory such as FFT and GEMM, when compared
to a <a href="http://portablecl.org/cuda-backend.html">prior benchmark run</a>.
Pocl now often attains performance competitive with Nvidia's
proprietary OpenCL driver. We welcome contributions to identifying
and removing the root causes for any remaining problem areas.

<img src="img/pocl-nvidia-SHOC-October20.png" border="0" style="vertical-align: middle;" />

In particular, the following optimizations and improvements landed in the CUDA backend:

- Use 32-bit pointer arithmetic for local memory
  <a href="https://github.com/pocl/pocl/pull/822">#822</a>
- Use static CUDA memory blocks for OpenCL's constant <tt>__local</tt>
  blocks. Previous version of pocl used one dynamic shared CUDA memory block for
  OpenCL's constant <tt>__local</tt> blocks and <tt>__local</tt> function arguments.
  This resulted in poor SASS code generation due to a pointer aliasing issue.
  <a href="https://github.com/pocl/pocl/pull/838">#838</a>,
  <a href="https://github.com/pocl/pocl/pull/846">#846</a>,
  <a href="https://github.com/pocl/pocl/pull/824">#824</a>
- Use a higher unroll threshold in LLVM <a href="https://github.com/pocl/pocl/pull/826">#826</a>
- Implement more special functions <a href="https://github.com/pocl/pocl/pull/836">#836</a>
- Improve clEnqueueFillBufer <a href="https://github.com/pocl/pocl/pull/834">#834</a>

<h3>PowerPC support</h3>

Pocl v1.6 brings back support for PowerPC 8/9 with the internal test suite passing fully on the
pthread device and the CUDA device test suite pass rate is the same as the pass rate for CUDA
on an x86_64 machine. Pocl fills the gap of running OpenCL codes on PowerPC machines
as IBM's OpenCL CPU implementation is deprecated. This was tested on a PowerPC node with a
Tesla V100 on Lawrence Livermore National Laboratory's Lassen supercomputer.

<h3>Improved packaging support</h3>

In previous pocl releases, distributing a pocl binary built with various devices support
required that the build machine and the host machine have the same support for the devices.
With pocl v1.6, pocl can be compiled with as all devices enabled at build time,
and it will then check these devices for availability at run time.
This has enabled the conda package manager to distribute pocl binary packages with CUDA
support to be distributed for Linux-x86_64 and Linux-ppc64le. Pre-built packages of pocl are
available via the <a href="https://github.com/conda-forge/pocl-feedstock">conda-forge community package repository</a>
for Linux-x86_64, Linux-ppc64le, Linux-aarch64 and Darwin-x86_64
via the <a href="https://docs.conda.io/en/latest/">Conda user-level pacakge manager</a>.

<p>A more detailed changelog can be found <a href="http://portablecl.org/downloads/CHANGES">here</a>.

<h2>Acknowledgments</h2>

<p>
CUDA improvements, PowerPC support and packaging support described in this post were made by
Isuru Fernando and Matt Wala with assistance from Nick Christensen, and Andreas Klöckner,
all part of the Department of Computer Science at the University of Illinois at Urbana-Champaign.
The work was partially supported through awards OAC-1931577 and SHF-1911019 from the
US National Science Foundation, as well as award DE-NA0003963 from the US Department of Energy.
</p>

<p><a href="download.html">Download</a>.</p>

