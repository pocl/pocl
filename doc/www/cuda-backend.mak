<%!
        sub_page = "NVIDIA GPU support via CUDA backend"
%>

<%inherit file="basic_page.makt" />

<h1>April 2017: NVIDIA GPU support via CUDA backend</h1>

<p>
pocl now has experimental support for NVIDIA GPU devices via a new backend
which makes use of the LLVM NVPTX backend and the CUDA driver API.
This work was primarily carried out by James Price from the
<a href="http://uob-hpc.github.io">High Performance Computing group</a> at the
University of Bristol.
</p>


<h2>Status</h2>

<p>
Instructions for building and using the CUDA backend can be found in the
<a href="http://portablecl.org/docs/html/cuda.html">user manual</a>.
</p>

<p>
Although this backend is still a work in progress, many core features are
implemented, allowing real OpenCL applications to be run.
For example, we can run all of the OpenCL benchmarks from
<a href="https://github.com/vetter/shoc/wiki">SHOC</a>, with the exception of
those that require support for image types.
The performance on the SHOC benchmarks compared to NVIDIA's own OpenCL
implementation is shown below.
While there is still a lot of room for improvement in a few of these benchmarks,
many of them achieve performance close to the NVIDIA driver, and in one case
exceeds it.
At this stage we have only been focusing on implementing functionality, so we
believe there is the potential to significantly improve the performance of this
backend in the future.
</p>

<img src="img/pocl-nvidia-SHOC-April17.png" border="0" style="vertical-align: middle;" />


<p>
One key advantage of having an open source alternative to the proprietary NVIDIA
OpenCL implementation is our ability to add support for things that NVIDIA
doesn't.
For example, this backend allows us to run SPIR-based applications on NVIDIA
devices, such as SYCL codes compiled with
<a href="https://www.codeplay.com/products/computesuite/computecpp">Codeplay's
ComputeCpp compiler</a>.
We can also use this backend on ARM-based platforms with NVIDIA GPUs, such as
the Jetson TK1 and TX1 development boards, which NVIDIA doesn't publicly release
OpenCL support for.
In the future, this could extend to adding support for OpenCL subgroups, SPIR-V
consumption, or other features from recent versions of the OpenCL standard.
Finally, since this backend makes use of CUDA under-the-hood, we can also use
all of the CUDA development tools that NVIDIA provide (such as their visual
profiler), many of which currently don't support OpenCL directly.
</p>

<h4>Known limitations (at the time of writing):</h4>
<ul>
<li>image types and samplers are unimplemented</li>
<li>atomics are unimplemented</li>
<li>global offsets are unimplemented</li>
<li>get_work_dim() is unimplemented</li>
<li>printf format support is incomplete</li>
</ul>


<h2>Contributing</h2>

<p>
We welcome any contributions in the form of bug reports and pull requests.
In particular, we are keen to see contributions that fill in the remaining
functionality, as well as performance improvements.
If you're interested in helping out but aren't sure what to work on, drop into
the pocl IRC or get in touch with James Price for more information.
</p>
