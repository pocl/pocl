<%!
        sub_page = "Portable Computing Language (PoCL) v4.0 released"
%>
<%inherit file="basic_page.makt" />

<h1>June 22, 2023: Portable Computing Language (PoCL) v4.0 released</h1>

<title>Release Notes for PoCL 4.0</title>

<div class="document" id="release-notes-for-pocl-4-0">
<h1 class="title">Release Notes for PoCL 4.0</h1>

<div class="section" id="major-new-features">
<h1>Major new features</h1>
<div class="section" id="support-for-clang-llvm-16-0">
<h2>Support for Clang/LLVM 16.0</h2>
<p>PoCL now supports Clang/LLVM from 10.0 to 16.0 inclusive. The most PoCL-relevant
change of the new 16.0 release is support for <a class="reference external" href="https://releases.llvm.org/16.0.0/tools/clang/docs/LanguageExtensions.html#half-precision-floating-point">_Float16 type on x86 and ARM targets.</a></p>
</div>
<div class="section" id="cpu-driver">
<h2>CPU driver</h2>
<div class="section" id="support-for-program-scope-variables">
<h3>Support for program-scope variables</h3>
<p>Global variables in program-scope are now supported, along with static global
variables in function-scope, for both OpenCL C source and SPIR-V compilation. The implementation passes
the <tt class="docutils literal">basic/test_basic</tt> test of the OpenCL-CTS, and has been tested with
client applications through chipStar.</p>
<pre class="code c literal-block">
<span class="name">global</span><span class="whitespace"> </span><span class="keyword type">float</span><span class="whitespace"> </span><span class="name">testGlobalVar</span><span class="punctuation">[</span><span class="literal number integer">128</span><span class="punctuation">];</span><span class="whitespace">

</span><span class="name">__kernel</span><span class="whitespace"> </span><span class="keyword type">void</span><span class="whitespace"> </span><span class="name">test1</span><span class="whitespace"> </span><span class="punctuation">(</span><span class="name">__global</span><span class="whitespace"> </span><span class="keyword">const</span><span class="whitespace"> </span><span class="keyword type">float</span><span class="whitespace"> </span><span class="operator">*</span><span class="name">a</span><span class="punctuation">)</span><span class="whitespace"> </span><span class="punctuation">{</span><span class="whitespace">
  </span><span class="keyword type">size_t</span><span class="whitespace"> </span><span class="name">i</span><span class="whitespace"> </span><span class="operator">=</span><span class="whitespace"> </span><span class="name">get_global_id</span><span class="punctuation">(</span><span class="literal number integer">0</span><span class="punctuation">)</span><span class="whitespace"> </span><span class="operator">%</span><span class="whitespace"> </span><span class="literal number integer">128</span><span class="punctuation">;</span><span class="whitespace">
  </span><span class="name">testGlobalVar</span><span class="punctuation">[</span><span class="name">i</span><span class="punctuation">]</span><span class="whitespace"> </span><span class="operator">+=</span><span class="whitespace"> </span><span class="name">a</span><span class="punctuation">[</span><span class="name">i</span><span class="punctuation">];</span><span class="whitespace">
</span><span class="punctuation">}</span><span class="whitespace">

</span><span class="name">__kernel</span><span class="whitespace"> </span><span class="keyword type">void</span><span class="whitespace"> </span><span class="name">test2</span><span class="whitespace"> </span><span class="punctuation">(</span><span class="name">__global</span><span class="whitespace"> </span><span class="keyword">const</span><span class="whitespace"> </span><span class="keyword type">float</span><span class="whitespace"> </span><span class="operator">*</span><span class="name">a</span><span class="punctuation">)</span><span class="whitespace"> </span><span class="punctuation">{</span><span class="whitespace">
  </span><span class="keyword type">size_t</span><span class="whitespace"> </span><span class="name">i</span><span class="whitespace"> </span><span class="operator">=</span><span class="whitespace"> </span><span class="name">get_global_id</span><span class="punctuation">(</span><span class="literal number integer">0</span><span class="punctuation">)</span><span class="whitespace"> </span><span class="operator">%</span><span class="whitespace"> </span><span class="literal number integer">128</span><span class="punctuation">;</span><span class="whitespace">
  </span><span class="name">testGlobalVar</span><span class="punctuation">[</span><span class="name">i</span><span class="punctuation">]</span><span class="whitespace"> </span><span class="operator">*=</span><span class="whitespace"> </span><span class="name">a</span><span class="punctuation">[</span><span class="name">i</span><span class="punctuation">];</span><span class="whitespace">
</span><span class="punctuation">}</span><span class="whitespace">

</span><span class="name">__kernel</span><span class="whitespace"> </span><span class="keyword type">void</span><span class="whitespace"> </span><span class="name">test3</span><span class="whitespace"> </span><span class="punctuation">(</span><span class="name">__global</span><span class="whitespace"> </span><span class="keyword type">float</span><span class="whitespace"> </span><span class="operator">*</span><span class="name">out</span><span class="punctuation">)</span><span class="whitespace"> </span><span class="punctuation">{</span><span class="whitespace">
  </span><span class="keyword type">size_t</span><span class="whitespace"> </span><span class="name">i</span><span class="whitespace"> </span><span class="operator">=</span><span class="whitespace"> </span><span class="name">get_global_id</span><span class="punctuation">(</span><span class="literal number integer">0</span><span class="punctuation">)</span><span class="whitespace"> </span><span class="operator">%</span><span class="whitespace"> </span><span class="literal number integer">128</span><span class="punctuation">;</span><span class="whitespace">
  </span><span class="name">out</span><span class="punctuation">[</span><span class="name">i</span><span class="punctuation">]</span><span class="whitespace"> </span><span class="operator">=</span><span class="whitespace"> </span><span class="name">testGlobalVar</span><span class="punctuation">[</span><span class="name">i</span><span class="punctuation">];</span><span class="whitespace">
</span><span class="punctuation">}</span>
</pre>
</div>
<div class="section" id="support-for-generic-address-space">
<h3>Support for generic address space</h3>
<p>Generic AS is now supported, for both OpenCL C source and SPIR-V compilation.
PoCL now passes the <tt class="docutils literal">generic_address_space/test_generic_address_space</tt> test
of the OpenCL-CTS, and has been tested with CUDA/HIP applications through chipStar.</p>
<pre class="code c literal-block">
<span class="keyword type">int</span><span class="whitespace"> </span><span class="name function">isOdd</span><span class="punctuation">(</span><span class="keyword type">int</span><span class="whitespace"> </span><span class="operator">*</span><span class="name">val</span><span class="punctuation">)</span><span class="whitespace"> </span><span class="punctuation">{</span><span class="whitespace">
  </span><span class="keyword">return</span><span class="whitespace"> </span><span class="name">val</span><span class="punctuation">[</span><span class="literal number integer">0</span><span class="punctuation">]</span><span class="whitespace"> </span><span class="operator">%</span><span class="whitespace"> </span><span class="literal number integer">2</span><span class="punctuation">;</span><span class="whitespace">
</span><span class="punctuation">}</span><span class="whitespace">

</span><span class="name">__kernel</span><span class="whitespace"> </span><span class="keyword type">void</span><span class="whitespace"> </span><span class="name">test3</span><span class="whitespace"> </span><span class="punctuation">(</span><span class="name">__global</span><span class="whitespace"> </span><span class="keyword type">int</span><span class="whitespace"> </span><span class="operator">*</span><span class="name">in1</span><span class="punctuation">,</span><span class="whitespace"> </span><span class="name">__local</span><span class="whitespace"> </span><span class="keyword type">int</span><span class="whitespace"> </span><span class="operator">*</span><span class="name">in2</span><span class="punctuation">,</span><span class="whitespace"> </span><span class="name">__global</span><span class="whitespace"> </span><span class="keyword type">int</span><span class="whitespace"> </span><span class="operator">*</span><span class="name">out</span><span class="punctuation">)</span><span class="whitespace"> </span><span class="punctuation">{</span><span class="whitespace">
  </span><span class="keyword type">size_t</span><span class="whitespace"> </span><span class="name">i</span><span class="whitespace"> </span><span class="operator">=</span><span class="whitespace"> </span><span class="name">get_global_id</span><span class="punctuation">(</span><span class="literal number integer">0</span><span class="punctuation">);</span><span class="whitespace">
  </span><span class="name">out</span><span class="punctuation">[</span><span class="name">i</span><span class="punctuation">]</span><span class="whitespace"> </span><span class="operator">=</span><span class="whitespace"> </span><span class="name">isOdd</span><span class="punctuation">(</span><span class="name">in1</span><span class="operator">+</span><span class="name">i</span><span class="punctuation">)</span><span class="whitespace"> </span><span class="operator">+</span><span class="whitespace"> </span><span class="name">isOdd</span><span class="punctuation">(</span><span class="name">in2</span><span class="operator">+</span><span class="punctuation">(</span><span class="name">i</span><span class="whitespace"> </span><span class="operator">%</span><span class="whitespace"> </span><span class="literal number integer">128</span><span class="punctuation">)];</span><span class="whitespace">
</span><span class="punctuation">}</span>
</pre>
</div>
<div class="section" id="initial-support-for-cl-khr-subgroups">
<h3>Initial support for cl_khr_subgroups</h3>
<p>The default is a single subgroup that always executes the whole X-dimension's WIs.
Independent forward progress is not yet supported, but it's
not needed for CTS compliance, due to the corner case of only one SG in flight.</p>
<p>Additionally, there is partial implementation for <tt class="docutils literal">cl_khr_subgroup_shuffle</tt>,
<tt class="docutils literal">cl_intel_subgroups</tt> and <tt class="docutils literal">cl_khr_subgroup_ballot with caveats</tt>:</p>
<blockquote>
<ul class="simple">
<li><tt class="docutils literal">cl_khr_subgroup_shuffle</tt>: Passes the CTS, but only because it doesn't test
non-uniform(lock-step) behavior, see:
<a class="reference external" href="https://github.com/KhronosGroup/OpenCL-CTS/issues/1236">https://github.com/KhronosGroup/OpenCL-CTS/issues/1236</a></li>
<li><tt class="docutils literal">cl_khr_subgroup_ballot</tt>: sub_group_ballot() works for uniform calls, the rest
are unimplemented.</li>
<li><tt class="docutils literal">cl_intel_subgroups</tt>: The block reads/writes are unimplemented.</li>
</ul>
</blockquote>
</div>
<div class="section" id="initial-support-for-cl-intel-required-subgroup-size">
<h3>Initial support for cl_intel_required_subgroup_size</h3>
<p>This extension allows the programmer to specify the required subgroup size for
a kernel function. This can be important for algorithm correctness in some cases. It's used by chipStar to implement fixed width warps when needed. The programmer
can specify the size with a new kernel attribute:
<tt class="docutils literal"><span class="pre">__attribute__((intel_reqd_sub_group_size(&lt;int&gt;)))</span></tt></p>
<p>PoCL additionally implements <tt class="docutils literal">CL_DEVICE_SUB_GROUP_SIZES_INTEL</tt> parameter for <tt class="docutils literal">clGetDeviceInfo</tt> API,
however <tt class="docutils literal">CL_​KERNEL_​SPILL_​MEM_​SIZE_​INTEL</tt> and <tt class="docutils literal">CL_​KERNEL_​COMPILE_​SUB_​GROUP_​SIZE_​INTEL</tt> for
<tt class="docutils literal">clGetKernelWorkGroupInfo</tt> API are not yet implemented.</p>
</div>
<div class="section" id="initial-support-for-cl-khr-fp16">
<h3>Initial support for cl_khr_fp16</h3>
<p>PoCL now has partial support for <tt class="docutils literal">cl_khr_fp16</tt> when compiled with Clang/LLVM 16+.
The implementation relies on Clang, and may result in emulation (promoting to
fp32) if the CPU does not support the required instruction set. In
Clang/LLVM 16+, the following targets have native fp16 support: 32-bit and
64-bit ARM (depending on vendor), x86-64 with AVX512-FP16.
Currently only implemented for a part of builtin library functions,
those that are implemented with either an expression, or a Clang builtin.</p>
</div>
</div>
<div class="section" id="level-zero-driver">
<h2>Level Zero driver</h2>
<p>This is a new experimental driver that supports devices accessible via Level Zero API.</p>
<p>The driver has been tested with multiple devices (iGPU and dGPU),
and passes a large portion of PoCL tests (87% tests passed, 32 tests
fail out of 254), however it has not been finished nor optimized yet,
therefore it cannot be considered production quality.</p>
<p>The driver supports the following OpenCL extensions, in addition to atomics:
cl_khr_il_program, cl_khr_3d_image_writes,
cl_khr_fp16, cl_khr_fp64, cl_khr_subgroups, cl_intel_unified_shared_memory.
In addition, Specialization Constants and SVM are supported.</p>
<p>We also intend to use the driver for prototyping features not found in
the official Intel Compute Runtime OpenCL drivers, and for experimenting
with asynchronous execution with other OpenCL devices in the same PoCL platform.
One such feature currently implemented is the JIT kernel compilation, which is
useful with programs that have thousands of kernels but only launch a few of
them (e.g. when using SPIR-V IL produced from heavily templated C++ code).
For details, see the full driver documentation in <cite>doc/sphinx/source/level0.rst</cite>.</p>
<div class="section" id="support-for-cl-intel-unified-shared-memory">
<h3>Support for cl_intel_unified_shared_memory</h3>
<p>This extension, together with SPIR-V support and other new features, allows
using PoCL as an OpenCL backend for SYCL runtimes. This works with the both
CPU driver (tested on x86-64 &amp; ARM64) and the Level Zero driver. Vincent A. Arcila
has contributed a guide for building PoCL as SYCL runtime backend on ARM.</p>
<p>Additionally, there is a new testsuite integrated into PoCL for testing USM support,
<tt class="docutils literal"><span class="pre">intel-compute-samples</span></tt>. These are tests from <a class="reference external" href="https://github.com/intel/compute-samples">https://github.com/intel/compute-samples</a>
and PoCL currently passes 78% of the tests (12 tests failed out of 54).</p>
</div>
</div>
<div class="section" id="new-testsuites">
<h2>New testsuites</h2>
<p>There are also multiple new CTest testsuites in PoCL. For testing PoCL as a SYCL backend,
there are three new testsuites: <tt class="docutils literal"><span class="pre">dpcpp-book-samples</span></tt>, <tt class="docutils literal"><span class="pre">oneapi-samples</span></tt> and <tt class="docutils literal"><span class="pre">simple-sycl-samples</span></tt>.</p>
<ul class="simple">
<li><tt class="docutils literal"><span class="pre">dpcpp-book-samples</span></tt>: these are samples from <a class="reference external" href="https://github.com/Apress/data-parallel-CPP">https://github.com/Apress/data-parallel-CPP</a>
PoCL currently passes 90 out of 95 tests.</li>
<li><tt class="docutils literal"><span class="pre">oneapi-samples</span></tt>: these are samples from <a class="reference external" href="https://github.com/oneapi-src/oneAPI-samples">https://github.com/oneapi-src/oneAPI-samples</a>
However only a few have been enabled in PoCL for now, because each sample is a separate CMake project</li>
<li><tt class="docutils literal"><span class="pre">simple-sycl-samples</span></tt>: these are from <a class="reference external" href="https://github.com/bashbaug/simple-sycl-samples">https://github.com/bashbaug/simple-sycl-samples</a>
currently contains only 8 samples, PoCL passes all of them.</li>
</ul>
<p>For testing PoCL as chipStar's OpenCL backend: <tt class="docutils literal">chipStar</tt> testsuite. This builds
the runtime and the tests from <a class="reference external" href="https://github.com/CHIP-SPV/chipStar">https://github.com/CHIP-SPV/chipStar</a>, and
runs a subset of tests (approximately 800) with PoCL as the chipStar's backend.</p>
</div>
<div class="section" id="mac-os-x-support">
<h2>Mac OS X support</h2>
<p>Thanks to efforts of Isuru Fernando who stepped up to become the official Mac OSX port maintainer, PoCL's CPU driver has been again fixed to work on Mac OS X.
The current 4.0 release has been tested on these configurations:</p>
<p>MacOS 10.13 (Intel Sandybridge), MacOS 11.7 Intel (Ivybridge) with Clang 15.</p>
<p>Additionally, there are now Github Actions for CI testing of PoCL with Mac OS X,
testing 4 different configurations: LLVM 15 and 16, with and without ICD loader.</p>
</div>
<div class="section" id="github-actions">
<h2>Github Actions</h2>
<p>The original CI used by PoCL authors (Python Buildbot, <a class="reference external" href="https://buildbot.net">https://buildbot.net</a>)
has been converted to publicly accessible Github Actions CI. These are currently
set up to test PoCL with last two LLVM versions rigorously, and basic tests with
older LLVM versions. The most tested driver is the CPU driver, with multiple
configurations enabling or testing different features: sanitizers, external
testsuites, SYCL support, OpenCL conformance, SPIR-V support. There are also
basic tests for other experimental/WiP/research-drivers in PoCL: OpenASIP, Vulkan, CUDA, and LevelZero.</p>
</div>
</div>
<div class="section" id="bugfixes-and-minor-features">
<h1>Bugfixes and minor features</h1>
<ul class="simple">
<li>CMake: it's now possible to disable libhwloc support even when it's present,
using -DENABLE_HWLOC=0 CMake option</li>
<li>AlmaIF's OpenASIP backend now supports a standalone mode.
It generates a standalone C program from a kernel launch, which
can then be compiled and executed with ttasim or RTL simulation.</li>
<li>Added a user env POCL_BITCODE_FINALIZER that can be used to
call a custom script that manipulates the final bitcode before
passing it to the code generation.</li>
<li>New alternative work-group function mode for non-SPMD from Open SYCL:
Continuation-based synchronization is somewhat more general than the default one in PoCL's
current kernel compiler, but allows for fewer hand-rolled optimizations.
CBS is expected to work for kernels that PoCL's current kernel compiler
does not support. Currently, CBS can be manually enabled by setting
the environment variable <cite>POCL_WORK_GROUP_METHOD=cbs</cite>.</li>
<li>Linux/x86-64 only: SIGFPE handler has been changed to skip instructions
causing division-by-zero, only if it occured in one of the CPU driver
threads; so division-by-zero errors are no longer hidden in user threads.</li>
<li>CUDA driver: POCL_CUDA_VERIFY_MODULE env variable has been replaced by POCL_LLVM_VERIFY</li>
<li>CUDA driver: compilation now defaults to <cite>-ffp-contract=fast</cite>, previously it was <cite>-ffp-contract=on</cite>.</li>
<li>CUDA driver: support for Direct Peer-to-Peer buffer migrations
This allows much better performance scaling in multi-GPU scenarios</li>
<li>OpenCL C: <cite>-cl-fast-relaxed-math</cite> now defaults to <cite>-ffp-contract=fast</cite>, previously it was <cite>-ffp-contract=on</cite>.</li>
<li>CPU drivers: renamed 'basic' to 'cpu-minimal' and 'pthread' driver to 'cpu',
to reflect the hardware they're driving instead of implementation details.</li>
<li>CPU drivers: POCL_MAX_PTHREAD_COUNT renamed to POCL_CPU_MAX_CU_COUNT;
the old env. variable is deprecated but still works</li>
<li>CPU drivers: Added a new POCL_CPU_LOCAL_MEM_SIZE environment for overriding the
local memory size.</li>
<li>CPU drivers: OpenCL C printf() flushes output after each call instead of waiting
for the end of the kernel command. This makes it more useful for debugging
kernel segfaults.</li>
</ul>
</div>
</div>
