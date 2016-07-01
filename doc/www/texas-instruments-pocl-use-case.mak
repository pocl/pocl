<%!
        sub_page = "Kernel De-SPMD Compilation for Texas Instruments' DSPs"
%>

<%inherit file="basic_page.makt" />

<p>[2016-07-01] pocl developers were recently happy to find out that also Texas Instruments
is using a core component, the "de-spmd" kernel compiler of pocl, in their OpenCL SDK for
a range of DSPs.</p>

<p>After querying for details, we received the following response (published
here verbatim with permission):<p>

<span style="text-align:center">
<div style="margin:20px">
<p><i>
Thanks for the good work in POCL.  We at Texas Instruments found the LLVM
passes from POCL very useful.  The modular design of POCL enables us to
incorporate passes into TI's OpenCL C kernel compilation flow. We use POCL
LLVM passes to transform OpenCL C kernel function written as a work-item
function into an aggregated work-group function which iterates through all
work-items.  We support OpenCL on various TI's SoCs (ARM as host + DSP as device)
as well as desktop with attached PCI-E cards (x86 as host + DSP on PCI-E as device).
List of supported platforms can be found <a href="http://downloads.ti.com/mctools/esd/docs/opencl/intro.html">here</a>.</p>

<p>We have done some modifications to the POCL LLVM passes.  For example, since we
do not perform re-compilation at every kernel invocation, we parameterize the loop
bounds when generating loops around work-items.  We offer both online and offline kernel compilation.
We annotate POCL privatized data with meta data so that our down-stream compiler tool chain can do
a better alias analysis.  We do not privatize allocas that only exist in a single POCL analyzed
ParallelRegion.  All our modifications to POCL as well as our OpenCL runtime source code are
available on the public <a href="http://git.ti.com/gitweb/?p=opencl/ti-opencl.git">git repository</a>
(you already know this address:), you are welcome to merge any of our changes back to POCL
if you find them useful.  I think POCL licenses and TI licenses we use for OpenCL source code are compatible.</p>

<p>Thanks again for the excellent work, from the Texas Instruments Multicore Tools Team (supporting
TI's multicore platforms with compiler and runtime including OpenCL, OpenMP and OpenMP Accelerated Model)!</p>

<p>- Yuan</p>
</i>
</p>

</div>
</span>

<p>It seems their pocl kernel compiler version is not the most recent one as upstream
pocl was quite recently added the generation of work-group functions with dynamic
local size and proper offline compilation.  However, we will definitely take a
closer look at their branch if the other mentioned improvements could be upstreamed!</p>
<br/>
<p>As always, if you have found any other interesting use cases for pocl,
<a href="http://portablecl.org/discussion.html">please let us know</a>, we'll be
happy to tell about them here.</p>

<p><i>- Pekka</i></p>


