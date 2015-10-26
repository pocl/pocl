<%!
        sub_page = "Portable Computing Language (pocl) v0.12 released"
%>
<%inherit file="basic_page.makt" />

<h1>October 26th, 2015: pocl v0.12 released</h2>

<p>This release adds:
<ul>
<li> support for HSA-compliant devices (tested with AMD Kaveri's GPU)
    ( for installation steps & current status, please see the HSA section
     in <a href="http://www.portablecl.org/docs/html/hsa.html">Pocl documentation</a>)
<li> Little endian MIPS32 now passes almost all tests
<li> kernel compiler support for LLVM/Clang 3.7
<li> improved caching of compiled OpenCL kernels
</ul>
</p>

<p>
We consider pocl ready for wider scale testing, although the OpenCL
standard is not yet fully implemented, and it contains known bugs.
The pocl test suite compiles and runs most of the ViennaCL 1.5.1
examples, Rodinia 2.0.1 benchmarks, Parboil benchmarks, OpenCL
Programming Guide book samples, VexCL test cases, Luxmark v2.0,
most of the AMD APP SDK v2.9 OpenCL samples and piglit OpenCL tests
among others.
</p>

<p>
Note: 0.12 is the last version that builds with older LLVM versions.
We plan to prune code needed to support for the old versions in the
next release and focus on the current stable LLVM version and possibly
the previous one for easier transition.
</p>


<h2>Acknowledgements</h2>

<p>We'd like to thank Finnish Funding Agency for Technology and Innovation
(project "Parallel Acceleration 2", funding decision 40115/13), Academy of
Finland (funding decision 253087) and ARTEMIS joint undertaking under grant
agreement no 641439 (ALMARVI) for funding the development of
this release.</p>

<p>A pocl developer E. Schnetter acknowledges support from the Perimeter
Institute, as well as funding from NSERC (Canada) via a Discovery Grant and
from NSF (USA) via OCI Award 0905046.</p>

<p><a href="http://portablecl.org/download.html">Download</a>.</p>
