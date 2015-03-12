<%!
        sub_page = "Portable Computing Language (pocl) v0.11 released"
%>
<%inherit file="basic_page.makt" />

<h1>March 12th, 2015: pocl v0.11 released</h2>

<p>This release adds:
<ul>
<li>kernel compiler support for LLVM/Clang 3.6,
<li>caching of compiled OpenCL kernels
<li>initial Android support
<li>experimental Windows support (many things still broken there)
<li>two new examples, Cloverleaf and Halide, updated AMDSDK examples
<li>better debugging possibilities
<li>initial MIPS architecture support
</ul>

</p>

<p>
We consider pocl ready for wider scale testing, although the OpenCL
standard is not yet fully implemented, and it contains known bugs.
The pocl test suite compiles and runs most of the ViennaCL 1.5.1
examples, Rodinia 2.0.1 benchmarks, Parboil benchmarks, OpenCL
Programming Guide book samples, VexCL test cases, Luxmark v2.0,
most of the AMD APP SDK v2.9 OpenCL samples and piglit OpenCL tests
among others.</p>

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
