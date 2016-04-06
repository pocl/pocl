<%!
        sub_page = "Portable Computing Language (pocl) v0.13 released"
%>
<%inherit file="basic_page.makt" />

<h1>April 2016: pocl v0.13 released</h2>

<h2>Release highlights</h2>

<ul>
<li> kernel compiler support for LLVM/Clang 3.8</li>
<li> initial (partial) OpenCL 2.0 support</li>
<li> CMake build system almost on parity with autotools</li>
<li> Improved HSA support</li>
<li> Other optimizations and bug fixes</li>
</ul>
</p>

<p>We consider pocl ready for wider scale testing, although the OpenCL
standard is not yet fully implemented, and it contains known bugs.
The pocl test suite compiles and runs most of the ViennaCL 1.5.1
examples, Rodinia 2.0.1 benchmarks, Parboil benchmarks, OpenCL
Programming Guide book samples, VexCL test cases, Luxmark v2.0,
most of the AMD APP SDK v2.9 OpenCL samples and piglit OpenCL tests
among others.</p>

<h2>Acknowledgements</h2>

<p>We'd like to thank thank Finnish Funding Agency for Technology
and Innovation (project "Parallel Acceleration 3", funding decision
1134/31/2015) and ARTEMIS JU under grant agreement no 621439
(ALMARVI). Special thanks to HSA Foundation who sponsors the work on
implementing the HSA support.</p>

<p><a href="http://portablecl.org/download.html">Download</a>.</p>
