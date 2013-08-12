<%!
        sub_page = "Portable Computing Language (pocl) v0.8 released"
%>
<%inherit file="basic_page.makt" />

<p>The Internet, August 2013.</p>

<p>Pocl's goal is to become an efficient open source (MIT-licensed)
implementation of the OpenCL 1.2 (and soon OpenCL 2.0) standard.</p>

<p>In addition to producing an easily portable open-source OpenCL
implementation, another major goal of this project is improving
performance portability of OpenCL programs with compiler
optimizations, reducing the need for target-dependent manual
optimizations.</p>

<p>At the core of pocl is the kernel compiler that consists of a set
of LLVM passes used to statically transform kernels into work-group
functions with multiple work-items, even in the presence of work-group
barriers. These functions are suitable for parallelization in multiple
ways (SIMD, VLIW, superscalar,...).</p>

<p>This release adds support for LLVM/Clang 3.3, employs inner loop
parallelization in the kernel compiler, uses Vecmathlib for inlineable
efficient math library implementations, contains plenty of bug fixes,
and provides several new OpenCL API implementations.</p>

<p>We consider pocl ready for wider scale testing, although the OpenCL
1.2 standard is not yet fully implemented, and it contains known bugs.
The pocl 0.8 test suite compiles and runs most of the ViennaCL 1.3.1
examples, Rodinia 2.0.1 benchmarks, Parboil benchmarks, OpenCL
Programming Guide book samples, VexCL test cases, Luxmark v2.0, and
most of the AMD APP SDK v2.8 OpenCL samples, among others.</p>

<p>Note: This project was originally called Portable OpenCL.</p>

<h2>Acknowledgements</h2>

<p>We'd like to thank the Radio Implementation Research Team from
Nokia Research Center and Academy of Finland (funding decision 253087)
for funding the development of this release.</p>

<p>E. Schnetter acknowledges support from the Perimeter Institute, as
well as funding from NSERC (Canada) via a Discovery Grant and from NSF
(USA) via OCI Award 0905046.</p>

<h2>Links</h2>

<ul>

<li><a href="http://pocl.sourceforge.net/">Home page</a></li>

<li><a href="http://pocl.sourceforge.net/pocl-0.8.html">This
announcement</a></li>

<li><a href="https://sourceforge.net/projects/pocl/files/CHANGES">Change log</a></li>

<li><a href="https://sourceforge.net/projects/pocl/">Download</a></li>

</ul>
