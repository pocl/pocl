<%inherit file="basic_page.makt" />
<p>Portable OpenCL (pocl) is a MIT-licensed open source implementation of the OpenCL 
standard which can be easily adapted for new targets and devices, both for homogeneous
CPU and heterogenous GPUs/accelerators.</p>

<p>pocl uses  <a href="http://clang.llvm.org">Clang</a> as an OpenCL C frontend and 
<a href="http://llvm.org">LLVM</a> for the kernel compiler implementation, 
and as a portability layer. Thus, if your desired target has an LLVM backend, it
should be able to get OpenCL support easily by using pocl.</p>

<p>The goal is to accomplish improved performance portability using a  kernel 
compiler that can generate multi-work-item work-group functions that exploit 
various types of parallel hardware resources: VLIW, superscalar, SIMD, SIMT,
multicore, multithread ...</p>

<p>In addition to providing a portable open source implementation of OpenCL, another 
goal of the project is to serve as a research platform for issues in parallel
programming on heterogeneous platforms.</p>

<h2>Current status (2012-12-07)</h2>

<p>A lot of OpenCL programs, projects and test suites work out of the box, but there are still
unimplemented OpenCL 1.2 APIs. These will be added gradually as needed by new tested applications.</p>

<p>Patch contributions welcomed, of course!</p>

<p>The following OpenCL applications are tested regularly in the pocl test suite:</p>

<ul>
  <li><a href="http://viennacl.sourceforge.net/">ViennaCL</a> 1.3.1 examples</li>
  <li><a href="http://lava.cs.virginia.edu/Rodinia/download_links.htm">Rodinia</a> 2.0.1</li>
  <li><a href="http://impact.crhc.illinois.edu/parboil.aspx">Parboil</a> Benchmarks (most of them)</a>
  <li><a href="https://code.google.com/p/opencl-book-samples/">OpenCL Programming Guide</a> book samples (most of them)</a>
</ul>

<p>pocl has been tested on the following platforms <a href="http://llvm.org/docs/GettingStarted.html#hardware">supported by LLVM</a>:</p>

<ul>
  <li>X86_64/Linux (host&amp;device)</li>
  <li>PowerPC64/Linux (host&amp;device)</li>
  <li>PowerPC32/Linux (host&amp;device)</li>
  <li>ARM v7/Linux (host&amp;device)</li>
  <li>Multiple VLIW-style TTA processors designed using <a href="http://tce.cs.tut.fi">TCE</a> (host X86_64&amp;instruction set simulator+FPGA-over-PCIe prototype as a device</li>
</ul>

  
<h2>Feature highlights (2012-12-07)</h2>
<ul>
  <li>icd support</li>
  <li>configurable list of devices</li>
  <li>two ways to generate work-group code: work-item loops and full work-item replication</li>
</ul>
