<%inherit file="basic_page.makt" />
<p>Portable Computing Language (pocl) aims to become a MIT-licensed open source implementation of the OpenCL 
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

<p>Additional purpose of the project is to serve as a research platform for issues 
in parallel programming on heterogeneous platforms.</p>

<h1>News</h1>

<h2>2015-12-04:
  <a href="publications.html">Added a master's thesis from Delft and a publication from Bristol</a>
</h2>
<p>The master's thesis from TUDelft uses pocl for OpenCL support on a research platform.
The Bristol's publication uses pocl in context of an OpenCL to FPGA flow.
</p>
<p>If any of you have more publications or thesis about using or enhancing pocl somehow, please let 
<a href="discussion.html">us know</a>.</p>

<h2>2015-10-26: <a href="pocl-0.12.html">Portable Computing Language
(pocl) v0.12 released</a></h2>

<h2>2015-03-12: <a href="pocl-0.11.html">Portable Computing Language
(pocl) v0.11 released</a></h2>

<h2>2014-09-16: Added a '<a href="publications.html">publications</a>' section with a couple of papers in it</h2>

<h2>2014-09-04: <a href="pocl-0.10.html">Portable Computing Language
(pocl) v0.10 released</a></h2>

<h2>2014-07-24: pocl's <a href="http://krblogs.com/post/92744765821/opencl-for-android-masses">Android port</a> 
now at beta.</a></h2>

<h2>2014-01-29: <a href="pocl-0.9.html">Portable Computing Language
(pocl) v0.9 released</a></h2>

<h2>2013-09-12: <a href="rpm.html">pocl included in Fedora</a></h2>

<h2>2013-08-12: <a href="pocl-0.8.html">Portable Computing Language
(pocl) v0.8 released</a></h2>

<h2>2013-02-04: Project renamed to POrtable Computing Language (pocl) </h2>

<p>The project was renamed to POrtable Computing Language (pocl) to avoid trademark
uncertainties (was: Portable OpenCL). It's still pronounced "pogle"! :)</p>

<h2>2013-01-09: 0.7 released!</h2>

<p>The source package, the change log, and the release annoucement are <a href="/downloads">here</a>.</p> 

<h1>Current status</h1>

<p>A lot of OpenCL programs, projects and test suites work out of the box, but there are still
unimplemented OpenCL APIs. These will be added gradually as needed by new tested applications.</p>

<p>Patch contributions welcomed, of course!</p>

<p>The following OpenCL applications are known to work with pocl:</p>

<ul>
  <li><a href="http://viennacl.sourceforge.net/">ViennaCL</a> 1.5.1 examples</li>
  <li><a href="http://lava.cs.virginia.edu/Rodinia/download_links.htm">Rodinia</a> 2.0.1</li>
  <li><a href="http://impact.crhc.illinois.edu/parboil.aspx">Parboil</a> Benchmarks (most of them)</a>
  <li><a href="https://code.google.com/p/opencl-book-samples/">OpenCL Programming Guide</a> book samples (most of them)</a>
  <li><a href="http://developer.amd.com/tools/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/">AMD APP SDK v2.9</a>
  OpenCL samples (most of them)</a>
  <li><a href="http://www.luxrender.net/wiki/LuxMark">Luxmark v2.0</a>
  <li><a href="http://piglit.freedesktop.org/">piglit (97%+ of the tests pass)</a>
</ul>

<p>pocl has been tested on the following platforms <a href="http://llvm.org/docs/GettingStarted.html#hardware">supported by LLVM</a>:</p>

<ul>
  <li>X86_64/Linux (host&amp;device)</li>
  <li>MIPS32/Linux (host&amp;device)</li>
  <li>ARM v7/Linux (host&amp;device)</li>
  <li>AMD HSA APUs/Linux (host&amp;device)</li>
  <li>Multiple VLIW-style TTA processors designed using <a href="http://tce.cs.tut.fi">TCE</a> in heterogeneous host-device setups.</li>
</ul>
  
<h1>Feature highlights</h1>
<ul>
  <li>portable kernel compiler with horizontal autovectorization of work-groups (experimental)</li>
  <li>efficient math built-in libraries</li>
  <li>core APIs implemented in C for improved portability to bare bone machines</li>
  <li>ICD support</li>
  <li>HSA device support (experimental)</li>
  <li>experimental Android support</li>
</ul>

<br />
<hr />
<p align="center">

<a class="twitter-timeline"  href="https://twitter.com/portablecl"  data-widget-id="399622661979918336">Tweets from @portablecl</a>
    <script>!function(d,s,id){var js,fjs=d.getElementsByTagName(s)[0],p=/^http:/.test(d.location)?'http':'https';if(!d.getElementById(id)){js=d.createElement(s);js.id=id;js.src=p+"://platform.twitter.com/widgets.js";fjs.parentNode.insertBefore(js,fjs);}}(document,"script","twitter-wjs");</script>

</p>

