<%inherit file="basic_page.makt" />
<p>Pocl is a portable open source (MIT-licensed) implementation of the OpenCL
standard (1.2 with some 2.0 features supported). In addition to producing an
easily portable open-source OpenCL implementation, another major goal of this
project is improving performance portability of OpenCL programs with the
kernel compiler and the task runtime, reducing the need for target-dependent
manual optimizations.</p>

<p>pocl uses  <a href="http://clang.llvm.org">Clang</a> as an OpenCL C frontend and
<a href="http://llvm.org">LLVM</a> for kernel compiler implementation,
and as a portability layer. Thus, if your desired target has an LLVM backend, it
should be able to get OpenCL support easily by using pocl.</p>

<p>pocl currently has backends supporting many CPUs, ASIPs (TCE/TTA),
NVIDIA GPUs (via CUDA), HSA-supported GPUs and multiple private off-tree
targets.</p>

<p>In addition to providing an open source implementation of OpenCL for
various platforms, an additional purpose of the project is to serve as a research
platform for issues in parallel programming of heterogeneous platforms.</p>

<h1>News</h1>

<h2>2019-04-04: <a href="pocl-1.3.html">Portable Computing Language (pocl) v1.3 released</a></h2>

<h2>2019-02-07: <a href="pocl-in-think-silicon.html">pocl powering Think Silicon's ultra-low power GPGPUs</a>

<h2>2018-09-25: <a href="pocl-1.2.html">Portable Computing Language (pocl) v1.2 released</a></h2>

<h2>2018-09-18: Matrix-2000 and pocl</h2>

<p>Dr. Jianbin Fang from NUDT sent us <a href="nudt-pocl-use-case.html">a nice
description</a> of how they benefitted from pocl for adding OpenCL support on
their Matrix-2000 accelerator.</p>

<h2>2018-03-09: <a href="pocl-1.1.html">Portable Computing Language
(pocl) v1.1 released</a></h2>

<h2>2017-12-19: <a href="pocl-1.0.html">Portable Computing Language
(pocl) v1.0 released</a></h2>

<h2>2017-04-25: <a href="cuda-backend.html">NVIDIA GPU support via CUDA backend</a></h2>

pocl now has experimental support for NVIDIA GPU devices via a new backend
which makes use of the LLVM NVPTX backend and the CUDA driver API.
This work was primarily carried out by James Price from the
<a href="http://uob-hpc.github.io">High Performance Computing group</a> at the
University of Bristol. Read more about it <a href="cuda-backend.html">here</a>.

<p>The source package, the change log, and the release annoucement are <a href="/downloads">here</a>.</p> 

<h2>Older news <a href="old_news.html">items here</a></h2>

<h1>Current Status</h1>

<p>Passes most of the tests in the Khronos OpenCL 1.2 conformance suite. Development towards
2.x compliance started.</p>

<h1>Feature Examples</h1>
<ul>
  <li>portable kernel compiler with horizontal autovectorization of work-groups</li>
  <li>core runtime APIs implemented in C for improved portability to bare bone machines</li>
  <li>automated kernel compiler cache</li>
  <li>driver framework that allows seamless integration of diversity of device types in
  the same OpenCL context</li>
  <li>ICD support</li>
</ul>

<br />
<hr />
<p align="center">

<a class="twitter-timeline"  href="https://twitter.com/portablecl"  data-widget-id="399622661979918336">Tweets from @portablecl</a>
    <script>!function(d,s,id){var js,fjs=d.getElementsByTagName(s)[0],p=/^http:/.test(d.location)?'http':'https';if(!d.getElementById(id)){js=d.createElement(s);js.id=id;js.src=p+"://platform.twitter.com/widgets.js";fjs.parentNode.insertBefore(js,fjs);}}(document,"script","twitter-wjs");</script>

</p>

