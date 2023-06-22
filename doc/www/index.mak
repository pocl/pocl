<%inherit file="basic_page.makt" />
<p>PoCL is a portable open source (MIT-licensed) implementation of the
OpenCL standard (1.2 with some 2.0 features supported). In addition to
being an easily portable multi-device (truely heterogeneous)
open-source OpenCL implementation, a major goal of this project is
improving interoperability of diversity of OpenCL-capable devices by
integrating them to a single centrally orchestrated platform. Also
one of the key goals longer term is to enhance performance portability
of OpenCL programs across device types utilizing runtime and compiler
techniques.</p>

<p>Upstream PoCL currently supports various CPUs, NVIDIA GPUs via libcuda,
HSA-supported GPUs and TCE ASIPs (experimental, see: <a href="http://openasip.org">OpenASIP</a>).
It is also known to have multiple (private) adaptations in active production
use.</p>

<p>PoCL uses  <a href="http://clang.llvm.org">Clang</a> as an OpenCL C frontend and
<a href="http://llvm.org">LLVM</a> for kernel compiler implementation,
and as a portability layer. Thus, if your desired target has an LLVM backend, it
should be able to get OpenCL support easily by using PoCL.</p>

<h1>News</h1>

<h2>2023-06-22: <a href="pocl-4.0.html">Portable Computing Language (PoCL) v4.0 released</a></h2>

<h2>2022-12-05: <a href="pocl-3.1.html">Portable Computing Language (PoCL) v3.1 released</a></h2>

<h2>2022-11-15: <a href="almaif.html">Advanced hardware accelerator support through AlmaIF</a></h2>

<h2>2022-06-10: <a href="pocl-3.0.html">Portable Computing Language (PoCL) v3.0 released</a></h2>

##<h2>2021-10-12: <a href="pocl-1.8.html">Portable Computing Language (PoCL) v1.8 released</a></h2>

##<h2>2021-05-19: <a href="pocl-1.7.html">Portable Computing Language (PoCL) v1.7 released</a></h2>

##<h2>2020-12-16: <a href="pocl-1.6.html">Portable Computing Language (PoCL) v1.6 released</a></h2>

<h2>2020-08-14: <a href="http://portablecl.org/docs/html/debug.html">Debugging OpenCL applications with PoCL</a></h2>

##<h2>2020-04-03: <a href="pocl-1.5.html">Portable Computing Language (pocl) v1.5 released</a></h2>

##<h2>2019-10-14: <a href="pocl-1.4.html">Portable Computing Language (pocl) v1.4 released</a></h2>

<h2>2019-07-15: Hardware Accelerators in POCL</h2>

<p>PoCL received support for CL_DEVICE_TYPE_CUSTOM via addition of a hardware accelerator framework.
It consists of an example driver (pocl-accel) that relies on a "pocl standard" control interface and
an enumeration of "pocl-known" built-in kernels.
The example accelerator is generated using the <a href="http://openasip.org">TCE tools</a>.
</p>

<p>For more information, please read a
<a href="https://www.computer.org/publications/tech-news/accelerator-framework-for-portable-computing-language">blog post</a>
about it in the Heterogeneous System Architecture section of the IEEE Computer Society tech news or
the usage instructions in the <a href="http://portablecl.org/docs/html/accel.html">user manual</a>.
</p>

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
3.0 compliance with some 2.x features started.</p>

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

