<%!
        sub_page = "Portable Computing Language (pocl) v1.4 released"
%>
<%inherit file="basic_page.makt" />

<h1>October 2019: pocl v1.4 released</h2>

<h2>Release Highlights</h2>

<ul>
<li>Support for LLVM/Clang 8.0 and 9.0</li>
<li>Support for LLVM older than 6.0 has been removed.</li>
<li>Improved SPIR and SPIR-V support for CPU device</li>
<li>pocl-accel: An example driver and support infrastructure for OpenCL 1.2
   CL_DEVICE_TYPE_CUSTOM hardware accelerators which implement a memory
   mapped control interface.</li>
<li>It's now possible to build a relocatable pocl install.</li>
</ul>

<p>A more detailed changelog <a href="http://portablecl.org/downloads/CHANGES">here</a>.

<p>Please note that there's an official pocl maintenance policy in place.
<a href="http://portablecl.org/docs/html/maintainer-policy.html">This text</a>
describes the policy and how you can get your favourite project
that uses OpenCL to remain regression free in the future pocl releases.</p>

<h2>Acknowledgements</h2>

<p>
Most of the code that landed to the pocl code base during this release
cycle was produced for the needs of research projects funded by various
sources. <a href="http://tuni.fi/cpc">Customized Parallel Computing</a> research group of
<a href="http://www.tuni.fi">Tampere University</a>, Finland likes to thank the Academy of Finland (funding
decision 297548), ECSEL JU project FitOptiVis (project number 783162) and
HSA Foundation for funding most of the development work in this release.
Much appreciated!
</p>

<p><a href="download.html">Download</a>.</p>
