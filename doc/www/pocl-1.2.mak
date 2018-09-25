<%!
        sub_page = "Portable Computing Language (pocl) v1.2 released"
%>
<%inherit file="basic_page.makt" />

<h1>September 25th 2018: pocl v1.2 released</h2>

<h2>Release Highlights</h2>

<ul>
<li>Support for LLVM/Clang 7.0 and 6.0.</li>
<li>HWLOC 2.0 support</li>
</ul>

<h2>Notes</h2>
<ul>
<li>Build on 64bit ARM with LLVM 7.0 is broken (it works with previous LLVM versions). This is a regression in LLVM.</li>
<li>Vecmathlib support is deprecated and will likely be removed in the next release, unless some users come forward.</li>
</ul>
</p>

<p>Please note that there's an official pocl maintenance policy in place.
<a href="http://portablecl.org/docs/html/maintainer-policy.html">This text</a>
describes the policy and how you can get your favourite project
that uses OpenCL to remain regression free in the future pocl releases.</p>

<p>We are looking for active maintainers to look after the ARM(64) and MacOS
ports. If you are interested in helping to keep pocl working well on these
platforms, please let us know!</p>

<h2>Acknowledgements</h2>

<p>Most of the code that landed to the pocl code base during this release
cycle was produced for the needs of research funded by various sources.
The Customized Parallel Computing research group of Tampere University of
Technology (Finland) likes to thank the Academy of Finland (funding
decision 297548), Business Finland (FiDiPro project StreamPro,
1846/31/2014), ECSEL JU project FitOptiVis (project number 783162) and
HSA Foundation for funding most of the development work in this release.
Much appreciated!
</p>

<p><a href="download.html">Download</a>.</p>
