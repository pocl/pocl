<%!
        sub_page = "Portable Computing Language (pocl) v0.14 released"
%>
<%inherit file="basic_page.makt" />

<h1>April 2017: pocl v0.14 released</h2>

<h2>Release highlights</h2>

<ul>
<li>Support for LLVM/Clang 4.0 and 3.9.</li>
<li<A new binary format containing the final executable bits, which enables
  running OpenCL programs on hosts without online compiler support.</li>
<li>Initial support for out-of-order command queue task scheduling.</li>
<li>A plenty of bugfixes also to some very long standing bugs.</li>
<li>Other optimizations and bug fixes</li>
</ul>
</p>

<p>Please note that there's now an official pocl maintenance policy in place.
<a href="http://portablecl.org/docs/html/maintainer-policy.html">This text</a>
describes the policy and how you can get your favourite project
that uses OpenCL to remain regression free in the future pocl releases.</p>

<p>We are looking for active maintainers to look after the ARM(64) and MacOS
ports. If you are interested in helping to keep pocl working well on these
platforms, please let us know!</p>

<h2>Acknowledgements</h2>

<p>Most of the code that landed to the pocl code base during this release
cycle was produced while conducting research funded by various sources.
The Customized Parallel Computing research group of Tampere
University of Technology (Finland) likes to thank the Finnish Funding
Agency for Technology and Innovation (project "Parallel Acceleration 3",
funding decision 1134/31/2015), ARTEMIS JU (grant agreement #621439, ALMARVI)
and Academy of Finland (decision #297548, PLC).</p>

<p><a href="download.html">Download</a>.</p>
