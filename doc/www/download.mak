<%!
        sub_page = "Download"
%>
<%inherit file="basic_page.makt" />
<p>pocl uses the <a href="http://bazaar.canonical.com/">Bazaar</a> version control system and 
<a href="http://launchpad.net/pocl">Launchpad</a> for code hosting and <a href="http://bugs.launchpad.net/pocl">
bug tracking</a>.</p>

<p>The main development branch is kept stable (should pass all tests all the time), 
thus the best starting point for using and developing pocl is to check it out from 
the version control system:</p>

<pre>
        bzr co lp:pocl
</pre>

We also package releases regularly, usually after each new LLVM release:

<ul>

<li><b>Version 0.8:</b> Uses LLVM 3.3.
            <a href="http://pocl.sourceforge.net/downloads/pocl-0.8.tar.gz">pocl-0.8.tar.gz (1.3M)</a>, 
            <a href="http://pocl.sourceforge.net/downloads/CHANGES">change log</a>,
            <a href="http://pocl.sourceforge.net/pocl-0.8.html">release notes</a></li>

<li><b>Version 0.7:</b> Uses LLVM 3.2.
            <a href="http://pocl.sourceforge.net/downloads/pocl-0.7.tar.gz">pocl-0.7.tar.gz (1.0M)</a>, 
            <a href="http://pocl.sourceforge.net/downloads/CHANGES">change log</a>, 
            <a href="http://pocl.sourceforge.net/downloads/ANNOUNCEMENT">release notes</a></li>

<li><b>Version 0.6:</b> Uses LLVM 3.1.
            <a href="https://launchpad.net/pocl/0.6/0.6.0/+download/pocl-0.6.tar.gz">pocl-0.6.tar.gz (780K)</a>, 
            <a href="https://launchpadlibrarian.net/112874115/CHANGES">change log</a>, 
            <a href="https://launchpadlibrarian.net/112874413/notes-0.6.txt">release notes</a></li>
</ul>

All downloadable files, including previous releases and release
candidates, are available on <a
href="http://sourceforge.net/projects/pocl/files/">SourceForge</a>.
