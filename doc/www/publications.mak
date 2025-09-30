## -*- coding: utf-8 -*-
<%!
        sub_page = "Publications"
%>
<%inherit file="basic_page.makt" />
<%namespace name="comp" file="components.mak"/>

<p>Academic publications such as research papers or master's thesis about PoCL or using PoCL
for something are listed here.</p>

<p>If you use or somehow benefit from PoCL in your research, please cite the "pocl-paper"
below (<a href="pocl-paper.bib">bibtex</a>) and let us know. We are happy to add
your paper to this page to get it more visibility.</p>

<h1>About PoCL the open source project itself</h1>

${comp.load_publications("pocl-publications.txt")}

<h1>Publications using PoCL somehow</h1>

${comp.load_publications("pocl-using-publications.txt")}

