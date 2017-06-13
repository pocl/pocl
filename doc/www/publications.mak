## -*- coding: utf-8 -*-
<%!
        sub_page = "Download"
%>
<%inherit file="basic_page.makt" />
<%def name="load_publications(filename)">
<%
        lines = open(filename, 'r').readlines()
        i = iter(lines)
%>
<ul>
<%
        papers = []
        try:
                while True:
                      authors = next(i).strip()
                      title = next(i).strip()
                      url = next(i).strip()
                      place = next(i).strip()
                      if place.endswith("."): place = place[0:-1]
                      empty = next(i)
                      papers.append((authors.strip(), title.strip(), url, place))
        except:
                pass
%>
%for (authors, title, link, place) in papers:
<li>${authors}:<br />
<span class='paperTitle'>"<a href="${link}">${title}</a>"</span><br />
${place}.
</li>
%endfor

</ul>
</%def>

<p>Publications (research papers, thesis) about or using pocl are listed here.
If you use or somehow benefit from pocl in your research, please cite the "pocl-paper"
below (<a href="pocl-paper.bib">bibtex</a>) and let us know. We are happy to add 
your paper to this page.</p>

<h1>About pocl (the open source project) itself</h1>

${load_publications("pocl-publications.txt")}

<h1>Publications using pocl somehow</h1>

${load_publications("pocl-using-publications.txt")}


