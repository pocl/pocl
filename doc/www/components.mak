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
