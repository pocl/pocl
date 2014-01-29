<%!
        sub_page = "pocl added to Fedora"
%>
<%inherit file="basic_page.makt" />

<h2>11th September, 2013</h2>

<p>Thank's to Fabian Deutsch, pocl has now RPM packages that are
included in the Fedora distribution.</p>

<p>The users of Fedora 19 or later can now instal pocl easily with yum:</p>

<pre>
yum install pocl
</pre>

This installs the pocl to be used via an ICD loader.

The following installs a version that enables programs to link directly 
against the pocl implementation:

<pre>
yum install pocl-devel
</pre>




