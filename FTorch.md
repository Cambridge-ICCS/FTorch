---
project: FTorch
summary: FTorch - A library for coupling (Py)Torch machine learning models to Fortran codes.<br>Written in modern Fortran (2008) with source code available on [GitHub](https://github.com/Cambridge-ICCS/FTorch) it has been used in [multiple scientific projects](https://cambridge-iccs.github.io/FTorch/page/community/case_studies.html).<br>The associated JOSS paper can read [here](https://joss.theoj.org/papers/10.21105/joss.07602).
license: mit
project_github: https://github.com/Cambridge-ICCS/FTorch
author: ICCS Cambridge
author_pic: https://raw.githubusercontent.com/Cambridge-ICCS/FTorch/refs/heads/main/pages/media/ICCS_logo.png
author_description: Research Software Engineering team supporting software development and standards across the climate sciences domain.
github: https://github.com/Cambridge-ICCS
website: https://iccs.cam.ac.uk
page_dir: pages
media_dir: ./pages/media/
css: ./pages/media/theme/ftorch.css
favicon: ./pages/media/FTorch_Favicon.jpg
src_dir: ./src
         ./utils
output_dir: ./doc
exclude_dir: **/build*
extra_filetypes: c   //
                 cpp //
                 h //
                 py  #
preprocess: true
dbg: false
macro: GPU_DEVICE_NONE=0
       GPU_DEVICE_CUDA=1
       GPU_DEVICE_XPU=11
       GPU_DEVICE_MPS=12
       GPU_DEVICE_HIP=1
sort: alpha
source: true
graph: true
externalize: true
md_extensions: markdown.extensions.toc
               markdown.extensions.tables
               markdown.extensions.fenced_code
---

--------------------

![logo](|media|/FTorch_logo.svg "FTorch logo"){width=100%}


Description
-----------

It is desirable to run machine learning (ML) models directly in Fortran.
However, ML models are often trained in some other language using popular frameworks (say, Python and PyTorch).

FTorch is a library enabling users to directly couple their PyTorch models to Fortran code enabling the development of hybrid models that combine high-performance scientific computing with modern ML techniques.
It supports running on both CPU and GPU, and is tested on UNIX and Windows operating systems

This site contains user and API documentation for FTorch as well as community information.
There are also installation instructions for the library and numerous examples of
performing coupling.


Using and Citing FTorch
-----------------------

FTorch is published in JOSS. To cite it in your work please refer to:

<div style="margin-left: 2em;">
Atkinson et al., (2025). FTorch: a library for coupling PyTorch models to Fortran.<br>
<em>Journal of Open Source Software</em>, 10(107), 7602, <a href="https://doi.org/10.21105/joss.07602">https://doi.org/10.21105/joss.07602</a>
</div><br>

For examples of the many projects and publications that rely on FTorch please see our [user case studies page](|page|/community/case_studies.html).
If you use FTorch and want to be included please get in touch.

The [presentations](|page|/community/presentations.html) page has a full list of recent talks about FTorch with links to slides and recordings.


News
----

For the latest updates and developments please see the
[news page](|page|/community/news_archive.html) and
[join our user mailing list](https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=ftorch-announce).


Training
--------

We offer training on FTorch in the form of tutorials and workshops. The
companion repository
[https://github.com/Cambridge-ICCS/FTorch-workshop](https://github.com/Cambridge-ICCS/FTorch-workshop)
provides a set of exercises and solutions to help users get started with FTorch.
Upcoming in-person sessions will be listed on the [news page](|page|/community/news_archive.html).


Contact
--------

For feature requests, bug reports, or assistance in using FTorch please get in touch via
GitHub by either [raising an issue](https://github.com/Cambridge-ICCS/FTorch/issues)
or [opening an discussion](https://github.com/Cambridge-ICCS/FTorch/discussions).

For general enquiries please contact ICCS via [iccs@maths.cam.ac.uk](mailto:iccs@maths.cam.ac.uk).


License
-------

The FTorch source code, related files and documentation are
distributed under an [MIT License which can be viewed here](|page|/LICENSE.html).
