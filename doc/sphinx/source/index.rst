ntkpdf
======

ntkpdf analyses **already-trained PDF fits** made with n3fit or colibri through
the Neural Tangent Kernel (NTK). It reads saved parameter snapshots and studies
PDFs, losses, kernel eigensystems, features and loss curvature. **It never runs
a fit.**

The package connects validphys's data and report machinery to colibri's NTK
providers. These pages describe how to use that connection and how to extend
it without changing its basis conventions or multiplying large cached objects.
They target physicists and developers in the NNPDF/colibri ecosystem.

.. toctree::
   :maxdepth: 2

   getting-started
   design/index
   api/index

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
