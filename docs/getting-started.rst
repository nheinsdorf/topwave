Installation
=============

The :doc:`getting-started` guide is intended to assist the user with installing the library.

Install using ``pip``
---------------------
The :obj:`topwave` library requires Python 3.9 and above. It can **soon** be installed from
`PyPI <https://pypi.org/project/topwave/>`_ using ``pip``.

.. code-block:: console

   $ python3 -m pip install topwave

Install from Source
-------------------

To install from Source, you can ``git clone`` the repository with

.. code-block:: console

   $ git clone https://github.com/nheinsdorf/topwave
   $ cd topwave
   $ python3 -m pip install -r requirements.txt
   $ python3 setup.py


Import the :obj:`topwave` package and verify that it was installed correctly

.. ipython:: python

    import topwave as tp
    tp.__version__

