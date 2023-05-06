.. topwave documentation master file, created by
   sphinx-quickstart on Wed May  3 15:49:42 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: ./_static/topwave-heading.png
   :align: center

The `topwave` library is a python package to quickly set up single-particle models and calculate their topological invariants.

Topwave interfaces with the `Python Materials Genomic library <https://pymatgen.org/>`_. You can use either crystallographic
data formats like .cif directly or build your own crystal structures using pymatgen's functionality. The crystal can then
be populated with bosonic or fermionic degrees of freedom. The interactions can be generated to automatically
respect space group symmetries. It's also super easy to construct supercell and even twisted structures.

Quasi-particle spectra and quantities derived thereof can easily be calculated. Electronic band structures,
magnon dispersion, magnetization, surface states and some response functions are just a few examples.

Using Wilson loops the Berry curvature and band topological invariants can be efficiently evaluated.

.. admonition:: Disclaimer
   :class: warning

   Topwave is still under development. This is only an alpha version. Please report bugs.



Topological classification of single-particle band structures have become a standard procedure in many branches of
condensed matter physics. Many libraries to compute topological invariants, and even more to produce band structures exist.
In my research as a PhD student I was playing around with a lot of effective models for many different types of materials.
Topwave uses mostly well-known standard procedures. My attempt to streamline the process of creating and classifying
effective lattice models is what provided the code base of this package.

.. admonition:: Do you like topwave?
   :class: seealso

   Give us a star on GitHub! Tell you friends and colleagues!

Features
--------

- Fast-assembly of crystal structures using crystallographic data or pymatgen
- Use of space group symmetries to quickly create models and interactions
- Magnon Dispersion and Neutron Scattering using linear spin wave theory
- Electronic band structures of tight binding models
- Computation of topological invariants using Wilson loops
- Supercell calculations for magnetic relaxation and topological surface states
- Easy construction of twisted structures

Roadmap
-------

- Multi Orbital Tight Binding Models
- Higher Order Topological Invariants
- BdG Hamiltonians
- Consistent Mean Field Calculations
- Wannier90 Interface
- Model Construction based on Irreducible Representations.

Acknowledgements
----------------

This library is being developed within my research as a joint PhD student of the Max Planck Institute for Solid State
Research in Stuttgart, Germany and the University of British Columbia, Canada. I would like to thank Xianxin Wu, Jean-Claude Pussy and
my supervisor Andreas Schnyder.

I used the `PythTB <https://www.physics.rutgers.edu/pythtb/>`_ and `SpinW <https://spinw.org/>`_ libraries for testing and
to get an idea of what kind of user interface I want.

Topwave interfaces with `Python Materials Genomic library <https://pymatgen.org/>`_ which provides an extensive toolbox
for everything related to crystallography.

Citation
--------

If this library was useful to you in your research, please cite us::

   @software{Heinsdorf_Topwave_2023,
      title = {{topwave: Toolbox for Topology of Single-Particle Spectra}},
      author = {Heinsdorf, Niclas},
      year = {2023},
      url = {https://github.com/nheinsdorf/topwave},
   }


.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :caption: Tutorials
   :hidden:
   :maxdepth: 2

   tutorials/intro-to-coupling.rst

.. toctree::
   :caption: Examples
   :hidden:
   :maxdepth: 2

   examples/breathing_pyrochlore.rst

.. toctree::
   :caption: Reference
   :hidden:
   :maxdepth: 2

   api



