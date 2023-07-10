Graphene
========

Graphene is the fruitfly of (band) topological systems. Let us set up a simple tight-binding model
and investigate its topological properties.

We use a `pymatgen Structure <https://pymatgen.org/pymatgen.core.structure.html#pymatgen.core.structure.Structure>`_ as
as input. We start by constructing a hexagonal unit cell. Pymatgen structures are always three-dimensional,
so we just put some vacuum along the c-direction of the unit cell. We also set the space group symmetry
to P6/mmm (#191).

.. ipython:: python

    from pymatgen.core.structure import Lattice, Structure

    space_group = 191
    lattice = Lattice.hexagonal(1.42, 10)
    structure = Structure.from_spacegroup(sg=space_group, lattice=lattice, species=['C'], coords=[[1 / 3, 2 / 3, 0]])
    print(structure)

Next, we instantiate a tight-binding model using that structure.

.. ipython:: python

    import topwave as tp
    model = tp.TightBindingModel(structure)

Graphene has hybrid orbitals sp2 on its sites with three major inplane lobes at 120 degree and the pz orbital
sticking out of plane. The inplane orbitals form strong sigma bonds, whereas the pz orbitals form pi bonds. These bonds
are responsible for graphene's low energy properties, so we only need to consider a single orbital per site. CITE?

We can confirm that a single orbital per site is `topwave`'s default setting by printing the site properties.

.. ipython:: python

    model.show_site_properties()

Great, the `orbitals` column shows one orbital for each site. Next, we generate the possible couplings up to
next-nearest neighbors and output them.

.. ipython:: python

    model.generate_couplings(max_distance=1.5, space_group=space_group)
    model.show_couplings()

There are three nearest neighbors bonds, about 0.82 Angstrom apart. We now couple these sites by some
hopping amplitude that accounts for the mobile electrons along the pi bonds. We can select the bonds from the
list in different ways. Because we input the space group symmetry, `topwave` automatically classifies the bonds
based on symmetry. Graphene has rotational symmetry and all the nearest neighbors are symmetrically equivalent,
which is why they share the same symmetry index. Let's assign the hopping using that symmetry index.

.. ipython:: python

    t = -1
    model.set_coupling(attribute_value=0, strength=t, attribute='symmetry_id')
    model.show_couplings()

We can see in the table that first three bonds now have strength -1. Next, we want to compute the
band structure of our model. As an input we just need a list of k-points. We
could create the list manually, or use the functionality of :class:`topwave.set_of_kpoints.Path`.

.. ipython:: python

    path = tp.Path([[0, 0, 0],
                    [1 / 2, 0, 0],
                    [1 / 3, 1 / 3, 0],
                    [0, 0, 0]])
    spec = tp.Spec(model, path)

    # Let's plot it
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for band in spec.energies.T:
        ax.plot(band, ls='-', lw=2.5, c='deeppink')
    ax.set_xticks(path.node_indices);
    @savefig graphene_bands.png
    ax.set_xticklabels([r'$\Gamma$', 'M', 'K', r'$\Gamma$']);

We can see graphene's famous Dirac point at the K-point. It's time reversal partner,
the K'-point, sits at opposite momentum. Let us construct the Wilson loop around the K-point using
class:`topwave.set_of_kpoints.Circle` and calculate the Berry phase for the valence band.
We also add a small staggered onsite potential to the sublattices. This so-called mass gap is trivial,
and makes the Berry curvature at the K- and K'-point well defined.

.. ipython:: python

    mass_term = 0.005
    model.set_onsite_scalar(0, mass_term)
    model.set_onsite_scalar(1, -mass_term)

    circle = tp.Circle(radius=0.02, center=[1 / 3, 1 / 3, 0], normal=[0, 0, 1])
    spec = tp.Spec(model, circle)
    print(spec.get_berry_phase(band_indices=[0]))

As expected, the Berry phase is quantized to (-)pi. Because the system is time-reversal invariant,
the Berry phase around the K'-point must have the opposite sign. There is no net flux of Berry curvature. Let us have a look
at the distribution of Berry flux over the entire two-dimensional Brillouin zone. We cover the plane with
small plaquettes and compute the flux through each plaquette.

.. ipython:: python

    # Create a cover of plaquettes of the xy-plane (with the z-direction as the norm).
    num_x, num_y = 13, 13
    plaquettes = tp.get_plaquette_cover('z', num_x, num_y)

    # Calculate the spectrum for each plaquette.
    spectra = [tp.Spec(model, plaquette) for plaquette in plaquettes]

    # Compute the Berry phase of each spectrum and plot it.
    fluxes = np.array([spectrum.get_berry_phase(band_indices=[0]) for spectrum in spectra])

    fig, ax = plt.subplots()
    im = ax.imshow(fluxes.reshape((num_x, num_y)), origin='lower', cmap='PiYG', vmin=-np.pi, vmax=np.pi, extent=[-0.5, 0.5, -0.5, 0.5])
    cbar = fig.colorbar(im, ax=ax)
    ax.set_xlabel(r'$k_x$')
    @savefig flux.png
    ax.set_ylabel(r'$k_y$')

    # Compute the Chern number by integrating out the Berry curvature flux.
    chern_number = fluxes.sum() / (2 * np.pi)
    print('Chern number: %.4f' % chern_number)


There are two peaks of Berry flux with opposite sign at K and K'. The Berry flux exactly compensates
(it is an odd function under time-reversal, so the integral over the whole Brillouin zone vanishes).
To get an overall Berry curvature, let us break time-reversal symmetry. We add spin-orbit coupling terms that
point out-of the plane to all the bonds. We do not use the symmetry functionality of `topwave` in this case, because
we want to explicitly break time-reversal symmetry by introducing fluxes à la Haldane (which regular
spin-orbit coupling does not do).

The orientation of the couplings is generated automatically and arbitrary. We could just stare at the table
with the couplings above to see how to choose the sign of the spin-orbit terms. We want to make sure the next-nearest
neighbor couplings form closed flux loops. Let us plot how the lattice vectors connect the sublattices to see how to
choose the signs of the spin-orbit terms.

.. ipython:: python

    fig, ax = plt.subplots()
    origin_A = model.structure[0].frac_coords[:2]
    origin_B = model.structure[1].frac_coords[:2]
    for index_A, index_B in zip(range(3,6), range(6, 9)):
        coupling_A, coupling_B = model.couplings[index_A], model.couplings[index_B]
        arrow_A, arrow_B = coupling_A.lattice_vector[:2], coupling_B.lattice_vector[:2]
        ax.arrow(*origin_A, *arrow_A, head_width=0.1, color='deeppink', length_includes_head=True)
        ax.arrow(*origin_B, *arrow_B, head_width=0.1, color='red', length_includes_head=True)
        origin_A += arrow_A
        origin_B += arrow_B

    @savefig graphene_haldane_flux.png
    ax.set_aspect('equal')

The pink triangle does not form a closed loop (no finite flux/winding. We can either flip the orientation of the coupling,
or just account for the orientation of the bond when putting the spin-orbit interaction. Let us do the former and plot
the triangles again.

.. ipython:: python

    model.invert_coupling(3)

    fig, ax = plt.subplots()
    origin_A = model.structure[0].frac_coords[:2]
    origin_B = model.structure[1].frac_coords[:2]
    for index_A, index_B in zip(range(3,6), range(6, 9)):
        coupling_A, coupling_B = model.couplings[index_A], model.couplings[index_B]
        arrow_A, arrow_B = coupling_A.lattice_vector[:2], coupling_B.lattice_vector[:2]
        ax.arrow(*origin_A, *arrow_A, head_width=0.1, color='deeppink', length_includes_head=True)
        ax.arrow(*origin_B, *arrow_B, head_width=0.1, color='red', length_includes_head=True)
        origin_A += arrow_A
        origin_B += arrow_B

    @savefig graphene_haldane_flux_inverted.png
    ax.set_aspect('equal')

Perfect! We the loops are closed and have the same sense of rotation. So let us assign some
spin-orbit terms that point out of plane to all these bonds and plot the spectrum, and the flux again.
We could choose the next-nearest neighbors based on their symmetry classification again, but let us select them based on
distance this time (which we just read off the tables from above).


.. ipython:: python

    # this sets the strength of the complex hopping term
    lamda = 0.03
    model.set_spin_orbit(attribute_value=1.42,
                         vector=[0, 0, 1],
                         strength=lamda,
                         attribute='distance')

    spec = tp.Spec(model, path)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    for band in spec.energies.T:
        ax1.plot(band, ls='-', lw=2.5, c='deeppink')
    ax1.set_xticks(path.node_indices)
    ax1.set_xticklabels([r'$\Gamma$', 'M', 'K', r'$\Gamma$'])

    fluxes = np.array([tp.Spec(model, plaquette).get_berry_phase([0]) for plaquette in plaquettes])

    im = ax2.imshow(fluxes.reshape((num_x, num_y)), origin='lower', cmap='PiYG', vmin=-np.pi, vmax=np.pi, extent=[-0.5, 0.5, -0.5, 0.5])
    cbar = fig.colorbar(im, ax=ax2)
    ax2.set_xlabel(r'$k_x$')
    ax2.set_ylabel(r'$k_y$')
    fig.set_size_inches(10, 3)
    @savefig graphene_gapped.png
    plt.tight_layout()

Let's focus on the plot on the left. A small gap has openend. Seems like we did everything right! Why does the
Berry flux look like a QR-code though? With the spin orbit coupling, we introduced a spin-dependent term, so the size
size of our Hilbert space was doubled. We are looking at four pairwise degenerate bands, not at two anymore. We can
confirm by checking the number of eigenvalues at any k-point (or by calling the `check_if_spinful`-method).

.. ipython:: python

    model.check_if_spinful()
    spec.energies[0].shape

The (abelian) Berry curvature is not well-defined for degenerate bands. In the Haldane model, the spinless or rather
spin-polarized case is considered. What we are looking at now is actually two copies of the Haldane model,
the so-called Kane-Mele model. We can spin-polarize our system by applying a strong external magnetic field.

.. ipython:: python

    magnetic_field_direction = [0, 0, 1]
    magnetic_field_strength = -30
    model.set_zeeman(magnetic_field_direction, magnetic_field_strength)
    spec = tp.Spec(model, path)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    for band in spec.energies.T:
        ax1.plot(band, ls='-', lw=2.5, c='deeppink')
    ax1.set_xticks(path.node_indices)
    ax1.set_xticklabels([r'$\Gamma$', 'M', 'K', r'$\Gamma$'])

    fluxes = np.array([tp.Spec(model, plaquette).get_berry_phase([0]) for plaquette in plaquettes])

    im = ax2.imshow(fluxes.reshape((num_x, num_y)), origin='lower', cmap='PiYG', vmin=-np.pi, vmax=np.pi, extent=[-0.5, 0.5, -0.5, 0.5])
    cbar = fig.colorbar(im, ax=ax2)
    ax2.set_xlabel(r'$k_x$')
    ax2.set_ylabel(r'$k_y$')
    fig.set_size_inches(10, 3)
    @savefig graphene_gapped_haldane.png
    plt.tight_layout()

    chern_number = fluxes.sum() / (2 * np.pi)
    print('Chern number: %.4f' % chern_number)

The external field separated the spin-up and -down branches. The spin-orbit

