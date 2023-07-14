Benalcazar-Bernevig-Hughes
==========================

The two-dimensional Benalcazar-Bernevig-Hughes (BBH) model is a minimal model for higher order topological insulator with
quadrupolar order.

Let's define a function that returns the BBH model with a given set of parameters.


.. ipython:: python

    from pymatgen.core.structure import Lattice, Structure

    def get_BBH_model(gamma: float, lamda: float, delta: float) -> tp.TightBindingModel:
        a = b = 1
        c = 10
        lattice = np.diag([a, b, c])
        coords = [[0.5, 0.5, 0], [0, 0, 0], [0, 0.5, 0], [0.5, 0, 0]]
        structure = Structure.from_spacegroup(sg=1, lattice=lattice, species=['C'] * 4, coords=coords)
        model = tp.TightBindingModel(structure)
        model.generate_couplings(0.5, 1)
        # solid lines
        model.invert_coupling(0)
        model.set_coupling(0, gamma)
        model.set_coupling(1, gamma)
        model.set_coupling(6, gamma)
        # dashed line
        model.invert_coupling(7)
        model.set_coupling(7, -gamma)
        # intercell hoppings
        # solid lines
        model.invert_coupling(2)
        model.set_coupling(2, lamda)
        model.set_coupling(3, lamda)
        model.set_coupling(4, lamda)
        # dashed line
        model.invert_coupling(5)
        model.set_coupling(5, -lamda)
        model.set_onsite_scalar(0, delta)
        model.set_onsite_scalar(1, delta)
        model.set_onsite_scalar(2, -delta)
        model.set_onsite_scalar(3, -delta)
        return model

Let's print the couplings and check that everything looks like in the reference
(the sign structure of the hoppings is important).

.. ipython:: python

    model = get_BBH_model(0.5, 1, 0.001)
    model.show_couplings()


Next, we create a path through the two-dimensional Brillouin zone and plot the band structure.

.. ipython:: python

    path = tp.Path([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]])
    spec = tp.Spec(get_BBH_model(0.5, 1, 0.001), path)
    fig, ax = plt.subplots()
    for band in spec.energies.T:
        ax.plot(band, c='k', ls='-')
    ax.set_xticks(path.node_indices);
    ax.set_xlim(0, path.node_indices[-1]);
    @savefig bbh_spaghetti.png
    ax.set_xticklabels([r'$\Gamma$', 'X', 'M', r'$\Gamma$']);

It's an insulator. However, there is a gap-closing point that marks a topological phasetransition.
We can calculate the dispersion on a finite sheet at the Gamma point and sweep through the first
parameter of the model to visualize the topological phase transition. We also project the eigenstates
onto the boundaries of the system to confirm that the zero energy modes are actually edge modes.

.. ipython:: python

    gammas = np.linspace(-1.5, 1.5, 51)
    num_cells = 6
    scaling_factors = [num_cells, num_cells, 1]
    delta = 0.001
    supercell = tp.get_supercell(get_BBH_model(gammas[0], 1, delta), scaling_factors)
    supercell.set_open_boundaries('xy')
    spec = tp.Spec(supercell, [0, 0, 0])
    energies = spec.energies
    wavefunctions = spec.psi
    projections = tp.get_projections(spec, {'unit_cell_x': [0, num_cells - 1], 'unit_cell_y': [0, num_cells - 1]})
    for gamma in gammas[1:]:
        supercell = tp.get_supercell(get_BBH_model(gamma, 1, delta), scaling_factors)
        supercell.set_open_boundaries('xy')
        spec = tp.Spec(supercell, [0, 0, 0])
        energies = np.vstack((energies, spec.energies))
        wavefunctions = np.vstack((wavefunctions, spec.psi))
        projections = np.vstack((projections, tp.get_projections(spec, {'unit_cell_x': [0, num_cells - 1], 'unit_cell_y': [0, num_cells - 1]})))
    fig, ax = plt.subplots()
    for band, projection in zip(energies.T, projections.T):
        ax.plot(gammas, band, ls='-', c='midnightblue')
        ax.scatter(gammas, band, c=projection, alpha=0.5, cmap=plt.get_cmap('Reds'), vmin=0, vmax=1)
    ax.set_xlim(gammas[0], gammas[-1]);
    ax.set_xlabel(r'$\gamma / \lambda$');
    @savefig bbh_transition.png
    ax.set_ylabel(r'Energy');

We found surface modes! In the figure above it looks like the gap closing is slightly away from gamma / lambda = 1.
However that is just because of the small size of our finite system.

If there are surface modes, we should be able to diagnose that by doing a bulk calculation
and calculating the Chern number by e.g. tracking the evolution of the Wannier charge
centers throughout the Brillouin zone. Let's try that.


.. ipython:: python

    line_cover = tp.get_ine
    model = tp.get_supercell(get_BBH_model(0.5, 1, delta), scaling_factors)
    supercell.set_open_boundaries('xy')


Weird. Why is there no well we have to calculate

.. ipython:: python

    num_cells = 10
    scaling_factors = [num_cells, num_cells, 1]
    delta = 0.001
    supercell = tp.get_supercell(get_BBH_model(0.5, 1, delta), scaling_factors)
    supercell.set_open_boundaries('xy')
    spec = tp.Spec(supercell, [0, 0, 0])
    density = spec.get_particle_density(0)
    # integrate out the sublattice degree of freedom
    density = density[:, :, 0, :, 0].sum(axis=2)

    fig, ax = plt.subplots()
    ax.imshow(density, origin='lower', cmap='bwr', extent=[1, num_cells, 1, num_cells])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    @savefig bbh_corner_modes.png
    ax.set_title('Electron Density')