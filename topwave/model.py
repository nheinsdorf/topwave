from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import numpy.typing as npt
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.groups import SpaceGroup
from scipy.linalg import norm
from scipy.optimize import minimize
from tabulate import tabulate

from topwave.constants import G_LANDE, MU_BOHR
from topwave.coupling import Coupling
from topwave.types import RealList, Vector, VectorList
from topwave import util

__all__ = ["Model", "SpinWaveModel", "TightBindingModel"]

class Model(ABC):
    """Base class that is used to build a model.

    This is an **abstract** base class. Use its child classes to instantiate a model.

    Examples
    --------

    Create a cubic lattice of cobalt atoms and space group symmetry P23 (#195) with pymatgen and use it to create a SpinWaveModel.

    .. ipython:: python

        from pymatgen.core.structure import Structure
        structure = Structure.from_spacegroup(sg=195, lattice=np.eye(3), species=['Co'], coords=[[0, 0, 0]])
        print(structure)

        model = tp.model.SpinWaveModel(structure)

    See Also
    --------
    :class:`topwave.model.SpinWaveModel`, :class:`topwave.model.TightBindingModel`
    """

    def __init__(self,
                 structure: Structure,
                 import_site_properties: bool = False) -> None:

        # NOTE: should I copy?
        self.structure = structure.copy()
        self.type = self.get_type()

        # allocate site properties and enumerate them by an index
        if not import_site_properties:
            for _, site in enumerate(self.structure):
                site.properties['index'] = _
                site.properties['magmom'] = None
                site.properties['onsite_scalar'] = 0.
                site.properties['onsite_vector'] = np.zeros(3, dtype=float)
                site.properties['orbitals'] = 1
                site.properties['Rot'] = None

                # supercell and twisted properties
                site.properties['cell_vector'] = None
                site.properties['uc_site_index'] = None
                site.properties['layer'] = None
        self.scaling_factors = None
        self.normal = None
        self.twist_tuple = None

        # put zero magnetic field
        self.zeeman = np.zeros(3, dtype=float)

        # allocate an empty list for the couplings
        self.couplings = []

    def delete_all_couplings(self) -> None:
        """Deletes all couplings."""

        self.couplings = []

    def generate_couplings(self,
                           max_distance: float,
                           space_group: int) -> None:
        """Generates couplings up to a distance and groups them based on the space group symmetry.

        Parameters
        ----------
        max_distance: float
            The distance up to which the couplings are generated.
        space_group: int
            The international number of the space group (1 to 230) that is used to group the couplings.

        Examples
        --------
        Find and create the nearest-neighbors and group them by space group symmetry P23 (#195).

        .. ipython:: python

            model.generate_couplings(max_distance=1, space_group=195)
            model.show_couplings()

        """

        neighbors = self.structure.get_symmetric_neighbor_list(max_distance, sg=space_group, unique=True)
        self.delete_all_couplings()
        index = 0
        for site1_id, site2_id, lattice_vector, _, symmetry_id, symmetry_op in zip(*neighbors):
            site1 = self.structure[site1_id]
            site2 = self.structure[site2_id]
            for orbital1, orbital2 in product(range(site1.properties['orbitals']), range(site2.properties['orbitals'])):
                coupling = Coupling(index, lattice_vector, site1, orbital1, site2, orbital2, int(symmetry_id), symmetry_op)
                self.couplings.append(coupling)
                index += 1

    def get_couplings(self,
                      attribute: str,
                      value: int | float) -> list[Coupling]:
        """Return couplings selected by some attribute.

        Parameters
        ----------
        attribute: str
            The attribute by which the couplings are selected. Options are 'is_set', 'index', 'symmetry_id' or 'distance'.
        value: int | float
            The value of the selected attribute.

        Returns
        -------
        list[Coupling]
            A list that contains the couplings that match the value of the selected attribute.

        Examples
        --------
        Select one of the couplings based on its index.

        .. ipython:: python

            model.get_couplings('index', 1)

        Select all the couplings based on their symmetry.

        .. ipython:: python

            model.get_couplings('symmetry_id', 0)

        See Also
        --------
        :class:`topwave.util.coupling_selector`

        """

        indices = util.coupling_selector(attribute=attribute, value=value, model=self)
        return [self.couplings[index] for index in indices]

    # NOTE: should I get rid of this and just replace it with get_couplings in spec?
    def get_set_couplings(self) -> list[Coupling]:
        """Returns couplings that have been assigned some exchange.

        Returns
        -------
        list[Coupling]
            A list that contains the couplings that were assigned some exchange.

        See Also
        --------
        :class:`topwave.model.Model.get_couplings`

        """

        indices = util.coupling_selector(attribute='is_set', value=True, model=self)
        return [self.couplings[index] for index in indices]

    @abstractmethod
    def get_type(self) -> str:
        """Returns the type of the model."""

    def invert_coupling(self,
                        index: int) -> None:
        """Inverts the orientation of a coupling.

        Parameters
        ----------
        index: int
            The index of the coupling that is inverted.

        Examples
        --------

        The lattice vector that connects a coupling before and after inversion:

        .. ipython:: python

            print(f'R = {model.couplings[0].lattice_vector}')
            model.invert_coupling(0)
            print(f'R = {model.couplings[0].lattice_vector}')


        """

        coupling = self.couplings[index]
        site1, site2 = coupling.site1, coupling.site2
        orbital1, orbital2 = coupling.site1.properties['orbitals'], coupling.site2.properties['orbitals']
        lattice_vector = coupling.lattice_vector
        symmetry_id, symmetry_op = coupling.symmetry_id, coupling.symmetry_op
        inverted_coupling = Coupling(index, -lattice_vector, site2, orbital2, site1, orbital1, symmetry_id, symmetry_op)
        self.couplings[index] = inverted_coupling

    def set_coupling(self,
                     attribute_value: int | float,
                     strength: float,
                     attribute: str = 'index') -> None:
        """Assigns (scalar) hopping/exchange to a selection of couplings.

        Parameters
        ----------
        attribute_value: int | float
            The value of the selected attribute.
        strength: float
            Strength of the hopping/exchange.
        attribute: str
            The attribute by which the couplings are selected. Options are 'is_set', 'index', 'symmetry_id' or 'distance'.

        Examples
        --------

        Assign ferromagnetic exchange with J=1 to all nearest neighbors. See the 'strength' column in the output.

        .. ipython:: python

            model.set_coupling(attribute_value=0, strength=1, attribute='symmetry_id')
            model.show_couplings()

        """

        couplings = self.get_couplings(attribute=attribute, value=attribute_value)
        for coupling in couplings:
            coupling.set_coupling(strength)

    def set_moments(self,
                    orientations: VectorList,
                    magnitudes: RealList = None) -> None:
        """Sets the magnetic moments on each site of the structure given in lattice coordinates.

        Parameters
        ----------
        orientations: VectorList
            A list of three-dimensional vectors that specify the direction of local magnetic moment on each site.
        magnitudes: RealList
            A list of floats that specifies the magnitude of the local moment for each site. If None, the length
            of the input vector is used. Default is None.

        Examples
        --------
        Put spin-1/2 moments on the site that point into the 111-direction.

        .. ipython:: python

            model.set_moments([[1, 1, 1]], [0.5])
            model.show_site_properties()

        """

        for _, (orientation, site) in enumerate(zip(orientations, self.structure)):
            # compute or save the magnitude (in lattice coordinates!).
            magnitude = norm(orientation) if magnitudes is None else magnitudes[_]
            orientation = np.array(orientation, dtype=np.float64).reshape((3,))
            # transform into cartesian coordinates (spin frame) and normalize.
            moment = self.structure.lattice.matrix.T @ orientation
            moment = moment / norm(moment)
            # calculate rotation matrix that rotates the spin to the quantization axis.
            site.properties['Rot'] = util.rotate_vector_to_ez(moment)
            site.properties['magmom'] = moment * magnitude

    def set_onsite_scalar(self,
                          index: int,
                          strength: float,
                          space_group: int = 1) -> None:
        """Sets a scalar onsite energy to a given site.

        For a TightBindingModel this is a site or orbital dependent onsite energy. For SpinWaveModel this term is ignored.

        Parameters
        ----------
        index: int
            The index of the site.
        strength: float
            The strength of the onsite term.
        space_group: int
            If a compatible space group symmetry is selected, the term will automatically be assigned to all
            symmetrically equivalent sites. Default is None.

        Examples
        --------
        We assign a onsite energy of E = 0.25 to the zeroth site. See the onsite scalar column in the output.

        .. ipython:: python

            model.set_onsite_scalar(0, 0.25)
            model.show_site_properties()

        """

        space_group = SpaceGroup.from_int_number(space_group)
        coordinates = space_group.get_orbit(self.structure[index].frac_coords)
        for coordinate in coordinates:
            cartesian_coordinate = self.structure.lattice.get_cartesian_coords(coordinate)
            site = self.structure.get_sites_in_sphere(cartesian_coordinate, 1e-06)[0]
            site.properties['onsite_scalar'] = float(strength)

    def set_onsite_vector(self,
                          index: int,
                          vector: Vector,
                          strength: float = None,
                          space_group: int = 1) -> None:
        """Sets an onsite vector to a given site.

        For a SpinWaveModel this corresponds to a single-ion anisotropy. For a TightBindingModel to a local magnetic field.

        Parameters
        ----------
        index: int
            The index of the site.
        vector: Vector
            The orientation of the term.
        strength: float
            The strength of the onsite term. If None, the length of the input vector is used. Default is None.
        space_group: int
            If a compatible space group symmetry is selected, the term will automatically be assigned to all
            symmetrically equivalent sites and the assigned vector will be rotated accordingly. Default is None.

        Notes
        -----
        If the model is a :class:`topwave.model.TightBindingModel` calling this method will make the model spinful.
        The dimension of the Hilbert space will be doubled. See :class:`topwave.model.TightBindingModel.check_if_spinful`.

        Examples
        --------
        We assign a single-ion anisotropy of strength A = 0.1 along the 111-direction. See the
        onsite vector column in the output.

        .. ipython:: python

            model.set_onsite_vector(0, [1, 1, 1], 0.1)
            model.show_site_properties()

        """

        input_vector = util.format_input_vector(orientation=vector, length=strength)
        space_group = SpaceGroup.from_int_number(space_group)
        coordinates, operations = space_group.get_orbit_and_generators(self.structure[index].frac_coords)
        for coordinate, operation in zip(coordinates, operations):
            cartesian_coordinate = self.structure.lattice.get_cartesian_coords(coordinate)
            site = self.structure.get_sites_in_sphere(cartesian_coordinate, 1e-06)[0]
            site.properties['onsite_vector'] = operation.apply_rotation_only(input_vector)

    def set_open_boundaries(self,
                            direction: str = 'xyz') -> None:
        """Sets the exchange/hopping and DM/SOC at the chosen boundary to zero.

        Parameters
        ----------
        direction: str
            The directions in lattice vectors along which the boundary conditions are set to open.
            'x' is along the direction of the first lattice vector. 'yz' along the other two, and 'xyz/ in all directions.

        Examples
        --------
        Set open boundaries in y- and z-direction so that we have a one-dimensional chain along x. See the strength column
        in the output.

        .. ipython:: python

            model.set_open_boundaries('yz')
            model.show_couplings()

        See Also
        --------
        :class:`topwave.util.get_boundary_couplings`

        """

        boundary_indices = util.get_boundary_couplings(model=self, direction=direction)
        for index in boundary_indices:
            self.unset_coupling(attribute_value=index, attribute='index')

    def set_spin_orbit(self,
                       attribute_value: int | float,
                       vector: Vector,
                       strength: float = None,
                       attribute: str = 'index') -> None:
        """Assigns spin dependent hopping/antisymmetric exchange to a selection of couplings.

        If the couplings are grouped by symmetry, the assigned exchanges will be rotated automatically according to
        space group symmetry.

        Parameters
        ----------
        attribute_value: int | float
            The value of the selected attribute.
        vector: Vector
            Orientation of the term.
        strength: float
            Strength of the term. If None, the length of the orientation is used. Default is None.
        attribute: str
            The attribute by which the couplings are selected. Options are 'is_set', 'index', 'symmetry_id' or 'distance'.

        Examples
        --------

        Assign antisymmetric exchange with finite z-component along nearest neighbor in the x-direction.
        See the 'spin-orbit vector' column in the output.

        .. ipython:: python

            model.set_spin_orbit(0, [0, 0, 0.05])
            model.show_couplings()

        """

        input_vector = util.format_input_vector(orientation=vector, length=strength)
        couplings = self.get_couplings(attribute=attribute, value=attribute_value)
        for coupling in couplings:
            spin_orbit = coupling.symmetry_op.apply_rotation_only(input_vector) if attribute == 'symmetry_id' else input_vector
            coupling.set_spin_orbit(spin_orbit)

    def set_zeeman(self,
                   orientation: Vector,
                   strength: float = None) -> None:
        """Sets a global Zeeman term.

        .. admonition:: Tip
            :class: tip

            For ferromagnetic systems a small Zeeman term is recommended to lift the soft mode above zero energy.

        Parameters
        ----------
        orientation: Vector
            The orientation of the Zeeman term.
        strength: float
            The strength of the Zeeman term in units of **Tesla**. If None, the length of the orientation is used. Default is None.

        Notes
        -----
        If the model is a :class:`topwave.model.TightBindingModel` calling this method will make the model spinful.
        The dimension of the Hilbert space will be doubled. See :class:`topwave.model.TightBindingModel.check_if_spinful`.

        Examples
        --------

        We set a Zeeman field of 0.2 Tesla in the 111-direction.

        .. ipython:: python

            model.set_zeeman([1, 1, 1], 0.2)
            model.show_site_properties()

        """


        self.zeeman = util.format_input_vector(orientation=orientation, length=strength)

    def show_couplings(self) -> None:
        """Prints the couplings."""

        header = ['index', 'symmetry index', 'symmetry operation', 'distance', 'lattice_vector', 'sublattice_vector',
                  'site1', 'orbital1', 'site2', 'orbital2', 'strength', 'spin-orbit vector']
        table = []
        for coupling in self.couplings:
            table.append([coupling.index, coupling.symmetry_id, coupling.symmetry_op.as_xyz_string(), coupling.distance,
                          coupling.lattice_vector, coupling.sublattice_vector, coupling.site1.properties['index'],
                          coupling.orbital1, coupling.site2.properties['index'], coupling.orbital2, coupling.strength,
                          coupling.spin_orbit])

        print(tabulate(table, headers=header, tablefmt='fancy_grid'))

    def show_site_properties(self) -> None:
        """Prints the site properties."""

        header = ['index', 'species', 'orbitals', 'coordinates (latt.)', 'coordinates (cart.)', 'magmom',
                  'onsite scalar', 'onsite vector', 'unit cell index', 'supercell vector', 'layer']
        table = []
        for site in self.structure:
            table.append([site.properties['index'], site.species, site.properties['orbitals'], site.frac_coords,
                          site.coords, site.properties['magmom'], site.properties['onsite_scalar'],
                          site.properties['onsite_vector'], site.properties['uc_site_index'],
                          site.properties['cell_vector'], site.properties['layer']])

        print(tabulate(table, headers=header, tablefmt='fancy_grid'))
        print(f'Zeeman: {self.zeeman}')
        print(f'Supercell Size: {self.scaling_factors}')

    def unset_coupling(self,
                       attribute_value: int | float,
                       attribute: str = 'index') -> None:
        """Removes exchanges from a coupling and makes it unset.

        Parameters
        ----------
        attribute: str
            The attribute by which the couplings are selected. Options are 'is_set', 'index', 'symmetry_id' or 'distance'.
        value: int | float
            The value of the selected attribute.

        Examples
        --------
        We unset the coupling along the x-direction.

        .. ipython:: python

            print(model.couplings[0].is_set)
            model.unset_coupling(0)
            model.couplings[0].is_set

        """

        indices = util.coupling_selector(attribute=attribute, value=attribute_value, model=self)
        for _ in indices:
            self.couplings[_].unset()

    # TODO: write unset_onsite_scalar and unset_onsite_vector (and maybe unset_all_site_properties?)
    def unset_moments(self):
        """Unsets all magnetic moments of the structure.

        Examples
        --------
        We set the magnetic moment on all sites to zero.

        .. ipython:: python

            model.unset_moments()
            model.show_site_properties()

        """

        for site in self.structure:
            site.properties['magmom'] = None

    def write_cif(self,
                  path: str,
                  write_magmoms: bool = True) -> None:
        """Saves the structure to a .mcif file.

        Parameters
        ----------
        path: str
            Where to save the structure.
        write_magmoms: bool
            If true the magnetic moments are written into the mcif file. Default is True.

        """

        if self.type == 'tightbinding':
            for site in self.structure:
                site.properties['magmom'] = site.properties['onsite_vector']
                CifWriter(self.structure, write_magmoms=write_magmoms).write_file(path)
                self.unset_moments()
        else:
            CifWriter(self.structure, write_magmoms=write_magmoms).write_file(path)


class SpinWaveModel(Model):
    """Class for Linear Spinwave models.

    Examples
    --------

    Create a ferromagnetic chain of cobalt atoms.

    .. ipython:: python

        # Create a three-dimensional cubic structure.
        from pymatgen.core.structure import Structure
        structure = Structure.from_spacegroup(sg=1, lattice=np.eye(3), species=['Co'], coords=[[0, 0, 0]])

        # Create a SpinWaveModel.
        model = tp.model.SpinWaveModel(structure)

        # Put a ferromagnetic configuration of local moments.
        model.set_moments([[0, 0, 1]])

        # Couple the local moments ferromagnetically along the x-direction.
        model.generate_couplings(1, 1)
        model.set_coupling(0, strength=-1)

        # Put local mom
        model.show_couplings()
        model.show_site_properties()

    See Also
    --------
    :class:`topwave.model.Model`, IntroductionToSpinWaveModel

    """



    # NOTE: if I can make the multiple inheritance with the abstract get-type method work, delete this again.
    def __init__(self,
                 structure: Structure,
                 import_site_properties: bool = False) -> None:
        super().__init__(structure, import_site_properties)
        self.type = 'spinwave'

    def get_classical_energy(self,
                             per_spin: bool = True) -> float:
        """Computes the classical ground state energy of the model with its current orientation of local moments.

        Parameters
        ----------
        per_spin: bool
            If true, the energy is divided by the number of local moments in the model. Default is True.

        Returns
        -------
        float
            The classical energy.

        Examples
        --------

        Compute the classical energy of the ferromagnetic Heisenberg chain with J=-1 meV.

        .. ipython:: python

            model.get_classical_energy()

        """

        energy = 0
        # exchange energy
        for coupling in self.couplings:
            energy += coupling.get_energy()

        # Zeeman energy and anisotropies
        for site in self.structure:
            magmom = site.properties['magmom']
            vector = site.properties['onsite_vector']
            energy += magmom @ np.diag(vector) @ magmom
            energy -= MU_BOHR * G_LANDE * (self.zeeman @ magmom)

        if per_spin:
            return energy / len(self.structure)
        return energy

    def get_type(self) -> str:
        """Returns the type of the Model.

        Returns
        -------
        str
            Returns 'spinwave'.

        """

        return 'spinwave'

    @staticmethod
    def __get_classical_energy_wrapper(directions: VectorList,
                                       magnitudes: RealList,
                                       model: Model) -> float:
        """Private method that is passed as a function to the minimizer."""

        directions = np.array(directions, dtype=float).reshape((-1, 3))
        model.set_moments(directions, magnitudes)
        return model.get_classical_energy()

    def get_classical_groundstate(self,
                                  random_init=False) -> None:
        """Minimize the classical energy by changing the orientation of the magnetic moments.

        This method can be used to find the classical groundstate configuration of a model. The magnitude of the
        local moments is fixed, and only their orientation is adjusted. Scipy's minimize function is used.

        .. admonition:: Frustration and (In)Commensurability
            :class: tip

            If the classical ground state is frustrated, there might not be a unique configuration that minimizes
            the classical energy. Consider putting an external magnetic field or single-ion anisotropies in that case.
            If the system is incommensurable consider minimizing the energy of a supercell and play around with open
            and periodic boundary conditions.

        .. admonition:: Todo
            :class: todo

            Add example and references to supercell and set_open_boundaries.

        Parameters
        ----------
        random_init: bool
            If True, the orientation of magnetic moments is randomly initialized. The moments need to be set before
            regardless to extract their magnitudes. If false the set orientations are used. Default is False.
        """

        moments = np.array([site.properties['magmom'] for site in self.structure], dtype=float)
        magnitudes = norm(moments, axis=1)

        # get initial moments
        x_0 = np.random.rand(self.structure.num_sites, 3) if random_init else moments

        res = minimize(SpinWaveModel.__get_classical_energy_wrapper, x_0, args=(magnitudes, self))

        # normalize the final configuration
        res.x = (res.x.reshape((-1, 3)).T / norm(res.x.reshape((-1, 3)), axis=1)).flatten(order='F')
        return res

    def set_single_ion_anisotropy(self,
                                  index: int,
                                  vector: Vector,
                                  strength: float = None,
                                  space_group: int = 1) -> None:
        """Sets a single-ion anisotropy to a given site. Same as Model.set_onsite_vector.

        Parameters
        ----------
        index: int
            The index of the site.
        vector: Vector
            The orientation of the term.
        strength: float
            The strength of the onsite term. If None, the length of the input vector is used. Default is None.
        space_group: int
            If a compatible space group symmetry is selected, the term will automatically be assigned to all
            symmetrically equivalent sites and the assigned vector will be rotated accordingly. Default is None.

        Examples
        --------
        We assign a single-ion anisotropy of strength A = 0.1 along the 001-direction. See the
        onsite vector column in the output.

        .. ipython:: python

            model.set_onsite_vector(0, [0, 0, 1], 0.1)
            model.show_site_properties()

        See Also
        --------
        :class:`topwave.model.Model.set_onsite_vector`

        """

        self.set_onsite_vector(index=index, vector=vector, strength=strength, space_group=space_group)


class TightBindingModel(Model):
    """Class for Linear Spinwave models.

    Examples
    --------

    Create a tightbinding model of graphene.

    .. ipython:: python

        from pymatgen.core.structure import Lattice, Structure
        lattice = Lattice.hexagonal(1.42, 10)
        structure = Structure.from_spacegroup(sg=191, lattice=lattice, species=['C'], coords=[[1 / 3, 2 / 3, 0]])
        print(structure)

        graphene = tp.model.TightBindingModel(structure)

    See Also
    --------
    :class:`topwave.model.Model`, IntroductionToTightBindingModel

    """

    # NOTE: if I can make the multiple inheritance with the abstract get-type method work, delete this again.
    def __init__(self,
                 structure: Structure,
                 import_site_properties: bool = False) -> None:
        super().__init__(structure, import_site_properties)
        self.type = 'tightbinding'

    def check_if_spinful(self) -> bool:
        """Checks whether the model is spinful or spinless (polarized).

        This method checks whether there are any terms set that are spin dependent. This includes Zeeman terms,
        spin-orbit coupling, or local magnetic field (onsite vector terms).

        Returns
        -------
        bool
            True if the system is spinful, False if not.

        Notes
        -----
        If the model is a :class:`topwave.model.TightBindingModel` calling this method will make the model spinful.
        The dimension of the Hilbert space will be doubled. See :class:`topwave.model.TightBindingModel.check_if_spinful`.


        Examples
        --------

        .. ipython:: python

            graphene.check_if_spinful()

        See Also
        --------
        :class:`topwave.model.Model.set_zeeman`, :class:`topwave.model.Model.set_spin_orbit`, :class:`topwave.model.Model.set_onsite_vector`
        """

        couplings = self.get_set_couplings()
        has_spin_orbit = any(any(coupling.spin_orbit != np.zeros(3, dtype=float)) for coupling in couplings)
        has_onsite_vector = any(any(site.properties['onsite_vector'] != np.zeros(3, dtype=float)) for site in self.structure)
        has_zeeman = any(self.zeeman != np.zeros(3, dtype=float))
        return any([has_spin_orbit, has_onsite_vector, has_zeeman])

    def get_type(self) -> str:
        """Overrides the 'get_type'-method to return the tightbinding type."""

        return 'tightbinding'

    def set_orbitals(self,
                     index: int,
                     num_orbitals: int):
        """Sets the number of orbitals on a given site.

        .. admonition:: Coming soon!
            :class: warning

            Still in Development. So far this only sets the number of orbitals as a site property, but it doesn't
            influence the spectrum.

        """

        self.structure[index].properties['orbitals'] = num_orbitals
