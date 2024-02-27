from io import StringIO
from itertools import product

import numpy as np
import numpy.typing as npt
from pymatgen.core.structure import Lattice, Structure
from topwave.coupling import Coupling
from topwave.model import Model, SpinWaveModel, TightBindingModel
from topwave import util
from topwave.types import RealList, Vector


def parse_from_wannier_hr_dat(filename: str) -> np.ndarray[np.float64]:
    """Wannier90 hopping parser of hr_dat files.

    Parameters
    ----------
    filename: str
        The Wannier90 hr_dat file.

    Returns
    -------
    dict
        A dictionary containing the hoppings (as dictionaries).
    """

    with open(filename) as f:
        f.readline()
        num_orbitals = int(f.readline())
        num_wigner_seitz_cells = int(f.readline())
        lines = f.readlines()

    num_header_lines = int(np.ceil(num_wigner_seitz_cells / 15.))
    degeneracies = np.array("".join(lines[:num_header_lines]).split(), dtype=int)

    hoppings = "".join(lines[num_header_lines:])
    hoppings = np.loadtxt(StringIO(hoppings))
    hoppings = hoppings.reshape(num_wigner_seitz_cells, num_orbitals, num_orbitals, 7)
    # lattice_vectors = hoppings[:, 0, 0, :3].astype(int)
    # strengths = hoppings[..., 5] + 1j * hoppings[..., 6] / degeneracies[:, None, None]
    hoppings[..., 5] /= degeneracies[:, None, None]
    hoppings[..., 6] /= degeneracies[:, None, None]
    return hoppings

def set_couplings_from_wannier_hr_dat(model: TightBindingModel, filename: str, strength_cutoff=0.05) -> None:
    """Sets generates and sets couplings for a model using a hr_dat file.

    Parameters
    ----------
    model: TightBindingModel
        The model that is used. Make sure it has the right number of sites and orbitals.
    filename: str
        The Wannier90 hr_dat file.
    strength_cutoff: float
        Strength below which hoppings are discarded.
    """

    hoppings = parse_from_wannier_hr_dat(filename)

    strengths = np.abs((hoppings[..., 5] + 1j * hoppings[..., 6]).reshape((hoppings.shape[0], -1)))
    cutoff_mask = np.all((strengths < strength_cutoff), axis=1)
    hoppings = hoppings[~cutoff_mask]

    cutoff_distance = 1.005 * np.linalg.norm(np.einsum('ij, nj -> ni', model.structure.lattice.matrix.T, hoppings[:, 0, 0, :3]), axis=1).max()
    model.generate_couplings(cutoff_distance, 1)

    nums_orbitals = [site.properties['orbitals'] for site in model.structure]
    dimension = sum(nums_orbitals)
    combined_index_nodes = np.concatenate(([0], np.cumsum(nums_orbitals)[:-1]), dtype=np.int64)

    for hopping in hoppings:
        for index1, index2 in product(range(hoppings.shape[1]), range(hoppings.shape[1])):
            site1_index, site2_index = np.digitize(index1, combined_index_nodes) - 1, np.digitize(index2, combined_index_nodes) - 1


