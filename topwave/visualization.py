from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from topwave.model import Model
from topwave.util import get_span_indices

class Plot2D:
    """Class for a two-dimensional plot that shows the couplings."""

    def __init__(self, model: Model, ax: plt.Axes, normal: str = 'z') -> None:
        self.model = model
        self.ax = ax

        self.id_x, self.id_y, self.id_z = get_span_indices(normal)

        self.lattice_vector1 = model.structure.lattice.matrix[self.id_x][[self.id_x, self.id_y]]
        self.lattice_vector2 = model.structure.lattice.matrix[self.id_y][[self.id_x, self.id_y]]

        self.plot_sites()

    def plot_brillouin_zone(self, normalize: bool = False) -> None:
        """Plots the Brillouin zone."""

        reciprocal_vector1 = self.model.structure.lattice.reciprocal_lattice.matrix[self.id_x][[self.id_x, self.id_y]]
        reciprocal_vector2 = self.model.structure.lattice.reciprocal_lattice.matrix[self.id_y][[self.id_x, self.id_y]]

        if normalize:
            length1 = np.linalg.norm(self.lattice_vector1)
            length2 = np.linalg.norm(self.lattice_vector2)
            reciprocal_vector1 = reciprocal_vector1 * length2 / np.linalg.norm(reciprocal_vector1)
            reciprocal_vector2 = reciprocal_vector2 * length1 / np.linalg.norm(reciprocal_vector2)

        self._plot_parallelogram(reciprocal_vector1, reciprocal_vector2, color='red')

    def plot_couplings(self) -> None:
        """Plots the couplings in the model."""

        # TODO: implement distinction based on soc as well (override couplings __eq__ and use sets)
        couplings = self.model.get_set_couplings()
        unique_strengths, inverse = np.unique([coupling.strength for coupling in couplings], return_inverse=True)
        cmap = plt.get_cmap('gist_rainbow')
        colors = cmap(np.linspace(0, 255, len(unique_strengths)).astype('int'))
        for (coupling, color_index) in zip(couplings, inverse):
            coords1 = np.array([coupling.site1.coords[self.id_x], coupling.site1.coords[self.id_y]])
            lattice_vector = coupling.lattice_vector[self.id_x] * self.lattice_vector1 \
                             + coupling.lattice_vector[self.id_y] * self.lattice_vector2
            coords2 = np.array([coupling.site2.coords[self.id_x], coupling.site2.coords[self.id_y]]) + lattice_vector
            self.ax.plot(*zip(coords1, coords2), c=colors[color_index], zorder=1)

    def plot_lattice_vectors(self) -> None:
        """Plots the unit cell."""

        self._plot_parallelogram(self.lattice_vector1, self.lattice_vector2, color='grey')

    def _plot_parallelogram(self, vector1: npt.ArrayLike, vector2: npt.ArrayLike, color: str = 'grey') -> None:
        """Plots a parallelogram spanned by two vectors."""

        head_width = self.site_plot.get_linewidth()[0] / 4
        self.ax.arrow(0, 0, vector1[0],  vector1[1],
                      color='black', length_includes_head=True, head_width=head_width)
        self.ax.arrow(0, 0, vector2[0], vector2[1],
                      color='black', length_includes_head=True, head_width=head_width)

        polygon = Polygon(np.array([[0, 0], vector1, vector1 + vector2, vector2]),
                          closed=True)
        patch_collection = PatchCollection([polygon], alpha=0.2, color=color, zorder=0)
        self.ax.add_collection(patch_collection)

    def plot_sites(self) -> None:
        """Plots the sites of the structure. The opacity encodes different layers."""

        tol_digits = 4
        alpha_min = 0.2
        coords_z = np.array([site.coords[self.id_z] for site in self.model.structure])
        unique_coords, indices = np.unique(np.round(coords_z, tol_digits), return_inverse=True)
        num_layers = len(unique_coords)
        alphas = np.linspace(1., alpha_min, num_layers)
        self.ax.set_aspect('equal')
        for (site, index) in zip(self.model.structure, indices):
            self.site_plot = self.ax.scatter(site.coords[self.id_x], site.coords[self.id_y],
                                             alpha=alphas[index], color='blue', zorder=2)

