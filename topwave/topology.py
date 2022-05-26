from topwave.model import Spec

import numpy as np


class WCC_evolution(object):
    """
    Class for the evolution of Wannier Charge Centers on some path

    Parameters
    ----------
    model : topwave.model.Model
        The model the evolution of which is to be calculated.
    loops : list
        A list of loops where each loop is a closed set of k-points.
    occ : list
        List of integers that specify all occupied bands.

    Attributes
    ----------
    MODEL : topwave.model.Model
        This is where model is stored.
    NLOOP : int
        Number of Wilson loops
    KS : list
        This is where loops is stored. It is a numloops-long list of
        numk-points x 3 numpy.ndarrays.
    OCC : list
        This is where occ is stored.
    NOCC : int
        This gives the number of occupied bands.
    N : int
        Number of magnetic sites in model.
    WCCs : numpy.ndarray
        This is where the Wannier Charge Centers are stored. Shape is
        NOCC x NLOOP. It's not converted into a numpy.ndarray in case the


    Methods
    -------
    generate_couplings(maxdist, sg=None):
        Given a maximal distance (in Angstrom) all periodic bonds are
        generated and grouped by symmetry based on the provided sg.

    """

    def __init__(self, model, loops, occ, test):
        # allocate memory for the results
        self.NLOOP = len(loops)
        self.OCC = occ
        self.NOCC = len(occ)
        self.N = len(model.STRUC)
        self.WCCs = np.zeros((self.NOCC, self.NLOOP))

        # for each loop generate the spectrum and calculate its WCC
        for _, loop in enumerate(loops):
            spec = Spec(model, loop)
            if test:
                spec.wilson_loop_test(occ)
                print('testing')
            else:
                spec.wilson_loop(occ)
            self.WCCs[:, _] = spec.wannier_center
