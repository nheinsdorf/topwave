import numpy as np

ComplexVector = np.ndarray[np.complex128] | list[complex]
ComplexList = np.ndarray[np.complex128] | list[complex]
Complex2x2 = np.ndarray[np.complex128]
IntVector = np.ndarray[np.int64] | list[int]
Matrix = np.ndarray[np.complex128]
ListOfRealList = np.ndarray[np.float64] | list[np.ndarray[np.float64]] | list[list[float]]
RealList = np.ndarray[np.float64] | list[float]
Real3x3 = np.ndarray[np.float64]
SquareMatrix = np.ndarray[np.complex128]
SquareMatrixList = np.ndarray[np.complex128] | list[np.ndarray[np.complex128]]
Vector = np.ndarray[np.float64] | list[float]
VectorList = np.ndarray[np.float64] | list[np.ndarray[np.float64]] | list[list[float]]