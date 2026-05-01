import numpy as np
from scipy.linalg import eigh

"""
tfim_hamiltonian
    Build the N-qubit TFIM Hamiltonian as a dense matrix
    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
    With periodic boundary conditions

exact_groundState
    Returns ground state energy and wavefunction.
phase_diagram
    Compute ground state energy per site accross the phase transition.
"""

# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])


def tfim_hamiltonian(N, J=1.0, h=1.0):
    dim = 2**N
    H = np.zeros((dim, dim))

    for i in range(N):
        # ZZ term between site i and i+1 (mod N)
        ops = [I] * N
        ops[i] = Z
        ops[(i + 1) % N] = Z
        ZZ = ops[0]
        for op in ops[1:]:
            ZZ = np.kron(ZZ, op)
        H -= J * ZZ

    # X term at site i
    ops = [I] * N
    ops[i] = X
    XI = ops[0]
    for op in ops[1:]:
        XI = np.kron(XI, op)
    H -= h * XI

    return H


def exact_groundState(N, J=1.0, h=1.0):
    H = tfim_hamiltonian(N, J, h)
    energies, vectors = eigh(H)
    return energies[0], vectors[:, 0]


def phase_diagram(N, J=1.0, h_values=None):
    if h_values is None:
        h_values = np.linspace(0, 2, 50)
    energies = []
    for h in h_values:
        E0, _ = exact_groundState(N, J, h)
        energies.append(E0 / N)
    return h_values, np.array(energies)
