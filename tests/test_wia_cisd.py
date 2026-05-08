import numpy as np
import pytest
from manybody_lib.closed_shell.cisd.cisd_functions import (
    compute_L,
    compute_norm,
    compute_rho_pq,
    compute_rho_pqrs,
    compute_wia,
    compute_wia_cisd,
)

N_OCC = 2
N_VIRT = 4
N_MO = N_OCC + N_VIRT


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def system(rng):
    o = slice(0, N_OCC)
    v = slice(N_OCC, N_MO)

    # Symmetric one-body matrix: h[p,q] = h[q,p]
    A = rng.standard_normal((N_MO, N_MO))
    h = (A + A.T) / 2

    # Two-body array with symmetry u[p,q,r,s] = u[q,p,s,r]
    B = rng.standard_normal((N_MO, N_MO, N_MO, N_MO))
    u = (B + B.transpose(1, 0, 3, 2)) / 2

    # CISD amplitudes
    R0 = np.float64(rng.standard_normal())
    R1 = rng.standard_normal((N_VIRT, N_OCC))

    # R2 symmetry: R2[a,b,i,j] = R2[b,a,j,i]
    C = rng.standard_normal((N_VIRT, N_VIRT, N_OCC, N_OCC))
    R2 = (C + C.transpose(1, 0, 3, 2)) / 2

    L0, L1, L2 = compute_L(R0, R1, R2)
    norm = compute_norm(L0, L1, L2, R0, R1, R2)
    R0, R1, R2 = R0 / np.sqrt(norm), R1 / np.sqrt(norm), R2 / np.sqrt(norm)
    L0, L1, L2 = L0 / np.sqrt(norm), L1 / np.sqrt(norm), L2 / np.sqrt(norm)
    assert compute_norm(L0, L1, L2, R0, R1, R2) == pytest.approx(1.0)
    rho_pq = compute_rho_pq(L0, L1, L2, R0, R1, R2)
    rho_pqrs = compute_rho_pqrs(L0, L1, L2, R0, R1, R2)

    return h, u, rho_pq, rho_pqrs, o, v


def test_compute_wia_equals_compute_wia_cisd(system):
    h, u, rho_pq, rho_pqrs, o, v = system

    w_ia_general = compute_wia(h, u, rho_pq, rho_pqrs, o, v)
    w_ia_cisd = compute_wia_cisd(h, u, rho_pq, rho_pqrs, o, v)

    np.testing.assert_allclose(
        w_ia_cisd,
        w_ia_general,
        rtol=0.0,
        atol=1e-12,
        err_msg="compute_wia_cisd does not match compute_wia for CISD densities",
    )
