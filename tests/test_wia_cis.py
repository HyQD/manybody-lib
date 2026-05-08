import numpy as np
import pytest
from manybody_lib.closed_shell.cis.cis_functions import (
    compute_L,
    compute_norm,
    compute_rho_pq,
    compute_rho_pqrs,
    compute_wia,
    compute_wia_cis,
)

N_OCC = 2
N_VIRT = 4
N_MO = N_OCC + N_VIRT


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

    # CIS amplitudes
    R0 = np.float64(rng.standard_normal())
    R1 = rng.standard_normal((N_VIRT, N_OCC))

    L0, L1 = compute_L(R0, R1)
    norm = compute_norm(L0, L1, R0, R1)
    R0, R1 = R0 / np.sqrt(norm), R1 / np.sqrt(norm)
    L0, L1 = L0 / np.sqrt(norm), L1 / np.sqrt(norm)
    assert compute_norm(L0, L1, R0, R1) == pytest.approx(1.0)
    rho_pq = compute_rho_pq(L0, L1, R0, R1)
    rho_pqrs = compute_rho_pqrs(L0, L1, R0, R1)

    return h, u, rho_pq, rho_pqrs, o, v


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


def test_compute_wia_equals_compute_wia_cis(system):
    h, u, rho_pq, rho_pqrs, o, v = system

    w_ia_general = compute_wia(h, u, rho_pq, rho_pqrs, o, v)
    w_ia_cis = compute_wia_cis(h, u, rho_pq, rho_pqrs, o, v)

    np.testing.assert_allclose(
        w_ia_cis,
        w_ia_general,
        rtol=0.0,
        atol=1e-12,
        err_msg="compute_wia_cis does not match compute_wia for CIS densities",
    )
