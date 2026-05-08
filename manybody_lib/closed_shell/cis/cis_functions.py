import numpy as np
from opt_einsum import contract
from scipy.sparse.linalg import LinearOperator, bicgstab
from utils import Counter


def CI_step(R, e_ref, f, u, o, v, dt, atol=1e-10):

    dim_cis = len(R.ravel())

    Z = R - 0.5 * dt * Hc(R, e_ref, f, u, o, v)
    Ap_lambda = (
        lambda R_in: R_in.ravel()
        + dt / 2 * Hc(R_in, e_ref=e_ref, f=f, u=u, o=o, v=v).ravel()
    )

    Ap_linear = LinearOperator((dim_cis, dim_cis), matvec=Ap_lambda)
    local_counter = Counter()
    R_new, info = bicgstab(
        Ap_linear,
        Z.ravel(),
        x0=R.ravel(),
        rtol=0,
        atol=atol,
        callback=local_counter,
    )

    return R_new, local_counter.counter


def compute_wia(h, u, rho_pq, rho_pqrs, o, v):

    """
    In general (regardless of the CI truncation level)
    w^i_a = <\Psi|[H, E_{a,i}]|\Psi>
          = h^p_a \rho^i_p - h^i_q \rho^q_a + u^{pq}_{ra} \rho^{i}_{pq} - u^{iq}_{rs} \rho^{rs}_{aq}
    """

    n_occ = o.stop
    n_virt = v.stop - n_occ

    w_ia = np.zeros((n_occ, n_virt), dtype=h.dtype)

    w_ia += contract("pa, ip -> ia", h[:, v], rho_pq[o, :])
    w_ia -= contract("iq,qa->ia", h[o, :], rho_pq[:, v])

    w_ia += contract("pqra, ripq->ia", u[:, :, :, v], rho_pqrs[:, o, :, :])
    w_ia -= contract("iqrs, rsaq->ia", u[o, :, :, :], rho_pqrs[:, :, v, :])

    return w_ia


def compute_wia_cis(h, u, rho_pq, rho_pqrs, o, v):

    """
    CIS-specific version of compute_wia. Exploits the fact that in CIS the
    only non-zero blocks of rho_pqrs are those with at most one virtual index
    in the first or third position (no vv-type blocks).

    Non-zero rho_pqrs blocks used:
      [o,o,o,o], [o,o,o,v], [o,o,v,o], [v,o,o,o], [v,o,v,o], [v,o,o,v],
      [o,v,v,o]  (and symmetry-related blocks [o,v,o,o], [o,v,o,v])

    All four blocks of rho_pq are non-zero in CIS.
    """

    n_occ = o.stop
    n_virt = v.stop - n_occ

    w_ia = np.zeros((n_occ, n_virt), dtype=h.dtype)

    # --- one-body terms (unchanged; all rho_pq blocks non-zero in CIS) ---
    w_ia += contract("pa, ip -> ia", h[:, v], rho_pq[o, :])
    w_ia -= contract("iq, qa -> ia", h[o, :], rho_pq[:, v])

    # --- two-body term 1: u[p,q,r,a] * rho_pqrs[r,i,p,q] ---
    # split over the 6 non-zero (p,q,r) sub-blocks
    w_ia += contract("jkla, lijk -> ia", u[o, o, o, v], rho_pqrs[o, o, o, o])
    w_ia += contract("jbla, lijb -> ia", u[o, v, o, v], rho_pqrs[o, o, o, v])
    w_ia += contract("bkla, libk -> ia", u[v, o, o, v], rho_pqrs[o, o, v, o])
    w_ia += contract("jkca, cijk -> ia", u[o, o, v, v], rho_pqrs[v, o, o, o])
    w_ia += contract("bkca, cibk -> ia", u[v, o, v, v], rho_pqrs[v, o, v, o])
    w_ia += contract("jbca, cijb -> ia", u[o, v, v, v], rho_pqrs[v, o, o, v])

    # --- two-body term 2: u[i,q,r,s] * rho_pqrs[r,s,a,q] ---
    # split over the 3 non-zero (r,s,q) sub-blocks
    w_ia -= contract("ijkl, klaj -> ia", u[o, o, o, o], rho_pqrs[o, o, v, o])
    w_ia -= contract("ijck, ckaj -> ia", u[o, o, v, o], rho_pqrs[v, o, v, o])
    w_ia -= contract("ijkb, kbaj -> ia", u[o, o, o, v], rho_pqrs[o, v, v, o])

    return w_ia


def compute_L(R0, R1):
    L0 = R0.conj()
    L1 = 2 * R1.T.conj()
    return L0, L1


def Hc(c, e_ref, f, u, o, v):
    n_occ = o.stop
    n_virt = v.stop - n_occ
    c_0 = c[0]
    c_ai = c[1:].reshape((n_virt, n_occ))

    Hc_0 = np.array([sigma0(e_ref, f, u, c_0, c_ai, o, v)])
    Hc_ai = sigma_ai(e_ref, f, u, c_0, c_ai, o, v)

    return np.concatenate((Hc_0, Hc_ai.ravel()))


def compute_norm(L0, L1, R0, R1):
    norm_psi = L0 * R0 + np.einsum("ia,ai->", L1, R1)
    return norm_psi


def sigma0(E0, f, u, r0, r1, o, v):
    """
    |psi> = r0|0> + sum_{ai} r^a_i|ai>

    sigma_0 = <0|H|psi>
    """
    sigma0 = E0 * r0
    sigma0 += 2 * contract("ia, ai->", f[o, v], r1)
    return sigma0


def sigma_ai(E0, f, u, r0, r1, o, v):

    """
    sigma_ai = <ai|H|psi>
    """

    sigma_ai = E0 * r1
    sigma_ai += r0 * f[v, o]

    sigma_ai += contract("ab,bi->ai", f[v, v], r1)
    sigma_ai -= contract("ji, aj->ai", f[o, o], r1)

    sigma_ai += 2 * contract("ajib, bj->ai", u[v, o, o, v], r1)
    sigma_ai -= contract("ajbi, bj->ai", u[v, o, v, o], r1)

    return sigma_ai


def compute_rho_pq(L0, L1, R0, R1):

    """
    \rho^q_p = <0|L*E_{p,q}*R|0>
    """

    n_occ, n_virt = R1.shape[1], R1.shape[0]
    n_mo = n_occ + n_virt
    o, v = slice(0, n_occ), slice(n_occ, n_mo)

    rho_pq = np.zeros((n_mo, n_mo), dtype=R0.dtype)
    I_occ = np.eye(n_occ)

    rho_pq[o, o] += 2 * L0 * R0 * I_occ
    rho_pq[o, o] -= contract("ja, ai -> ji", L1, R1)
    rho_pq[o, o] += 2 * contract("ij, ka, ak -> ji", I_occ, L1, R1)

    rho_pq[v, o] = 2 * L0 * R1

    rho_pq[o, v] += R0 * L1

    rho_pq[v, v] += contract("ia, bi -> ba", L1, R1)

    return rho_pq


def compute_rho_pqrs(L0, L1, R0, R1):
    """
    \rho^{rs}_{pq} = <0|L*{e}_{p,q,r,s}*R|0>
    """

    n_occ, n_virt = R1.shape[1], R1.shape[0]
    n_mo = n_occ + n_virt
    o, v = slice(0, n_occ), slice(n_occ, n_mo)

    rho_pqrs = np.zeros((n_mo, n_mo, n_mo, n_mo), dtype=R0.dtype)
    I_occ = np.eye(n_occ)

    rho_pqrs[o, o, o, o] += 4 * L0 * R0 * np.einsum("ik,jl->klij", I_occ, I_occ)
    rho_pqrs[o, o, o, o] -= 2 * L0 * R0 * np.einsum("il,jk->klij", I_occ, I_occ)
    rho_pqrs[o, o, o, o] += contract("il, ka, aj -> klij", I_occ, L1, R1)
    rho_pqrs[o, o, o, o] += contract("jk, la, ai -> klij", I_occ, L1, R1)
    rho_pqrs[o, o, o, o] -= 2 * contract("jl, ka, ai -> klij", I_occ, L1, R1)
    rho_pqrs[o, o, o, o] -= 2 * contract("ik, la, aj -> klij", I_occ, L1, R1)
    rho_pqrs[o, o, o, o] -= 2 * contract(
        "il, jk, ma, am -> klij", I_occ, I_occ, L1, R1
    )
    rho_pqrs[o, o, o, o] += 4 * contract(
        "ik, jl, ma, am -> klij", I_occ, I_occ, L1, R1
    )

    rho_pqrs[v, o, o, o] -= 2 * L0 * np.einsum("ik, aj -> akij", I_occ, R1)
    rho_pqrs[v, o, o, o] += 4 * L0 * np.einsum("jk, ai -> akij", I_occ, R1)

    rho_pqrs[o, v, o, o] = rho_pqrs[v, o, o, o].transpose(1, 0, 3, 2)

    rho_pqrs[o, o, v, o] -= R0 * np.einsum("ij, ka -> jkai", I_occ, L1)
    rho_pqrs[o, o, v, o] += 2 * R0 * np.einsum("ik, ja -> jkai", I_occ, L1)

    rho_pqrs[o, o, o, v] = rho_pqrs[o, o, v, o].transpose(1, 0, 3, 2)

    rho_pqrs[v, o, v, o] -= contract("ja, bi -> bjai", L1, R1)
    rho_pqrs[v, o, v, o] += 2 * contract("ij, ka, bk -> bjai", I_occ, L1, R1)

    rho_pqrs[o, v, o, v] = rho_pqrs[v, o, v, o].transpose(1, 0, 3, 2)

    rho_pqrs[v, o, o, v] += 2 * contract("ja, bi -> bjia", L1, R1)
    rho_pqrs[v, o, o, v] -= contract("ij, ka, bk -> bjia", I_occ, L1, R1)

    rho_pqrs[o, v, v, o] = rho_pqrs[v, o, o, v].transpose(1, 0, 3, 2)

    return rho_pqrs
