import numpy as np
from opt_einsum import contract
from scipy.sparse.linalg import LinearOperator, bicgstab
from utils import Counter


def CI_step(R, e_ref, f, u, o, v, dt, atol=1e-10):

    dim_cisd = len(R.ravel())

    Z = R - 0.5 * dt * Hc(R, e_ref, f, u, o, v)
    Ap_lambda = (
        lambda R_in: R_in.ravel()
        + dt / 2 * Hc(R_in, e_ref=e_ref, f=f, u=u, o=o, v=v).ravel()
    )

    Ap_linear = LinearOperator((dim_cisd, dim_cisd), matvec=Ap_lambda)
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


def compute_wia_cisd(h, u, rho_pq, rho_pqrs, o, v):
    """
    CISD-specific version of compute_wia. Exploits the block-sparsity of the
    CISD density matrices: only the non-zero blocks of rho_pqrs are contracted.

    The general expression is (for any CI truncation level)
      w^i_a = h^p_a rho^i_p - h^i_q rho^q_a
            + u^{pq}_{ra} rho^{i}_{pq} - u^{iq}_{rs} rho^{rs}_{aq}

    Non-zero rho_pqrs blocks used for the two-body terms (on top of those
    already present in CIS):
      two-body term 1 (second index in o): additionally [o,o,v,v], [v,o,v,v]
      two-body term 2 (third index in v):  additionally [o,o,v,v], [v,o,v,v],
                                           [o,v,v,v], [v,v,v,o], [v,v,v,v]
    """

    n_occ = o.stop
    n_virt = v.stop - n_occ

    w_ia = np.zeros((n_occ, n_virt), dtype=h.dtype)

    # --- one-body terms ---
    w_ia += contract("pa, ip -> ia", h[:, v], rho_pq[o, :])
    w_ia -= contract("iq, qa -> ia", h[o, :], rho_pq[:, v])

    # --- two-body term 1: u[p,q,r,a] * rho_pqrs[r,i,p,q]  (second idx in o) ---
    # blocks shared with CIS
    w_ia += contract("jkla, lijk -> ia", u[o, o, o, v], rho_pqrs[o, o, o, o])
    w_ia += contract("jbla, lijb -> ia", u[o, v, o, v], rho_pqrs[o, o, o, v])
    w_ia += contract("bkla, libk -> ia", u[v, o, o, v], rho_pqrs[o, o, v, o])
    w_ia += contract("jkca, cijk -> ia", u[o, o, v, v], rho_pqrs[v, o, o, o])
    w_ia += contract("bkca, cibk -> ia", u[v, o, v, v], rho_pqrs[v, o, v, o])
    w_ia += contract("jbca, cijb -> ia", u[o, v, v, v], rho_pqrs[v, o, o, v])
    # blocks new in CISD
    w_ia += contract("bcla, libc -> ia", u[v, v, o, v], rho_pqrs[o, o, v, v])
    w_ia += contract("bcda, dibc -> ia", u[v, v, v, v], rho_pqrs[v, o, v, v])

    # --- two-body term 2: u[i,q,r,s] * rho_pqrs[r,s,a,q]  (third idx in v) ---
    # blocks shared with CIS
    w_ia -= contract("ijkl, klaj -> ia", u[o, o, o, o], rho_pqrs[o, o, v, o])
    w_ia -= contract("ijck, ckaj -> ia", u[o, o, v, o], rho_pqrs[v, o, v, o])
    w_ia -= contract("ijkb, kbaj -> ia", u[o, o, o, v], rho_pqrs[o, v, v, o])
    # blocks new in CISD
    w_ia -= contract("iblk, lkab -> ia", u[o, v, o, o], rho_pqrs[o, o, v, v])
    w_ia -= contract("ibck, ckab -> ia", u[o, v, v, o], rho_pqrs[v, o, v, v])
    w_ia -= contract("iclb, lbac -> ia", u[o, v, o, v], rho_pqrs[o, v, v, v])
    w_ia -= contract("ijcd, cdaj -> ia", u[o, o, v, v], rho_pqrs[v, v, v, o])
    w_ia -= contract("ibcd, cdab -> ia", u[o, v, v, v], rho_pqrs[v, v, v, v])

    return w_ia


def compute_L(R0, R1, R2):
    L0 = R0.conj()
    L1 = 2 * R1.T.conj()
    L2 = (4 * R2 - 2 * R2.swapaxes(2, 3)).transpose(2, 3, 0, 1).conj()
    return L0, L1, L2


def Hc(c, e_ref, f, u, o, v):
    n_occ = o.stop
    n_virt = v.stop - n_occ
    c_0 = c[0]
    c_ai = c[1 : n_virt * n_occ + 1].reshape((n_virt, n_occ))
    c_abij = c[n_virt * n_occ + 1 :].reshape((n_virt, n_virt, n_occ, n_occ))

    Hc_0 = np.array([sigma0(e_ref, f, u, c_0, c_ai, c_abij, o, v)])
    Hc_ai = sigma_ai(e_ref, f, u, c_0, c_ai, c_abij, o, v)
    Hc_abij = sigma_abij(e_ref, f, u, c_0, c_ai, c_abij, o, v)

    return np.concatenate((Hc_0, Hc_ai.ravel(), Hc_abij.ravel()))


def compute_norm(L0, L1, L2, R0, R1, R2):
    norm_psi = (
        L0 * R0
        + np.einsum("ia,ai->", L1, R1)
        + 0.25
        * (np.einsum("ijab,abij->", L2, R2) + np.einsum("jiba,abij->", L2, R2))
    )
    return norm_psi


def sigma0(E0, f, u, r0, r1, r2, o, v):
    """
    |psi> = r0|0> + sum_{ai} r^a_i|ai> + 0.5*\sum_{abij} r^{ab}_{ij}|abij>

    sigma_0 = <0|H|psi>
    """
    sigma0 = E0 * r0
    sigma0 += 2 * contract("ia, ai->", f[o, v], r1)
    sigma0 += 2 * contract("ijab, abij->", u[o, o, v, v], r2)
    sigma0 -= contract("ijba, abij->", u[o, o, v, v], r2)
    return sigma0


def sigma_ai(E0, f, u, r0, r1, r2, o, v):

    """
    sigma_ai = <ai|H|psi>
    """

    sigma_ai = E0 * r1
    sigma_ai += r0 * f[v, o]

    sigma_ai += contract("ab,bi->ai", f[v, v], r1)
    sigma_ai -= contract("ji, aj->ai", f[o, o], r1)

    sigma_ai += 2 * contract("ajib, bj->ai", u[v, o, o, v], r1)
    sigma_ai -= contract("ajbi, bj->ai", u[v, o, v, o], r1)

    sigma_ai += 2 * contract("jb, abij->ai", f[o, v], r2)
    sigma_ai -= contract("jb, abji->ai", f[o, v], r2)

    sigma_ai -= 2 * contract("jkib, abjk->ai", u[o, o, o, v], r2)
    sigma_ai += contract("jkbi, abjk->ai", u[o, o, v, o], r2)

    sigma_ai += 2 * contract("ajbc, bcij->ai", u[v, o, v, v], r2)
    sigma_ai -= contract("ajcb, bcij->ai", u[v, o, v, v], r2)

    return sigma_ai


def sigma_abij(E0, f, u, r0, r1, r2, o, v):
    """
    sigma_abij = <abij|H|psi>
    """
    sigma_abij = E0 * r2
    sigma_abij += r0 * u[v, v, o, o]
    sigma_abij += contract("abkl, klij->abij", r2, u[o, o, o, o])
    sigma_abij += contract("cdij, abcd->abij", r2, u[v, v, v, v])

    Pabij = contract("ai, bj -> abij", f[v, o], r1)
    Pabij -= contract("ak, bkji->abij", r1, u[v, o, o, o])
    Pabij += contract("ac, bcji->abij", f[v, v], r2)
    Pabij -= contract("kj, abik->abij", f[o, o], r2)
    Pabij += contract("cj, abic->abij", r1, u[v, v, o, v])

    Pabij += 2 * contract("acik, bkjc->abij", r2, u[v, o, o, v])
    Pabij -= contract("acik, bkcj->abij", r2, u[v, o, v, o])

    Pabij -= contract("acki, bkjc->abij", r2, u[v, o, o, v])
    Pabij -= contract("ackj, bkci->abij", r2, u[v, o, v, o])

    sigma_abij += Pabij
    sigma_abij += Pabij.swapaxes(0, 1).swapaxes(2, 3)

    return sigma_abij


def compute_rho_pq(L0, L1, L2, R0, R1, R2):

    """
    \rho^q_p = <0|L*E_{p,q}*R|0>
    """

    n_occ, n_virt = R1.shape[1], R1.shape[0]
    n_mo = n_occ + n_virt
    o, v = slice(0, n_occ), slice(n_occ, n_mo)

    rho_pq = np.zeros((n_mo, n_mo), dtype=R0.dtype)
    I_occ = np.eye(n_occ)

    rho_pq[o, o] += 2 * L0 * R0 * I_occ
    rho_pq[o, o] += contract("ij, klab, abkl -> ji", I_occ, L2, R2)
    rho_pq[o, o] -= contract("ja, ai -> ji", L1, R1)
    rho_pq[o, o] -= contract("kjab, baik -> ji", L2, R2)
    rho_pq[o, o] += 2 * contract("ij, ka, ak -> ji", I_occ, L1, R1)

    rho_pq[v, o] = 2 * L0 * R1
    rho_pq[v, o] += 2 * contract("jb, abij -> ai", L1, R2)
    rho_pq[v, o] -= contract("jb, abji -> ai", L1, R2)

    rho_pq[o, v] += R0 * L1
    rho_pq[o, v] += contract("bj, ijab -> ia", R1, L2)

    rho_pq[v, v] += contract("ijac, bcij -> ba", L2, R2)
    rho_pq[v, v] += contract("ia, bi -> ba", L1, R1)

    return rho_pq


def compute_rho_pqrs(L0, L1, L2, R0, R1, R2):
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
    rho_pqrs[o, o, o, o] += contract("il, mkab, abmj -> klij", I_occ, L2, R2)
    rho_pqrs[o, o, o, o] += contract("jk, mlab, baim -> klij", I_occ, L2, R2)
    rho_pqrs[o, o, o, o] -= 2 * contract(
        "jl, mkab, baim -> klij", I_occ, L2, R2
    )
    rho_pqrs[o, o, o, o] -= 2 * contract(
        "ik, mlab, abmj -> klij", I_occ, L2, R2
    )
    rho_pqrs[o, o, o, o] += contract("il, ka, aj -> klij", I_occ, L1, R1)
    rho_pqrs[o, o, o, o] += contract("jk, la, ai -> klij", I_occ, L1, R1)
    rho_pqrs[o, o, o, o] -= 2 * contract("jl, ka, ai -> klij", I_occ, L1, R1)
    rho_pqrs[o, o, o, o] -= 2 * contract("ik, la, aj -> klij", I_occ, L1, R1)
    rho_pqrs[o, o, o, o] -= contract(
        "il, jk, mnab, abmn -> klij", I_occ, I_occ, L2, R2
    )
    rho_pqrs[o, o, o, o] += 2 * contract(
        "ik, jl, mnab, abmn -> klij", I_occ, I_occ, L2, R2
    )
    rho_pqrs[o, o, o, o] += contract("klab, abij -> klij", L2, R2)
    rho_pqrs[o, o, o, o] -= 2 * contract(
        "il, jk, ma, am -> klij", I_occ, I_occ, L1, R1
    )
    rho_pqrs[o, o, o, o] += 4 * contract(
        "ik, jl, ma, am -> klij", I_occ, I_occ, L1, R1
    )

    rho_pqrs[v, o, o, o] -= 2 * L0 * np.einsum("ik, aj -> akij", I_occ, R1)
    rho_pqrs[v, o, o, o] += 4 * L0 * np.einsum("jk, ai -> akij", I_occ, R1)
    rho_pqrs[v, o, o, o] += contract("kb, abji -> akij", L1, R2)
    rho_pqrs[v, o, o, o] -= 2 * contract("kb, abij -> akij", L1, R2)
    rho_pqrs[v, o, o, o] += contract("ik, mb, abmj -> akij", I_occ, L1, R2)
    rho_pqrs[v, o, o, o] -= 2 * contract("jk, mb, abmi -> akij", I_occ, L1, R2)
    rho_pqrs[v, o, o, o] -= 2 * contract("ik, mb, abjm -> akij", I_occ, L1, R2)
    rho_pqrs[v, o, o, o] += 4 * contract("jk, mb, abim -> akij", I_occ, L1, R2)

    rho_pqrs[o, v, o, o] = rho_pqrs[v, o, o, o].transpose(1, 0, 3, 2)

    rho_pqrs[o, o, v, o] -= contract("bi, jkab -> jkai", R1, L2)
    rho_pqrs[o, o, v, o] -= R0 * np.einsum("ij, ka -> jkai", I_occ, L1)
    rho_pqrs[o, o, v, o] += 2 * R0 * np.einsum("ik, ja -> jkai", I_occ, L1)
    rho_pqrs[o, o, v, o] -= np.einsum("ij, bm, mkba -> jkai", I_occ, R1, L2)
    rho_pqrs[o, o, v, o] += 2 * np.einsum("ik, bm, mjba -> jkai", I_occ, R1, L2)

    rho_pqrs[o, o, o, v] = rho_pqrs[o, o, v, o].transpose(1, 0, 3, 2)

    rho_pqrs[v, v, o, o] -= 2 * L0 * R2.transpose(0, 1, 3, 2)
    rho_pqrs[v, v, o, o] += 4 * L0 * R2

    rho_pqrs[o, o, v, v] = R0 * L2

    rho_pqrs[v, o, v, o] -= contract("ja, bi -> bjai", L1, R1)
    rho_pqrs[v, o, v, o] += 2 * contract(
        "ij, kmac, bckm -> bjai", I_occ, L2, R2
    )
    rho_pqrs[v, o, v, o] += 2 * contract("ij, ka, bk -> bjai", I_occ, L1, R1)
    rho_pqrs[v, o, v, o] -= contract("kjac, bcki -> bjai", L2, R2)
    rho_pqrs[v, o, v, o] -= contract("kjca, bcik -> bjai", L2, R2)

    rho_pqrs[o, v, o, v] = rho_pqrs[v, o, v, o].transpose(1, 0, 3, 2)

    rho_pqrs[v, o, o, v] += 2 * contract("ja, bi -> bjia", L1, R1)
    rho_pqrs[v, o, o, v] -= contract("ij, kmac, bckm -> bjia", I_occ, L2, R2)
    rho_pqrs[v, o, o, v] -= contract("ij, ka, bk -> bjia", I_occ, L1, R1)
    rho_pqrs[v, o, o, v] -= contract("kjca, bcki -> bjia", L2, R2)
    rho_pqrs[v, o, o, v] += 2 * contract("kjca, bcik -> bjia", L2, R2)

    rho_pqrs[o, v, v, o] = rho_pqrs[v, o, o, v].transpose(1, 0, 3, 2)

    rho_pqrs[v, v, v, o] -= contract("ja, bcij -> bcai", L1, R2)
    rho_pqrs[v, v, v, o] += 2 * contract("ja, bcji -> bcai", L1, R2)

    rho_pqrs[v, v, o, v] = rho_pqrs[v, v, v, o].transpose(1, 0, 3, 2)

    rho_pqrs[v, o, v, v] = contract("cj, ijba -> ciab", R1, L2)

    rho_pqrs[o, v, v, v] = rho_pqrs[v, o, v, v].transpose(1, 0, 3, 2)

    rho_pqrs[v, v, v, v] = contract("ijab, cdij -> cdab", L2, R2)

    return rho_pqrs
