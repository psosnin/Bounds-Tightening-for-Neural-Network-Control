import numpy as np

from ..utils.helpers import relu


def getBounds(params, input_bounds):
    """
    Compute intermediate bounds on the network using the Fast-Lin algorithm given
    the input domain L <= x <= U.
    Parameters:
        params: list of tuples (Wi, bi) for the i = 1, ..., L layers in the network
        input_bounds: tuple if vectors (L, U) such that L <= x <= U
    """
    W, b = zip(*params)
    lower_bounds, upper_bounds = [], []
    A = [W[0]]  # A^0 matrix

    # lower and upper bound of the input
    x0_l, x0_u = input_bounds

    # the first layer bounds are the same as interval arithmetic
    lower_bounds.append(b[0] + relu(W[0]) @ x0_l - relu(-W[0]) @ x0_u)
    upper_bounds.append(b[0] + relu(W[0]) @ x0_u - relu(-W[0]) @ x0_l)

    # compute bounds for every subsequent layer using the fastlin algorithm
    for m in range(1, len(W)):
        # get bounds from the previous layer
        lm_minus_1, um_minus_1 = lower_bounds[m - 1], upper_bounds[m - 1]
        # construct the D^{m-1} activation matrix
        Dm_minus_1 = um_minus_1 / (um_minus_1 - lm_minus_1)
        Dm_minus_1[np.where(um_minus_1 <= 0)] = 0
        Dm_minus_1[np.where(lm_minus_1 >= 0)] = 1
        Dm_minus_1 = np.diag(Dm_minus_1)
        # construct the new A^{m-1} matrix and multiply all saved A^k matrices by the new one
        Am_minus_1 = W[m] @ Dm_minus_1
        A = [Am_minus_1 @ Ak for Ak in A] + [Am_minus_1]  # this step requires backpropagating the new A matrix
        # initialise containers for T^k and H^k matrices
        T = [np.zeros((bk.size, b[m].size)) for bk in b[:m]]
        H = [np.zeros((bk.size, b[m].size)) for bk in b[:m]]
        # compute bias terms T^k and H^k for each previous layer
        for k in range(m):
            lk, uk = lower_bounds[k], upper_bounds[k]  # bounds for this layer
            Ik = np.where((lk <= 0) & (uk >= 0))[0]  # partition of ambiguous neurons in this layer
            for r in Ik:
                for j in range(b[m].size):
                    if A[k + 1][j, r] > 0:
                        T[k][r, j] = lk[r]
                    else:
                        H[k][r, j] = lk[r]

        # compute new bounds for this layer
        nu = b[m] + sum([A[k + 1] @ b[k] for k in range(m)])
        mu_neg = - sum([(A[k + 1] * H[k].T).sum(1) for k in range(m)]) + relu(A[0]) @ x0_l - relu(-A[0]) @ x0_u
        mu_pos = - sum([(A[k + 1] * T[k].T).sum(1) for k in range(m)]) + relu(A[0]) @ x0_u - relu(-A[0]) @ x0_l

        lower_bounds.append(nu + mu_neg)
        upper_bounds.append(nu + mu_pos)

    return [input_bounds] + list(zip(lower_bounds, upper_bounds))
