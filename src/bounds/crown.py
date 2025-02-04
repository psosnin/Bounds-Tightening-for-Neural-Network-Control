import numpy as np

from ..utils.helpers import relu


def getBounds(params, input_bounds):
    """
    Compute intermediate bounds on the network using the CROWN algorithm given
    the input domain L <= x <= U.
    Parameters:
        params: list of tuples (Wi, bi) for the i = 1, ..., L layers in the network
        input_bounds: tuple if vectors (L, U) such that L <= x <= U
        log: optional weights and biases logging
    """
    W, b = zip(*params)
    # initialise containers for the bounds
    lower_bounds, upper_bounds = [], []

    # coefficients for the upper and lower linear relaxation of each neuron (0 index not used)
    alpha_l, alpha_u, beta_l, beta_u = [None], [None], [None], [None]

    # lower and upper bound of the input
    x0_l, x0_u = input_bounds

    # the first layer bounds are the same as interval arithmetic
    lower_bounds.append(b[0] + relu(W[0]) @ x0_l - relu(-W[0]) @ x0_u)
    upper_bounds.append(b[0] + relu(W[0]) @ x0_u - relu(-W[0]) @ x0_l)

    # for each subsequent layer, apply the CROWN bounds
    for m in range(2, len(W) + 1):
        # get bounds from the previous layer
        lm_minus_1, um_minus_1 = lower_bounds[m - 2], upper_bounds[m - 2]

        # compute partitions of the neurons in the previous layer
        I_pos, I = np.where(lm_minus_1 >= 0), np.where((lm_minus_1 <= 0) & (um_minus_1 >= 0))

        # initialise coefficients of linear relaxations of each neuron in the previous layer
        alpha_u.append(np.zeros_like(lm_minus_1))
        alpha_l.append(np.zeros_like(lm_minus_1))
        beta_u.append(np.zeros_like(lm_minus_1))
        beta_l.append(np.zeros_like(lm_minus_1))

        # compute coefficients of the linear relaxations of each neuron in the previous layer
        alpha_u[m-1][I] = (um_minus_1 / (um_minus_1 - lm_minus_1))[I]
        beta_u[m-1][I] = - lm_minus_1[I]
        alpha_u[m-1][I_pos] = 1
        alpha_l[m-1][I_pos] = 1

        # the slope of the lower bound can be any number in [0, 1] -> fastlin uses u / (u - l)
        # here we adaptively choose either 0 or 1 to minimise the area of the relaxation
        # alpha_l[m-1][I] = (um_minus_1 >= np.abs(lm_minus_1)).astype(int)
        alpha_l[m-1][I] = 0  # somehow 0 lower bound slope is tighter than adaptive?

        # initialise containers for the equivalent weights and biases for each previous layer
        # recomputing these at every layer isn't the most efficient but it's easier to understand
        Lambda = [np.zeros((b[m-1].size, Wk.shape[1])) for Wk in W[:m]] + [np.eye(b[m-1].size)]
        Omega = [np.zeros((b[m-1].size, Wk.shape[1])) for Wk in W[:m]] + [np.eye(b[m-1].size)]
        lambda_ = [None] + [np.zeros((b[m-1].size, Wk.shape[1])) for Wk in W[1:m]]  # lambda0 is not used
        omega = [None] + [np.ones((b[m-1].size, Wk.shape[1])) for Wk in W[1:m]]  # omega0 is not used
        Delta = [None] + [np.zeros((Wk.shape[0], b[m-1].size)) for Wk in W[:m]]  # Delta0 is not used
        Theta = [None] + [np.zeros((Wk.shape[0], b[m-1].size)) for Wk in W[:m]]  # Theta0 is not used

        # compute the values of the equivalent weights and biases for each previous layer
        for k in range(m, 1, -1):  # k = 3, 2
            for j in range(b[m-1].size):
                for i in range(b[k-2].size):
                    if Lambda[k][j, :] @ W[k-1][:, i] >= 0:
                        lambda_[k-1][j, i] = alpha_u[k-1][i]
                        Delta[k-1][i, j] = beta_u[k-1][i]
                    else:
                        lambda_[k-1][j, i] = alpha_l[k-1][i]
                        Delta[k-1][i, j] = beta_l[k-1][i]
                    if Omega[k][j, :] @ W[k-1][:, i] >= 0:
                        omega[k-1][j, i] = alpha_l[k-1][i]
                        Theta[k-1][i, j] = beta_l[k-1][i]
                    else:
                        omega[k-1][j, i] = alpha_u[k-1][i]
                        Theta[k-1][i, j] = beta_u[k-1][i]

            Lambda[k-1] = (Lambda[k] @ W[k-1]) * lambda_[k-1]
            Omega[k-1] = (Omega[k] @ W[k-1]) * omega[k-1]

        # compute Lambda0 and Omega0 (we dont need lambda0 and omega0 for the first layer)
        Lambda[0] = (Lambda[1] @ W[0])
        Omega[0] = (Omega[1] @ W[0])

        # compute upper bound
        gamma_u = relu(Lambda[0]) @ x0_u - relu(-Lambda[0]) @ x0_l
        gamma_u += sum([Lambda[k + 1] @ b[k] for k in range(m)])
        gamma_u += sum([(Lambda[k + 1] * Delta[k + 1].T).sum(-1) for k in range(m)])
        upper_bounds.append(gamma_u)

        # compute lower bound
        gamma_l = relu(Omega[0]) @ x0_l - relu(-Omega[0]) @ x0_u
        gamma_l += sum([Omega[k + 1] @ b[k] for k in range(m)])
        gamma_l += sum([(Omega[k + 1] * Theta[k + 1].T).sum(-1) for k in range(m)])
        lower_bounds.append(gamma_l)

    return [input_bounds] + list(zip(lower_bounds, upper_bounds))
