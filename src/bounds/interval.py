from ..utils.helpers import relu


def getBounds(params, input_bounds):
    """
    Get bounds on the pre-activation variables for a neural network with the given parameters.
    Parameters:
        params: list of tuples (Wi, bi) for the i = 1, ..., L layers in the network
        input_bounds: bounds on the input to the network such that input_bounds[0] <= h0 <= input_bounds[1]
        log: optional weights and biases logging
    Returns:
        bounds: list of tuples (MLi, MUi) for i = 1, ..., L such that (MLi <= Wi @ hi-1 + bi <= MUi)
    """
    L, U = input_bounds
    bounds = []

    for l, (W, b) in enumerate(params):
        ML = b + relu(W) @ L - relu(-W) @ U
        MU = b + relu(W) @ U - relu(-W) @ L
        # equivalently we could have used ML = b + W @ xm - np.abs(W) @ xr
        bounds.append((ML, MU))
        L, U = relu(ML), relu(MU)
    return [input_bounds] + bounds


def getMultiStepBounds(params, input_bounds, A, B, K):
    """
    Calculate the pre-activation bounds for the network with the given parameters over multiple time-steps of the linear
    system

        x = A x + B f_nn(x)

    using interval bound propagation.
    Parameters:
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer
        input_bounds: tuple (L, U) where L is the lower bound and U is the upper bound on the input
        A: system state dynamics
        B: system control dynamics
        K: number of time-steps
    Returns:
        bounds: list of list of bounds such that bounds[k][l] is the bounds on the l-th hidden variable at timestep k
    """

    pre_relu_bounds = []

    for k in range(K):
        # get interval bounds through the whole network
        pre_relu_bounds.append(getBounds(params, input_bounds))
        u_bounds = pre_relu_bounds[-1][-1]  # bounds of the last hidden layer of the last time step
        # update input bounds for the next time step
        lb = relu(A) @ input_bounds[0] - relu(-A) @ input_bounds[1] + relu(B) @ u_bounds[0] - relu(-B) @ u_bounds[1]
        ub = relu(A) @ input_bounds[1] - relu(-A) @ input_bounds[0] + relu(B) @ u_bounds[1] - relu(-B) @ u_bounds[0]
        input_bounds = (lb, ub)

    pre_relu_bounds.append([input_bounds])  # add bounds for the final state
    return pre_relu_bounds
