#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    one_hot = np.zeros_like(gradient)
    memo = {}
    s = 1.0

    s *= 2
    denominator = 2 * np.sin(s)
    for i in range(len(weights)):
        one_hot[i] = s

        mi = i + 1
        memo[mi] = circuit(weights + one_hot)
        memo[-mi] = circuit(weights - one_hot)

        gradient[i] = (memo[mi] - memo[-mi]) / denominator
        one_hot[i] = 0

    s /= 2
    denominator = 2 * np.sin(s)
    # off-diagonal
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            # print(f"i == {i}, i == {j}")
            for sign_j in [1, -1]:
                for sign_i in [1, -1]:
                    one_hot = np.zeros_like(gradient)
                    one_hot[i] = sign_i * s
                    one_hot[j] = sign_j * s
                    hessian[i, j] += sign_i * sign_j * circuit(weights + one_hot)
            hessian[i, j] /= denominator ** 2
            hessian[j, i] = hessian[i, j]

    # diagonal
    f_n0 = circuit(weights)
    for wii in range(len(weights)):
        mii = wii + 1
        hessian[wii, wii] = (memo[mii] - f_n0 - (f_n0 - memo[-mii])) / denominator ** 2

    # np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    # print(gradient)
    # print()
    # print(hessian)
    # print()
    # print(dev.num_executions)
    # print()
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
