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
    one_hot_1 = np.zeros_like(gradient)
    one_hot_2 = np.zeros_like(hessian)
    memo = {}
    s = 1.0

    s *= 2
    denominator = 2 * np.sin(s)
    for wi in range(len(weights)):
        one_hot_1[wi] = s

        mi = wi + 1
        memo[mi] = circuit(weights + one_hot_1)
        memo[-mi] = circuit(weights - one_hot_1)

        gradient[wi] = (memo[mi] - memo[-mi]) / denominator
        one_hot_1[wi] = 0

    s /= 2
    denominator = 2 * np.sin(s)
    # off-diagonal TODO
    # for wi in range(len(weights)):
    #     for wj in range(wi + 1, len(weights)):
    #         # print(f"wi == {wi}, wi == {wj}")
    #         one_hot_2[wi] = one_hot_2[wj] = s
    #         hessian[wi, wj] = (circuit(weights + one_hot_2) - circuit(weights - one_hot_2)) / denominator
    #         hessian[wj, wi] = hessian[wi, wj]
    #         one_hot_2[wi] = one_hot_2[wj] = 0

    # diagonal
    f_n0 = circuit(weights)
    for wii in range(len(weights)):
        mii = wii + 1
        hessian[wii, wii] = (memo[mii] - f_n0 - (f_n0 - memo[-mii])) / denominator ** 2

    print(gradient)
    print()
    print(hessian)
    print()
    print(dev.num_executions)
    print()
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
