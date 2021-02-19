#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
n_wires = 3
dev = qml.device("default.qubit", wires=n_wires)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    def variational_circuit_wire_list(params, wire_list=None):
        """A layered variational circuit composed of two parametrized layers of single qubit rotations
        interleaved with non-parameterized layers of fixed quantum gates specified by
        ``non_parametrized_layer``.

        The first parametrized layer uses the first three parameters of ``params``, while the second
        layer uses the final three parameters.
        """
        if wire_list is None:
            wire_list = [0, 1, 2]

        non_parametrized_layer()
        qml.RX(params[0], wires=wire_list[0])
        qml.RY(params[1], wires=wire_list[1])
        qml.RZ(params[2], wires=wire_list[2])
        non_parametrized_layer()
        qml.RX(params[3], wires=wire_list[0])
        qml.RY(params[4], wires=wire_list[1])
        qml.RZ(params[5], wires=wire_list[2])

    # FS metric TODO
    F = np.zeros([6, 6], dtype=np.float64)
    one = np.pi / 2
    dev_F = qml.device("default.qubit", wires=2 * n_wires)

    @qml.qnode(dev_F)
    def term(params, sign_term, sign_i, i, sign_j, j):
        qml.BasisState(np.array([0, 0, 0, 0, 0, 0]), wires=range(6))

        variational_circuit_wire_list(params)

        shift = np.zeros_like(params)
        shift[i] = sign_i * one
        shift[j] = sign_j * one
        variational_circuit_wire_list(params + shift, wire_list=[3, 4, 5])

        # return qml.DiagonalQubitUnitary([1] * 6, wires=[0, 1, 2]), qml.DiagonalQubitUnitary([1] * 6, wires=[3, 4, 5])
        # return [
        #     [qml.expval(qml.PauliX(wire)) for wire in [0, 1, 2]],
        #     [qml.expval(qml.PauliX(wire)) for wire in [3, 4, 5]],
        # ]
        return sign_term * np.abs(qml.dot(
            [qml.expval(qml.PauliX(wire)) for wire in [0, 1, 2]],
            [qml.expval(qml.PauliX(wire)) for wire in [3, 4, 5]],
        )) ** 2

    for i in range(len(params)):
        for j in range(len(params)):
            print(term(params, -1, 1, i, 1, j))
            # F[i, j] += term(params, -1, 1, i, 1, j)
            # F[i, j] += term(params, 1, 1, i, -1, j)
            # F[i, j] += term(params, 1, -1, i, 1, j)
            # F[i, j] += term(params, -1, -1, i, -1, j)
    F /= 8
    F_inverse = np.linalg.inv(F)

    # gradient
    gradient = np.zeros_like(params, dtype=np.float64)
    shift_mask = np.zeros_like(params)
    s = 1
    denominator = 2 * np.sin(s)
    for i in range(len(params)):
        shift_mask[i] = s
        gradient[i] = (qnode(params + shift_mask) - qnode(params - shift_mask)) / denominator
        shift_mask[i] = 0

    # from https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient.html
    # print(np.round(qml.metric_tensor(circuit)(params), 8))

    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    print(f"gradient: {gradient}")
    natural_grad = F_inverse * gradient
    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
