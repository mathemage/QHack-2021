#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml
from pennylane import qaoa

# from matplotlib import pyplot as plt


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    # QHACK #
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    n_wires = graph.number_of_nodes()
    wires = range(n_wires)
    depth = 10

    # create a device,
    dev = qml.device("default.qubit", wires=n_wires)

    # set up the QAOA ansatz circuit
    # and measure the probabilities of that circuit using the given optimized parameters.
    cost_h, mixer_h = qaoa.max_independent_set(graph, constrained=True)

    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)

    def circuit(params, **kwargs):
        qml.layer(qaoa_layer, depth, params[0], params[1])

    # You will be provided with optimized parameters and will not need to optimize further.
    # cost_function = qml.ExpvalCost(circuit, cost_h, dev)
    # optimizer = qml.GradientDescentOptimizer()
    # # optimizer = qml.AdamOptimizer()
    # # steps = 20
    # # steps = 16
    # steps = 2
    # for i in range(steps):
    #     # params = optimizer.step(cost_function, params)
    #     params, prev_cost = optimizer.step_and_cost(cost_function, params)
    #     if i % 1 == 0:
    #         print(f"#{i}\t cost: {prev_cost}")

    # Your next step will be to analyze the probabilities
    # and determine the maximum independent set of the
    # graph.
    #
    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=wires)

    probs = probability_circuit(params[0], params[1])
    max_index = np.argmax(probs)
    # print(f"probs: {probs}")
    # max_prob = max(probs)
    # print(f"max_prob: {max_prob}")
    # print(f"max_index: {max_index}")

    # Return the maximum independent set as an ordered list of nodes.
    max_ind_set = []
    # print(wires)
    for w in wires:
        if max_index & (1 << (n_wires-w-1)):
            max_ind_set.append(w)
    # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)
