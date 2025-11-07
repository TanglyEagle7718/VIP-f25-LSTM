from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

def measure_vqc:
    
    num_features = len(input_data)
    
    param_map = {}
    for i in range(len(input_data)):
        param_map[x_params[i]] = input_data[i]
    for i in range(len(weights)):
        param_map[theta_params[i]] = weights[i]
        
    bound_circuit = circuit.assign_parameters(param_map)

    qubits_to_measure = list(range(num_features, 2 * num_features))  # measure the "cell state" qubits
    cbits_to_write    = list(range(num_features))
    bound_circuit.measure(qubits_to_measure, cbits_to_write)
    
    return