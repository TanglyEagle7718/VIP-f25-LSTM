from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
import numpy as np

def run_vqc(
    circuit: QuantumCircuit,
    input_params: ParameterVector,
    weight_params: ParameterVector,
    input_data: np.ndarray,
    weights: np.ndarray,
    shots: int = 1024
):
    """
    Binds parameters to a VQC and runs it on a simulator.

    Args:
        circuit: The VQC to execute.
        input_params: The ParameterVector for the input data.
        weight_params: The ParameterVector for the trainable weights.
        input_data: The classical input data vector.
        weights: The classical weights for the variational circuit.
        shots: The number of shots for the simulation.

    Returns:
        A dictionary of measurement counts.
    """
    param_map = {}
    for i, p in enumerate(input_params):
        param_map[p] = input_data[i]
    for i, p in enumerate(weight_params):
        param_map[p] = weights[i]

    bound_circuit = circuit.assign_parameters(param_map)

    # The measurement is assumed to be on the second half of the qubits.
    num_features = len(input_params)
    qubits_to_measure = list(range(num_features, 2 * num_features))
    cbits_to_write = list(range(num_features))
    bound_circuit.measure(qubits_to_measure, cbits_to_write)

    backend = AerSimulator(method='statevector', device='GPU')
    # Fallback to CPU if GPU is not available (optional, but good practice for robustness)
    # try:
    #     backend = AerSimulator(method='statevector', device='GPU')
    # except Exception:
    #     print("GPU not found or not supported, falling back to CPU simulator.")
    #     backend = AerSimulator(method='statevector', device='CPU')
    compiled_circuit = transpile(bound_circuit, backend)
    result = backend.run(compiled_circuit, shots=shots).result()
    
    return result.get_counts()

def counts_to_expectation(counts: dict, num_features: int) -> np.ndarray:
    """
    Converts measurement counts to a vector of Pauli-Z expectation values.

    Args:
        counts: A dictionary of measurement counts from a Qiskit simulation.
        num_features: The number of measured qubits.

    Returns:
        A numpy array of expectation values, one for each qubit.
    """
    total_shots = sum(counts.values())
    expectations = np.zeros(num_features)
    
    for bitstring, count in counts.items():
        # Qiskit bitstrings are little-endian, so reverse for easy indexing
        bitstring = bitstring[::-1]
        for i in range(num_features):
            if bitstring[i] == '0':
                # Eigenvalue of |0> is +1
                expectations[i] += count
            else:
                # Eigenvalue of |1> is -1
                expectations[i] -= count
    
    return expectations / total_shots
