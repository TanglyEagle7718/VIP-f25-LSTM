from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt


def create_output_processing_vqc(num_features: int):
    """
    Creates the Variational Quantum Circuit (VQC) for the QLSTM output processing block. (VQC 6)

    This circuit takes the new hidden state (h_t) as input, applies a
    variational transformation, and produces the final output (y_t).

    Args:
        num_features: The number of features in the input data.

    Returns:
        A tuple containing:
        - QuantumCircuit: The VQC for the output processing block.
        - ParameterVector: The parameters for the hidden state data encoding (h_t).
        - ParameterVector: The trainable variational parameters (weights) for this block.
    """
    num_qubits = 2 * num_features

    h_params = ParameterVector('h', length=num_features)
    theta_params = ParameterVector('θ_y', length=num_features * 2)

    qc = QuantumCircuit(num_qubits, num_features, name="VQC6_OutputProcBlock")

    # --- 1. State Preparation / Data Encoding ---
    # Encode the hidden state h_t onto the first set of qubits
    for i in range(num_features):
        qc.ry(h_params[i], i)

    qc.barrier()

    # --- 2. Variational Layer ---
    for i in range(num_features):
        qc.cx(i, i + num_features)

    qc.barrier()

    for i in range(num_qubits):
        qc.ry(theta_params[i], i)

    qc.barrier()

    # --- 3. Final Entanglement ---
    for i in range(num_features - 1):
        qc.cx(i + num_features, i + num_features + 1)

    return qc, h_params, theta_params


def run_simulation(circuit: QuantumCircuit, h_params: ParameterVector, theta_params: ParameterVector, hidden_state_data: list, weights: list):
    """
    Binds parameters to the circuit and runs a simulation.
    """
    num_features = len(hidden_state_data)

    param_map = {}
    for i in range(len(hidden_state_data)):
        param_map[h_params[i]] = hidden_state_data[i]
    for i in range(len(weights)):
        param_map[theta_params[i]] = weights[i]

    bound_circuit = circuit.assign_parameters(param_map)

    qubits_to_measure = list(range(num_features, 2 * num_features))
    cbits_to_write = list(range(num_features))
    bound_circuit.measure(qubits_to_measure, cbits_to_write)

    backend = AerSimulator()
    compiled_circuit = transpile(bound_circuit, backend)
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()

    return counts


if __name__ == '__main__':
    NUM_FEATURES = 2

    vqc, h_params, theta_params = create_output_processing_vqc(NUM_FEATURES)

    print("--- QLSTM Output Processing Block VQC (VQC 6) ---")
    try:
        display(vqc.draw('mpl', style='iqx'))
    except Exception:
        print(vqc.draw('text'))


    sample_hidden_state = [np.pi / 2, 3 * np.pi / 4]
    sample_weights = np.random.uniform(0, 2 * np.pi, len(theta_params)).tolist()

    print("\n--- Simulating VQC 6 ---")
    print(f"Sample Hidden State (h_t): {sample_hidden_state}")
    print(f"Sample Weights (VQC 6): {[f'{w:.2f}' for w in sample_weights]}")

    counts = run_simulation(
        vqc,
        h_params,
        theta_params,
        sample_hidden_state,
        sample_weights
    )

    print("\n--- VQC 6 Simulation Results ---")
    print("Measurement produces the final output vector (y_t).")
    print(f"VQC 6 Counts (Final Output y_t): {counts}")

    plot_histogram(counts, title='VQC 6 - Final Output Distribution')
    plt.show()
