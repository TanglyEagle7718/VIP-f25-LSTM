from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt


def create_hidden_block_vqc(num_features: int):
    """
    Creates the Variational Quantum Circuit (VQC) for the QLSTM hidden state processing block. (VQC 5)

    This circuit takes the updated cell state (c_t) and the previous hidden state (h_{t-1})
    as inputs, entangles them, and applies a variational transformation. The measurement
    of the hidden state qubits determines the new hidden state (h_t).

    Args:
        num_features: The number of features in the input data.

    Returns:
        A tuple containing:
        - QuantumCircuit: The VQC for the hidden state processing block.
        - ParameterVector: The parameters for the cell state data encoding (c_t).
        - ParameterVector: The trainable variational parameters (weights) for this block.
    """
    num_qubits = 2 * num_features

    # Cell state data parameters
    c_params = ParameterVector('c', length=num_features)
    theta_params = ParameterVector('θ_h', length=num_features * 2)

    qc = QuantumCircuit(num_qubits, num_features, name="VQC5_HiddenStateBlock")

    # --- 1. State Preparation / Data Encoding ---
    # Encode the new cell state c_t onto the first set of qubits
    for i in range(num_features):
        qc.ry(c_params[i], i)

    qc.barrier()

    # --- 2. Variational Layer ---
    # Entangle cell state qubits with hidden state qubits
    for i in range(num_features):
        qc.cx(i, i + num_features)

    qc.barrier()

    # Apply variational rotations
    for i in range(num_qubits):
        qc.ry(theta_params[i], i)

    qc.barrier()

    # --- 3. Final Entanglement ---
    # Entangle the hidden state qubits
    for i in range(num_features - 1):
        qc.cx(i + num_features, i + num_features + 1)

    return qc, c_params, theta_params


def run_simulation(circuit: QuantumCircuit, c_params: ParameterVector, theta_params: ParameterVector, cell_state_data: list, weights: list):
    """
    Binds parameters to the circuit and runs a simulation.

    Args:
        circuit: The quantum circuit to simulate.
        c_params: The parameter vector for cell state data.
        theta_params: The parameter vector for trainable weights.
        cell_state_data: A list of classical cell state values.
        weights: A list of classical weight values for the variational part.

    Returns:
        A dictionary of measurement counts.
    """
    num_features = len(cell_state_data)

    param_map = {}
    for i in range(len(cell_state_data)):
        param_map[c_params[i]] = cell_state_data[i]
    for i in range(len(weights)):
        param_map[theta_params[i]] = weights[i]

    bound_circuit = circuit.assign_parameters(param_map)

    # Measure the hidden state qubits
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

    # --- VQC 5: Hidden State Processing Block ---
    hidden_block_vqc, c_params, theta_params = create_hidden_block_vqc(NUM_FEATURES)

    print("--- QLSTM Hidden State Block VQC (VQC 5) ---")
    try:
        display(hidden_block_vqc.draw('mpl', style='iqx'))
    except Exception:
        print(hidden_block_vqc.draw('text'))


    # Assume the new cell state c_t is some classical vector
    sample_cell_state_data = [np.pi / 3, np.pi / 5]

    sample_weights = np.random.uniform(0, 2 * np.pi, len(theta_params)).tolist()

    print("\n--- Simulating VQC 5 ---")
    print(f"Sample Cell State (c_t): {sample_cell_state_data}")
    print(f"Sample Weights (VQC 5): {[f'{w:.2f}' for w in sample_weights]}")

    counts = run_simulation(
        hidden_block_vqc,
        c_params,
        theta_params,
        sample_cell_state_data,
        sample_weights
    )

    print("\n--- VQC 5 Simulation Results ---")
    print("Measurement is on the 'hidden state' qubits.")
    print(f"VQC 5 Counts (New Hidden State h_t): {counts}")

    plot_histogram(counts, title='VQC 5 - Hidden State Block Output')
    plt.show()
