from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt


def create_update_block_vqc(num_features: int):
    """
    Creates the Variational Quantum Circuit (VQC) for the QLSTM update block. (VQC 3)

    This circuit takes the current input (x_t) and the previous hidden state (h_{t-1})
    as inputs, entangles them, and applies a variational transformation to generate
    the candidate cell state (C_tilde). The measurement produces values that will be
    used to update the cell state.

    Args:
        num_features: The number of features in the input data.

    Returns:
        A tuple containing:
        - QuantumCircuit: The VQC for the update block.
        - ParameterVector: The parameters for the input data encoding (x_t).
        - ParameterVector: The trainable variational parameters (weights) for this block.
    """
    num_qubits = 2 * num_features

    # Input data parameters
    x_params = ParameterVector('x', length=num_features)
    
    # Trainable variational parameters
    theta_params = ParameterVector('θ_u', length=num_features * 2)
    
    qc = QuantumCircuit(num_qubits, num_features, name="VQC3_UpdateBlock")

    # --- 1. State Preparation / Data Encoding ---
    for i in range(num_features):
        qc.ry(x_params[i], i)
    
    qc.barrier()

    # --- 2. Variational Layer ---
    # Entangle input qubits with candidate cell state qubits
    for i in range(num_features):
        qc.cx(i, i + num_features)
    
    qc.barrier()

    # Apply parameterized rotations
    for i in range(num_qubits):
        qc.ry(theta_params[i], i)

    qc.barrier()

    # --- 3. Final Entanglement ---
    # Create entanglement among candidate cell state qubits
    for i in range(num_features - 1):
        qc.cx(i + num_features, i + num_features + 1)
        
    return qc, x_params, theta_params


def run_simulation(circuit: QuantumCircuit, x_params: ParameterVector, 
                   theta_params: ParameterVector, input_data: list, weights: list):
    """
    Binds parameters to the circuit and runs a simulation.
    
    Args:
        circuit: The quantum circuit to simulate.
        x_params: The parameter vector for input data.
        theta_params: The parameter vector for trainable weights.
        input_data: A list of classical input values.
        weights: A list of classical weight values for the variational part.

    Returns:
        A dictionary of measurement counts.
    """
    num_features = len(input_data)
    
    # Create parameter mapping
    param_map = {}
    for i in range(len(input_data)):
        param_map[x_params[i]] = input_data[i]
    for i in range(len(weights)):
        param_map[theta_params[i]] = weights[i]
        
    # Bind parameters to circuit
    bound_circuit = circuit.assign_parameters(param_map)

    # Measure the candidate cell state qubits
    qubits_to_measure = list(range(num_features, 2 * num_features)) 
    cbits_to_write = list(range(num_features))
    bound_circuit.measure(qubits_to_measure, cbits_to_write)

    # Run simulation
    backend = AerSimulator()
    compiled_circuit = transpile(bound_circuit, backend)
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    return counts


def extract_expectation_values(counts: dict, num_features: int) -> list:
    """
    Extracts expectation values from measurement counts.
    
    Args:
        counts: Dictionary of measurement counts from simulation.
        num_features: Number of features/qubits measured.
    
    Returns:
        List of expectation values for each qubit (in range [-1, 1]).
    """
    total_shots = sum(counts.values())
    expectations = [0.0] * num_features
    
    for bitstring, count in counts.items():
        # Bitstring is in reverse order (rightmost = qubit 0)
        for i in range(num_features):
            if bitstring[-(i+1)] == '1':
                expectations[i] -= count / total_shots
            else:
                expectations[i] += count / total_shots
    
    return expectations


if __name__ == '__main__':
    NUM_FEATURES = 2
    
    # Sample input data (encoded as rotation angles)
    sample_input_data = [np.pi / 2, np.pi / 4] 

    # --- VQC 3: Update Block ---
    update_block_vqc, x_params_u, theta_params_u = create_update_block_vqc(NUM_FEATURES)

    print("=" * 60)
    print("QLSTM Update Block VQC (VQC 3)")
    print("=" * 60)
    print("\nThis circuit generates the candidate cell state (C_tilde).")
    print("It will be combined with the forget and input gates to update")
    print("the cell state: c_t = f_t * c_{t-1} + i_t * C_tilde\n")
    
    try:
        update_block_vqc.draw('mpl', style='iqx')
        plt.show()
    except Exception:
        print(update_block_vqc.draw('text'))

    # Generate random weights for variational parameters
    sample_weights_u = np.random.uniform(0, 2 * np.pi, len(theta_params_u)).tolist()

    print("\n--- Simulating VQC 3 ---")
    print(f"Sample Input (x_t): {sample_input_data}")
    print(f"Sample Weights (VQC 3): {[f'{w:.2f}' for w in sample_weights_u]}")

    # Run simulation
    counts_u = run_simulation(
        update_block_vqc, 
        x_params_u, 
        theta_params_u, 
        sample_input_data, 
        sample_weights_u
    )

    print("\n--- VQC 3 Simulation Results ---")
    print("Measurement is on the 'candidate cell state' qubits.")
    print("These values represent the new information to potentially add")
    print("to the cell state.")
    print(f"VQC 3 Counts (Candidate Cell State C_tilde): {counts_u}")
    
    # Extract expectation values (these would be passed through tanh in full QLSTM)
    expectations = extract_expectation_values(counts_u, NUM_FEATURES)
    print(f"\nExpectation values (before tanh): {[f'{e:.3f}' for e in expectations]}")
    print(f"After tanh activation: {[f'{np.tanh(e):.3f}' for e in expectations]}")
    
    # Visualize results
    plot_histogram(counts_u, title='VQC 3 - Update Block Output (Candidate Cell State)')
    plt.show()