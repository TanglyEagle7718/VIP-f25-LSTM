from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt


def create_forget_block_vqc(num_features: int):
    """
    Creates the Variational Quantum Circuit (VQC) for the QLSTM forget block. (VQC 1)

    This circuit takes the current input (x_t) and the previous cell state (c_{t-1})
    as inputs, entangles them, and applies a variational transformation. The measurement
    of the cell state qubits determines the forget gate (f_t).

    Args:
        num_features: The number of features in the input data.

    Returns:
        A tuple containing:
        - QuantumCircuit: The VQC for the forget block.
        - ParameterVector: The parameters for the input data encoding (x_t).
        - ParameterVector: The trainable variational parameters (weights) for this block.
    """
    num_qubits = 2 * num_features

    # Input data parameters
    x_params = ParameterVector('x', length=num_features)
    theta_params = ParameterVector('θ_f', length=num_features * 2)

    qc = QuantumCircuit(num_qubits, num_features, name="VQC1_ForgetBlock")

    # --- 1. State Preparation / Data Encoding ---
    for i in range(num_features):
        qc.ry(x_params[i], i)
    
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
        
    return qc, x_params, theta_params


def create_input_block_vqc(num_features: int):
    """
    Creates the Variational Quantum Circuit (VQC) for the QLSTM input block. (VQC 2)

    This circuit takes the current input (x_t) and the previous hidden state (h_{t-1})
    as inputs, entangles them, and applies a variational transformation. The measurement
    of the hidden state qubits determines the input gate (i_t).

    Args:
        num_features: The number of features in the input data.

    Returns:
        A tuple containing:
        - QuantumCircuit: The VQC for the input block.
        - ParameterVector: The parameters for the input data encoding (x_t).
        - ParameterVector: The trainable variational parameters (weights) for this block.
    """
    num_qubits = 2 * num_features

    x_params = ParameterVector('x', length=num_features)
    
    theta_params = ParameterVector('θ_in', length=num_features * 2)
    
    qc = QuantumCircuit(num_qubits, num_features, name="VQC2_InputBlock")

    # --- 1. State Preparation / Data Encoding ---
    for i in range(num_features):
        qc.ry(x_params[i], i)

    qc.barrier()

    # --- 2. Variational Layer (same structure as VQC 1) ---
    for i in range(num_features):
        qc.cx(i, i + num_features)
    
    qc.barrier()

    for i in range(num_qubits):
        qc.ry(theta_params[i], i)

    qc.barrier()

    # --- 3. Final Entanglement (same structure as VQC 1) ---
    for i in range(num_features - 1):
        qc.cx(i + num_features, i + num_features + 1)
            
    return qc, x_params, theta_params


def run_simulation(circuit: QuantumCircuit, x_params: ParameterVector, theta_params: ParameterVector, input_data: list, weights: list):
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
    
    param_map = {}
    for i in range(len(input_data)):
        param_map[x_params[i]] = input_data[i]
    for i in range(len(weights)):
        param_map[theta_params[i]] = weights[i]
        
    bound_circuit = circuit.assign_parameters(param_map)

    qubits_to_measure = list(range(num_features, 2 * num_features)) 
    cbits_to_write    = list(range(num_features))
    bound_circuit.measure(qubits_to_measure, cbits_to_write)

    backend = AerSimulator()
    compiled_circuit = transpile(bound_circuit, backend)
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    return counts


if __name__ == '__main__':
    NUM_FEATURES = 2
    
    sample_input_data = [np.pi / 2, np.pi / 4] 

    # --- VQC 1: Forget Block ---
    forget_block_vqc, x_params_f, theta_params_f = create_forget_block_vqc(NUM_FEATURES)

    print("--- QLSTM Forget Block VQC (VQC 1) ---")
    try:
        display(forget_block_vqc.draw('mpl', style='iqx'))
    except Exception:
        print(forget_block_vqc.draw('text'))

    sample_weights_f = np.random.uniform(0, 2 * np.pi, len(theta_params_f)).tolist()

    print("\n--- Simulating VQC 1 ---")
    print(f"Sample Input (x_t): {sample_input_data}")
    print(f"Sample Weights (VQC 1): {[f'{w:.2f}' for w in sample_weights_f]}")

    counts_f = run_simulation(
        forget_block_vqc, 
        x_params_f, 
        theta_params_f, 
        sample_input_data, 
        sample_weights_f
    )

    print(f"VQC 1 Counts (Forget Gate f_t): {counts_f}\n")
    plot_histogram(counts_f, title='VQC 1 - Forget Block Output')
    plt.show()
    
    print("-" * 40)

    # --- VQC 2: Input Block ---
    input_block_vqc, x_params_i, theta_params_i = create_input_block_vqc(NUM_FEATURES)

    print("\n--- QLSTM Input Block VQC (VQC 2) ---")
    try:
        display(input_block_vqc.draw('mpl', style='iqx'))
    except Exception:
        print(input_block_vqc.draw('text'))

    sample_weights_i = np.random.uniform(0, 2 * np.pi, len(theta_params_i)).tolist()

    print("\n--- Simulating VQC 2 ---")
    print(f"Sample Input (x_t): {sample_input_data}")
    print(f"Sample Weights (VQC 2): {[f'{w:.2f}' for w in sample_weights_i]}")

    counts_i = run_simulation(
        input_block_vqc, 
        x_params_i, 
        theta_params_i, 
        sample_input_data, 
        sample_weights_i
    )

    print("\n--- VQC 2 Simulation Results ---")
    print("Measurement is on the 'hidden state' qubits.")
    print(f"VQC 2 Counts (Input Gate i_t): {counts_i}")
    
    plot_histogram(counts_i, title='VQC 2 - Input Block Output')
    plt.show()
