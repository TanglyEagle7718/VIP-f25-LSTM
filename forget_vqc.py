from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt


def create_forget_block_vqc(num_features: int):
    """
    Creates the Variational Quantum Circuit (VQC) for the QLSTM forget block.

    This circuit takes the current input and the previous cell state as inputs,
    entangles them, and applies a variational transformation. The measurement
    of the cell state qubits determines which information to "forget".

    Args:
        num_features: The number of features in the input data. This also
                      determines the number of qubits for the cell state.

    Returns:
        A tuple containing:
        - QuantumCircuit: The VQC for the forget block.
        - ParameterVector: The parameters for the input data encoding.
        - ParameterVector: The trainable variational parameters (weights).
    """
    num_qubits = 2 * num_features

    x_params = ParameterVector('x', length=num_features)
    theta_params = ParameterVector('θ', length=num_features * 2)

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

    qubits_to_measure = list(range(num_features, 2 * num_features))  # measure the "cell state" qubits
    cbits_to_write    = list(range(num_features))
    bound_circuit.measure(qubits_to_measure, cbits_to_write)

    #bound_circuit.measure(range(num_features, 2 * num_features), range(num_features))
    
    backend = AerSimulator()
    compiled_circuit = transpile(bound_circuit, backend)
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()



    
    return counts

if __name__ == '__main__':
    NUM_FEATURES = 2

    forget_block_vqc, x_params, theta_params = create_forget_block_vqc(NUM_FEATURES)

    print("--- QLSTM Forget Block VQC Architecture ---")
    try:
        forget_block_vqc.draw('mpl', style='iqx')
    except ImportError:
        print(forget_block_vqc.draw('text'))


    sample_input_data = [np.pi / 2, np.pi / 4] 

    sample_weights = np.random.uniform(0, 2 * np.pi, len(theta_params)).tolist()

    print("\n--- Simulating with Sample Data ---")
    print(f"Sample Input (encoded as angles): {sample_input_data}")
    print(f"Sample Weights (randomly generated): {[f'{w:.2f}' for w in sample_weights]}")

    measurement_counts = run_simulation(
        forget_block_vqc, 
        x_params, 
        theta_params, 
        sample_input_data, 
        sample_weights
    )

    print("\n--- Simulation Results ---")
    print("Measurement is on the 'cell state' qubits.")
    print("A '0' suggests forgetting, a '1' suggests keeping that feature's memory.")
    print(f"Counts: {measurement_counts}")
    
    plot_histogram(measurement_counts, title='Forget Block Output Distribution')
    plt.show()

