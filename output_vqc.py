# output_vqc.py

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt


def create_output_block_vqc(num_features: int):
    """
    Creates the Variational Quantum Circuit (VQC) for the QLSTM Output Block (VQC3).

    Structure:
      - RY encode input x_t
      - CX entangle input -> output qubits
      - RY variational layer
      - CX-chain entanglement among output qubits

    Qubit layout (total = 2 * num_features):
        0 ... num_features-1            : input qubits
        num_features ... 2*num_features : output qubits (measured)

    Measurement is handled separately in run_output_vqc_simulation().
    """

    num_qubits = 2 * num_features

    x_params = ParameterVector("x", num_features)
    theta_params = ParameterVector("θ_out", num_qubits)

    qc = QuantumCircuit(num_qubits, num_features, name="VQC3_OutputBlock")

    # -------------------------------------------------------
    # 1. Input Encoding: RY rotations on first half of qubits
    # -------------------------------------------------------
    for i in range(num_features):
        qc.ry(x_params[i], i)

    qc.barrier()

    # -------------------------------------------------------
    # 2. Entangle input → output qubits
    # -------------------------------------------------------
    for i in range(num_features):
        qc.cx(i, i + num_features)

    qc.barrier()

    # -------------------------------------------------------
    # 3. Variational RY layer on all qubits
    # -------------------------------------------------------
    for i in range(num_qubits):
        qc.ry(theta_params[i], i)

    qc.barrier()

    # -------------------------------------------------------
    # 4. Final CX-chain among output qubits
    # -------------------------------------------------------
    for i in range(num_features - 1):
        qc.cx(i + num_features, i + num_features + 1)

    return qc, x_params, theta_params



def run_output_vqc_simulation(
    circuit: QuantumCircuit,
    x_params: ParameterVector,
    theta_params: ParameterVector,
    x_values: list,
    weight_values: list,
    shots: int = 1024
):
    """
    Binds parameters to the Output VQC and runs measurement simulation.

    Measurements:
        - Measure the second half of the qubits (output qubits)
        - Write into classical bits [0 .. num_features-1]

    Returns:
        A dictionary of measurement counts.
    """

    num_features = len(x_values)

    # Parameter binding dictionary
    param_map = {}

    # Bind x input values
    for i in range(num_features):
        param_map[x_params[i]] = x_values[i]

    # Bind trainable parameters θ_out
    for i in range(len(theta_params)):
        param_map[theta_params[i]] = weight_values[i]

    # Create a bound circuit
    bound_circuit = circuit.assign_parameters(param_map)

    # Which qubits to measure: second half (output qubits)
    qubits_to_measure = list(range(num_features, 2 * num_features))
    classical_bits = list(range(num_features))

    bound_circuit.measure(qubits_to_measure, classical_bits)

    backend = AerSimulator()
    compiled = transpile(bound_circuit, backend)
    result = backend.run(compiled, shots=shots).result()

    return result.get_counts()



# ---------- Quick Demo ----------
if __name__ == "__main__":
    NUM_FEATURES = 2
    x_in = [np.pi/3, np.pi/5]  # example input values

    # Create VQC
    vqc, x_params, theta_params = create_output_block_vqc(NUM_FEATURES)

    # Random trainable weights
    weights = np.random.uniform(0, 2*np.pi, len(theta_params)).tolist()

    print("\n--- OUTPUT VQC CIRCUIT ---")
    try:
        display(vqc.draw("mpl", style="iqx"))
    except:
        print(vqc.draw("text"))

    print("\n--- Running Simulation ---")
    print("Input x_t:", x_in)
    print("θ_out weights:", [round(w, 3) for w in weights])

    counts = run_output_vqc_simulation(
        vqc,
        x_params,
        theta_params,
        x_values=x_in,
        weight_values=weights,
    )

    print("\nMeasurement Results (Output Gate o_t):")
    print(counts)

    plot_histogram(counts, title="VQC3 - Output Block")
    plt.show()

