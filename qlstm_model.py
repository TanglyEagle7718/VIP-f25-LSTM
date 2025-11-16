import numpy as np
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp

# Import VQC creation functions from local files
from forget_vqc import create_forget_block_vqc
from input_vqc import create_input_block_vqc
from update_vqc import create_update_block_vqc
from output_vqc import create_output_block_vqc
from hidden_vqc import create_hidden_block_vqc
from output_processing import create_output_processing_vqc

# Import utility functions
from vqc_utils import run_vqc, counts_to_expectation

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


class QLSTM_Cell:
    """
    Represents a single time-step cell of the Quantum LSTM.
    """
    def __init__(self, num_features: int):
        self.num_features = num_features

        # --- Instantiate all 6 VQCs ---
        self.vqc1, self.x_params1, self.theta1 = create_forget_block_vqc(num_features)
        self.vqc2, self.x_params2, self.theta2 = create_input_block_vqc(num_features)
        self.vqc3, self.x_params3, self.theta3 = create_update_block_vqc(num_features)
        self.vqc4, self.x_params4, self.theta4 = create_output_block_vqc(num_features)
        self.vqc5, self.c_params5, self.theta5 = create_hidden_block_vqc(num_features)
        self.vqc6, self.h_params6, self.theta6 = create_output_processing_vqc(num_features)

        # --- Concatenate all trainable parameters ---
        self.thetas = list(self.theta1) + list(self.theta2) + list(self.theta3) + \
                      list(self.theta4) + list(self.theta5) + list(self.theta6)
        
        self.num_weights = len(self.thetas)
        
        # --- Initialize hidden and cell states ---
        self.h_t = np.zeros(num_features)
        self.c_t = np.zeros(num_features)

    def _execute_vqc(self, vqc, input_params, weight_params, input_data, weights):
        """Helper to run a VQC and get expectation values."""
        counts = run_vqc(vqc, input_params, weight_params, input_data, weights)
        return counts_to_expectation(counts, self.num_features)

    def forward(self, x_t: np.ndarray, weights: np.ndarray):
        """
        Executes the forward pass for a single time step.

        Args:
            x_t: The input vector for the current time step.
            weights: The full vector of trainable weights for all 6 VQCs.

        Returns:
            The final output prediction y_t for the current time step.
        """
        # --- Split the weights for each VQC ---
        w1 = weights[0 : len(self.theta1)]
        w2 = weights[len(self.theta1) : len(self.theta1) + len(self.theta2)]
        w3 = weights[len(self.theta1) + len(self.theta2) : len(self.theta1) + len(self.theta2) + len(self.theta3)]
        w4 = weights[len(self.theta1) + len(self.theta2) + len(self.theta3) : len(self.theta1) + len(self.theta2) + len(self.theta3) + len(self.theta4)]
        w5 = weights[len(self.theta1) + len(self.theta2) + len(self.theta3) + len(self.theta4) : len(self.theta1) + len(self.theta2) + len(self.theta3) + len(self.theta4) + len(self.theta5)]
        w6 = weights[len(self.theta1) + len(self.theta2) + len(self.theta3) + len(self.theta4) + len(self.theta5) : self.num_weights]


        # Store previous states
        h_prev = self.h_t
        c_prev = self.c_t

        # 1. Forget Gate (f_t) - VQC1
        # Input: x_t, h_{t-1} (using h_prev on the second qubit register)
        f_t_exp = self._execute_vqc(self.vqc1, self.x_params1, self.theta1, x_t, w1)
        f_t = sigmoid(f_t_exp)

        # 2. Input Gate (i_t) - VQC2
        # Input: x_t, h_{t-1}
        i_t_exp = self._execute_vqc(self.vqc2, self.x_params2, self.theta2, x_t, w2)
        i_t = sigmoid(i_t_exp)

        # 3. Candidate Cell State (C_tilde_t) - VQC3
        # Input: x_t, h_{t-1}
        C_tilde_t_exp = self._execute_vqc(self.vqc3, self.x_params3, self.theta3, x_t, w3)
        C_tilde_t = tanh(C_tilde_t_exp)

        # 4. Update Cell State (c_t)
        self.c_t = f_t * c_prev + i_t * C_tilde_t

        # 5. Output Gate (o_t) - VQC4
        # Input: x_t, h_{t-1}
        o_t_exp = self._execute_vqc(self.vqc4, self.x_params4, self.theta4, x_t, w4)
        o_t = sigmoid(o_t_exp)

        # 6. Hidden State Processing (tanh(c_t) part) - VQC5
        # Input: c_t, h_{t-1}
        tanh_c_t_exp = self._execute_vqc(self.vqc5, self.c_params5, self.theta5, self.c_t, w5)
        tanh_c_t = tanh(tanh_c_t_exp)

        # 7. Update Hidden State (h_t)
        self.h_t = o_t * tanh_c_t
        
        # 8. Final Output Processing - VQC6
        # Input: h_t, h_{t-1}
        y_t_exp = self._execute_vqc(self.vqc6, self.h_params6, self.theta6, self.h_t, w6)
        y_t = tanh(y_t_exp) # Final prediction often uses tanh

        return y_t

import matplotlib.pyplot as plt
from scipy.optimize import minimize

class QLSTM_Model:
    """
    A full QLSTM model capable of handling sequences and training.
    """
    def __init__(self, num_features: int):
        self.cell = QLSTM_Cell(num_features)
        self.weights = np.random.uniform(0, 2 * np.pi, self.cell.num_weights)

    def forward_sequence(self, X_sequence):
        """
        Processes a sequence of inputs and returns the sequence of predictions.
        """
        self.cell.h_t = np.zeros(self.cell.num_features)
        self.cell.c_t = np.zeros(self.cell.num_features)
        
        predictions = []
        for x_t in X_sequence:
            y_t = self.cell.forward(np.array([x_t]), self.weights)
            predictions.append(y_t[0])
        return np.array(predictions)

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

def objective_function(weights, model, X, y):
    """Function for the optimizer to minimize."""
    model.weights = weights
    y_pred = model.forward_sequence(X)
    return model.loss(y, y_pred)

def main():
    """Main function to run the QLSTM training and evaluation."""
    # --- 1. Setup the Model and Data ---
    num_features = 1
    model = QLSTM_Model(num_features=num_features)
    
    # Generate sine wave data
    T = 50
    time_steps = np.linspace(0, T, T + 1)
    data = np.sin(time_steps)
    
    X = data[:-1]
    y = data[1:]

    print("--- QLSTM Model Training ---")
    print(f"Number of features: {num_features}")
    print(f"Total trainable weights: {model.cell.num_weights}")
    print(f"Training data shape: X={X.shape}, y={y.shape}")

    # --- 2. Training ---
    initial_weights = model.weights
    
    iteration_count = 0
    def callback(xk):
        nonlocal iteration_count
        iteration_count += 1
        model.weights = xk
        y_pred = model.forward_sequence(X)
        current_loss = model.loss(y, y_pred)
        print(f"Iteration {iteration_count}: Loss: {current_loss:.4f}")

    print("\nStarting training...")
    res = minimize(
        objective_function, 
        initial_weights, 
        args=(model, X, y), 
        method='L-BFGS-B',
        options={'maxiter': 10, 'disp': True},
        callback=callback
    )

    print("\nTraining finished.")
    
    # --- 3. Store Final Weights and Evaluate ---
    final_weights = res.x
    model.weights = final_weights

    print("\n--- Evaluating Model ---")
    predictions = model.forward_sequence(X)
    final_loss = model.loss(y, predictions)
    print(f"Final Loss: {final_loss:.4f}")

    # --- 4. Plot Results ---
    plt.figure(figsize=(12, 6))
    plt.title("QLSTM Sine Wave Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.plot(time_steps[1:], y, label="True Data", color='blue', marker='o')
    plt.plot(time_steps[1:], predictions, label="QLSTM Predictions", color='orange', linestyle='--', marker='x')
    plt.legend()
    plt.grid(True)
    plt.savefig("qlstm_prediction.png")
    print("\nPrediction plot saved to 'qlstm_prediction.png'")

if __name__ == '__main__':
    main()
