import qiskit
from qiskit import Aer
from qiskit.visualization import plot_histogram
from qiskit.aqua.components.optimizers import AQGD
from qiskit.aqua.components.optimizers import COBYLA

import numpy as np
import matplotlib.pyplot as plt

class BellState():
    
    def __init__(self, number_of_qubits, initial_state_qubit, backend, shots, mode = "training"):
        
        self.number_of_qubits = number_of_qubits
        self.circuit = qiskit.QuantumCircuit(self.number_of_qubits) #qiskit takes care of adding classical bits when measurements are performed
        self.backend = backend
        self.shots = shots
        
        self.initial_state_qubit = initial_state_qubit
        self.circuit.initialize(self.initial_state_qubit,list(range(self.number_of_qubits)))
        
        self.theta_ry_q0 = qiskit.circuit.Parameter('theta_ry_q0')
        self.theta_ry_q1 = qiskit.circuit.Parameter('theta_ry_q1')
        
        self.circuit.ry(self.theta_ry_q0, 0)
        self.circuit.rx(self.theta_ry_q1, 1)
        self.circuit.cx(0,1)
        
        if(mode == "training"):
            self.circuit.measure_all()
        
                                
    def optimize_params(self, n_iterations, tolerance):
        
        cobyla_optimizer = COBYLA(maxiter = n_iterations, tol = tolerance, disp = True)
    
        return cobyla_optimizer.optimize(num_vars = 2, 
                                         variable_bounds=[(0, np.pi), (0, np.pi)],
                                         objective_function = self.entropy_loss,
                                         initial_point = np.random.uniform(0,2*np.pi,2))
        
        
    def squared_error_loss(self, parameter_vector):
        
        target_probabilities_vector = [0, 0.5, 0.5, 0]
        simulated_probabilities_vector, counts = self.get_probabilities_vector_and_counts(parameter_vector)
        
        error_as_difference = target_probabilities_vector - simulated_probabilities_vector
        cost_as_norm_error = np.inner(error_as_difference, error_as_difference)
        
        return cost_as_norm_error
    
    
    def entropy_loss(self, parameter_vector):
        
        target_probabilities_vector = [0, 0.5, 0.5, 0]
        simulated_probabilities_vector, counts = self.get_probabilities_vector_and_counts(parameter_vector)
        
        simulated_probabilities = np.array(simulated_probabilities_vector)
        target_probabilities = 2 * np.array(target_probabilities_vector) #scaling factor = 2
        epsilon = 1e-6
        
        binary_entropy = target_probabilities * np.log(simulated_probabilities + epsilon) + (1 - target_probabilities) * np.log((1 - simulated_probabilities) + epsilon)
        binary_entropy_loss = -1 * binary_entropy
        entropy_loss = np.sum(binary_entropy_loss)
        
        return entropy_loss
       
        
    def get_probabilities_vector_and_counts(self, parameter_vector):
        
        counts = self._simulate(parameter_vector).get_counts(self.circuit)

        probabilities_vector = np.zeros(2**self.number_of_qubits)
        states = ['00','01','10','11']

        i=0
        for state in states:
            if(state in counts):
                probabilities_vector[i] = counts[state]/self.shots
            i = i+1

        return probabilities_vector, counts

    
    def _prepare_parameter_bindings(self, parameter_vector):
        
        parameter_binds = {}
        parameter_binds[self.theta_ry_q0] = parameter_vector[0]
        parameter_binds[self.theta_ry_q1] = parameter_vector[1]
        
        return parameter_binds
    
    
    def _simulate(self, parameter_vector):
        
        parameter_binds = self._prepare_parameter_bindings(parameter_vector)
        job = qiskit.execute(self.circuit, self.backend, shots = self.shots, parameter_binds = [parameter_binds])
        
        return job.result()
    
    
    def get_state_vector(self, parameter_vector):
        return self._simulate(parameter_vector).get_statevector(self.circuit)
    
    def visualize_result(self, optimizer_result):
        
        optimized_parameter_vector = optimizer_result[0]
        final_loss = optimizer_result[1]
        probabilities_vector, counts = self.get_probabilities_vector_and_counts(optimized_parameter_vector)
        return counts, optimized_parameter_vector, final_loss
    
    def visualize_circuit(self):
        return self.circuit.draw()
    
    
if __name__=="__main__":
    
    #Set the circuit hyperparameters
    number_of_qubits = 2
    initial_state_qubit = [0]*(2**number_of_qubits) #Initial state = |00>
    initial_state_qubit[0] = 1
    
    
    backend=Aer.get_backend("qasm_simulator")
    shots = 100000
    
    circuit = BellState(number_of_qubits, initial_state_qubit, backend, shots)
    
    max_iteration = 100
    tolerance = 1e-6
    
    print("Training starts")
    result = circuit.optimize_params(max_iteration, tolerance)
    print("Training ends")
    
    result_counts, result_params, final_loss = circuit.visualize_result(result)
    histogram = plot_histogram(result_counts)
    print(result_params)
    
    #Checking final statevector
    backend = Aer.get_backend("statevector_simulator")
    circuit = BellState(number_of_qubits, initial_state_qubit, backend, shots, mode = "testing")
    
    print(circuit.get_state_vector(result_params))
    print(final_loss)