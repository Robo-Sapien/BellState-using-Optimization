import qiskit
from qiskit import Aer
from qiskit.visualization import plot_histogram
from qiskit.aqua.components.optimizers import AQGD
from qiskit.aqua.components.optimizers import COBYLA

import time
import numpy as np
import matplotlib.pyplot as plt

class BellState():
    '''
    This class assembles the necessary parametric Quantum Circuit & optimizes the
    parameters to produce the state |01> + |10> (both having 0.5 probablity).
    The resulting state is actually one of the 4 Bell States.
    The optimization algorithm used is COBYLA with a choice of the cost function
    between squared error loss & entropy loss. (Default: entropy loss)
    '''
    
    def __init__(self, number_of_qubits, initial_state_qubit, backend, shots, mode = "training"):
        '''
        Parameters:
            number_of_qubits       : The number of qubits in the Quantum Circuit
            initial_state_qubit    : Initial combined state for the circuit
            backend                : Backend to be used for simulation
            shots                  : Number of measurements per simulation iteration
            mode                   : Mode ("training"/"testing")
        '''
        
        #Define the circuit
        self.number_of_qubits = number_of_qubits
        self.circuit = qiskit.QuantumCircuit(self.number_of_qubits) #qiskit takes care of adding classical bits when measurements are performed
        
        #Initialise the backend & shots
        self.backend = backend
        self.shots = shots
        
        #Initialise the circuit with the input initial state
        self.initial_state_qubit = initial_state_qubit
        self.circuit.initialize(self.initial_state_qubit,list(range(self.number_of_qubits)))
        
        #Define the parameters for parametric gates
        self.theta_q0 = qiskit.circuit.Parameter('theta_q0')
        self.theta_q1 = qiskit.circuit.Parameter('theta_q1')
        
        #Define the structure of the Quantum Circuit
        self.circuit.ry(self.theta_q0, 0)
        self.circuit.ry(self.theta_q1, 1)
        self.circuit.cx(0,1)
        
        #Training Simulation is to be done via noise sampling
        if(mode == "training"):
            self.circuit.measure_all()
        
                                
    def optimize_params(self, n_iterations, tolerance):
        '''
        This function will take care of optimizing the parameters for the circuit.
        Algorithm being used: COBYLA
        
        Parameters:
            n_iterations  : Maximum number of iterations for the optimizer
            tolerance     : Consecutive loss tolerance value for convergence check
            
        Returns:
            List of optimized parameters, final loss value, iterations performed 
        '''
        
        cobyla_optimizer = COBYLA(maxiter = n_iterations, disp = True, tol = tolerance)
    
        return cobyla_optimizer.optimize(num_vars = 2, 
                                         variable_bounds=[(0, np.pi), (0, np.pi)],
                                         objective_function = self.entropy_loss,
                                         initial_point = np.random.uniform(0, 2*np.pi, 2))
        
        
    def squared_error_loss(self, parameter_vector):
        '''
        This function takes care of calculating the squared error loss 
        for probabilities vector.
        
        loss = ||target_vector - simulated_vector|| (L2 norm)
        
        Parameters:
            parameter_vector  : List of parameters for which cost/loss is to be determined
            
        Returns:
            Cost/Loss value
        '''
        
        target_probabilities_vector = [0, 0.5, 0.5, 0] #corresponding states: ['00', '01', '10', '11']
        simulated_probabilities_vector, counts = self.get_probabilities_vector_and_counts(parameter_vector)
        
        error_as_difference = target_probabilities_vector - simulated_probabilities_vector
        cost_as_norm_error = np.inner(error_as_difference, error_as_difference)
        
        return cost_as_norm_error
    
    
    def entropy_loss(self, parameter_vector):
        '''
        This function takes care of calculating the multi state/class entropy loss 
        for probabilities vector.
        
        loss = -1 * summation(binary_entropy for all stateProbabilities)
        
        Parameters:
            parameter_vector  : List of parameters for which cost/loss is to be determined
            
        Returns:
            Cost/Loss value
        '''
        
        target_probabilities_vector = [0, 0.5, 0.5, 0] #corresponding states: ['00', '01', '10', '11']
        simulated_probabilities_vector, counts = self.get_probabilities_vector_and_counts(parameter_vector)
        
        simulated_probabilities = np.array(simulated_probabilities_vector)
        target_probabilities = 2 * np.array(target_probabilities_vector) #scaling factor = 2
        epsilon = 1e-6 #to avoid NAN in np.log()
        
        binary_entropy = (target_probabilities * np.log(simulated_probabilities + epsilon) 
                          + (1 - target_probabilities) * np.log((1 - simulated_probabilities) + epsilon))
        binary_entropy_loss = -1 * binary_entropy
        entropy_loss = np.sum(binary_entropy_loss)
        
        return entropy_loss
       
        
    def get_probabilities_vector_and_counts(self, parameter_vector):
        '''
        This function calculates the simulated probabilities vector & counts
        using the provided parameters.
        
        Parameters:
            parameter_vector  : List of parameters
            
        Returns
            Probabilities vector & counts obtained after simulation
        '''
        
        counts = self._simulate(parameter_vector).get_counts(self.circuit)

        probabilities_vector = np.zeros(2 ** self.number_of_qubits)
        states = ['00','01','10','11']

        for i, state in enumerate(states):
            if(state in counts):
                probabilities_vector[i] = counts[state]/self.shots

        return probabilities_vector, counts

    
    def _prepare_parameter_bindings(self, parameter_vector):
        '''
        This helper function prepares the parameter bindings for running the simulation.
        '''
        
        parameter_binds = {}
        parameter_binds[self.theta_q0] = parameter_vector[0]
        parameter_binds[self.theta_q1] = parameter_vector[1]
        
        return parameter_binds
    
    
    def _simulate(self, parameter_vector):
        '''
        This helper function runs the simulation for the input parameters.
        '''
        
        parameter_binds = self._prepare_parameter_bindings(parameter_vector)
        job = qiskit.execute(self.circuit, self.backend, shots = self.shots, parameter_binds = [parameter_binds])
        
        return job.result()
    
    
    def get_result(self, optimizer_result):
        '''
        This function extracts meaningful result data from the list returned by optimizer
        
        Parameters:
            optimizer_result  : List returned by optimizer
            
        Returns:
            optimized counts, optimized params & final loss value (After simulation results)
        '''
        
        optimized_parameter_vector = optimizer_result[0]
        final_loss = optimizer_result[1]
        probabilities_vector, counts = self.get_probabilities_vector_and_counts(optimized_parameter_vector)
        
        return counts, optimized_parameter_vector, final_loss
    
    
if __name__=="__main__":
    
    #Setting up the circuit hyperparameters
    number_of_qubits = 2
    initial_state_qubit = [0] * (2 ** number_of_qubits) 
    initial_state_qubit[0] = 1 #Initial state = |00>
    
    backend=Aer.get_backend("qasm_simulator")
    shots = 100000
    
    #Creating the custom circuit object
    custom_circuit = BellState(number_of_qubits, initial_state_qubit, backend, shots)
    visual_circuit = custom_circuit.circuit.draw(output = 'mpl')
    
    #Training hyperparameters
    max_iteration = 100
    tolerance = 1e-6
    
    print("Training starts")
    start_time = time.time()
    
    result = custom_circuit.optimize_params(max_iteration, tolerance)
    
    end_time = time.time()
    print("Training ends\nTraining Time: {}".format(end_time - start_time))
    
    
    #Post training results
    result_counts, result_params, final_loss = custom_circuit.get_result(result)
    histogram = plot_histogram(result_counts)
    
    #Checking final statevector
    backend = Aer.get_backend("statevector_simulator")
    custom_circuit = BellState(number_of_qubits, initial_state_qubit, backend, shots, mode = "testing")
    final_statevector = custom_circuit._simulate(result_params).get_statevector(custom_circuit.circuit)
    
    print("Optimized paramters: {}\nFinal Statevector: {}\nFinal loss: {}".format(result_params, 
                                                                                   final_statevector, 
                                                                                   final_loss))
    