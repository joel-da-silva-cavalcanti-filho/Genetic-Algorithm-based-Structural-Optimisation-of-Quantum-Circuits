import numpy as np
import random
import time
from math import sqrt
from torch import tensor, inner
from core.ansatz_simulation_class import AnsatzSimulation
from collections import Counter
import itertools
import torch
import math
from pennylane.math import trace_distance


## NIELSEN, Michael A. and CHUANG, Isaac L. for distance between quantum states

# Gotta test AnsatzMutation

class RepairAnsatz:
    def __init__(self, n_qubits, max_layers):
        self.n_qubits = n_qubits
        self.max_layers = max_layers
        
    def _do(self, individual):
        if len(individual) > self.max_layers:
            new_size = random.randint(2, self.max_layers)
            selected_indexes = np.random.randint(0, high=len(individual), size=new_size)
            return [individual[index] for index in selected_indexes]
        else:
            return individual


def change_cnot_gate_to_one_qubit_gates(circuit_layer: list, gate_options: list, selected_qubit: int) -> list:

    """
    This function turns the control and target qubits into one-qubit gates

    """

    gate = circuit_layer[selected_qubit]
    gate_index = gate[5]
    if gate[0:4] == 'ctrl':
        target_index = circuit_layer.index('trgt_' + gate_index)
        circuit_layer[target_index] = random.choice(gate_options)
    elif gate[0:4] == 'trgt':
        control_index = circuit_layer.index('ctrl_' + gate_index)
        circuit_layer[control_index] = random.choice(gate_options)
    
    circuit_layer[selected_qubit] = random.choice(gate_options)
    
    return circuit_layer

def swap_cnot_gate(circuit_layer: list, selected_qubit: int):
    """
    This function swaps the control and target qubits in 
    the CNOT gate

    """

    gate = circuit_layer[selected_qubit]
    if gate[0:4] == 'ctrl': # testar esse if
        cnot_value = gate[5]
        target_index = circuit_layer.index('trgt_'+ cnot_value)
        circuit_layer[selected_qubit], circuit_layer[target_index] =  circuit_layer[target_index], circuit_layer[selected_qubit] 
    elif gate[0:4] == 'trgt':
        cnot_value = gate[5]
        control_index = circuit_layer.index('ctrl_' + cnot_value)
        circuit_layer[selected_qubit], circuit_layer[control_index] =  circuit_layer[control_index], circuit_layer[selected_qubit]
    
    return circuit_layer


def drop_layers(circuit):
    n_layers = random.randint(int(math.floor(len(circuit)/2)) - 1, len(circuit))
    selected_indexes = np.random.randint(0, high=len(circuit), size=n_layers)
    
    return [circuit[index] for index in selected_indexes]


class AnsatzMutation:

    """
    This Mutation Class randomly swaps Ansatz layers, changes the  
    the gates placed in certain positions

    """
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate
        self.one_qubit_gates = ['pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard']
        #self.gate_options_with_cnot = ['pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard']

    # Make it drop out a few layers during mutation to guarantee that it will remain shallow and diverse somehow
    def place_a_single_gate(self):
        return random.choice(self.one_qubit_gates) 
    
    def generate_partially_entangled_layer(self, layer):
        pairs = [(wire,wire+1) for wire in range(len(layer))]
        #print(pairs)
        selected_pair = random.choice(pairs)
        control, target = selected_pair
        
        layer = ['ctrl_0' if wire == control else
                'trgt_0' if wire == target else
                random.choice(self.one_qubit_gates)
                for wire in range(len(layer))]
    
        return layer
    
    def shuffle_layers(self, circuit):
        return random.sample(circuit, k=len(circuit))

    def _do(self, offspring):
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = []
            random_value = np.random.random()
            if random_value < self.mutation_rate:
                choice = np.random.randint(0, 2)
                # Randomly make the circuit ansatz drop a few layers or shuffle them
                if choice == 0:
                    random_mutation_action = np.random.randint(0,2)
                    if random_mutation_action == 0:
                        mutated_offspring.append(drop_layers(individual))
                    elif random_mutation_action == 1:
                        mutated_offspring.append(self.shuffle_layers(individual))
                elif choice == 1:
                    # Or make it change entanglement patterns, turn CNOTs into one qubit gates, or simply change gates
                    for circuit_layer in individual:
                            random_qubit_position = np.random.randint(0, len(circuit_layer))
                            random_gate = circuit_layer[random_qubit_position]
                            circuit_layer[random_qubit_position]
                            if random_gate[0:4] == 'ctrl' or random_gate[0:4] == 'trgt': 
                                random_mutation_action = np.random.randint(0,2)
                                if random_mutation_action == 0:
                                    mutated_individual.append(swap_cnot_gate(circuit_layer, random_qubit_position))
                                elif random_mutation_action == 1:
                                    mutated_individual.append(change_cnot_gate_to_one_qubit_gates(circuit_layer, self.one_qubit_gates, random_qubit_position))
                            else:
                                new_layer = circuit_layer
                                random_mutation_action = np.random.randint(0,2)
                                if random_mutation_action == 0:
                                    new_layer[random_qubit_position] = random.choice(self.one_qubit_gates)
                                    mutated_individual.append(new_layer)
                                else:
                                    # Besides mutating entanglement patterns and changing cnots into one qubit gates, we could add partial and full entaglement by mutation in a few layers
                                    # this would synthetize more rich and diverse quantum circuits over the iterations :)
                                    mutated_individual.append(self.generate_partially_entangled_layer(new_layer))
                                
                    mutated_offspring.append(mutated_individual)        
            else:
                mutated_offspring.append(individual)           
                

        offspring_a, offspring_b = mutated_offspring
        return offspring_a, offspring_b
    
def select_parents(population: list):
    random_generator = np.random.default_rng(seed=42)
    parent_a, parent_b = random_generator.choice(a = np.array(population), size=2, replace=False)
    if parent_a.shape[1] < parent_b.shape[1]:
        parent_a, parent_b = parent_b, parent_a
    return parent_a, parent_b


# Gotta test AnsatzCrossover
class AnsatzCrossover:
    """"
    The crossover happens between chromosomes of possibly different lengths,
    then the offspring might inherit the size of one of its parents
    """

    def __init__(self, crossover_rate: float):
        self.crossover_rate = crossover_rate

    def _do(self, parent_a, parent_b):
        random_number = np.random.random()
        if random_number < self.crossover_rate:
            return [parent_a, parent_b]
        
        random_generator = np.random.default_rng(seed=42)

        # The parent b is always the smallest one, and the parent a, the biggest one

        chromosome_length_a = len(parent_b)
        chromosome_length_b = len(parent_a)

        crossover_point_a = random_generator.integers(low=2, high=chromosome_length_a, size=1)[0]
        crossover_point_b = random_generator.integers(low=2, high=chromosome_length_b,size=1)[0]

        # Could flip a coin to change which parent comes first
        flip_coin = random_generator.choice([0, 1])

        if flip_coin == 0:
            offspring_a = parent_a[0:crossover_point_a] + parent_b[crossover_point_b:]
            offspring_b = parent_b[0:crossover_point_b] + parent_a[crossover_point_a:]
        else:
            offspring_a = parent_b[0:crossover_point_b] + parent_a[crossover_point_a:]
            offspring_b = parent_a[0:crossover_point_a] + parent_b[crossover_point_b:]

    
        return [offspring_a, offspring_b]
    
class RemoveEquivalentAnsätze:

    def __init__(self, n_tests, threshold_value, n_qubits):
        self.n_tests = n_tests
        self.threshold_value = threshold_value
        self.n_qubits = n_qubits
        self.circuit_ansatz = AnsatzSimulation(self.n_qubits)
        
        

    def generate_random_parameters(self, ansatz):
        gates = list(itertools.chain.from_iterable(ansatz.tolist()))
        gate_count = Counter(gates)
        n_params = gate_count['rx_gate'] + gate_count['ry_gate'] + gate_count['rz_gate']
        parameters = torch.rand(n_params)*math.pi

        return parameters

    def test_ansatz(self, ansatz, parameters, state_vector):
        return self.circuit_ansatz.simulate_circuit(input=state_vector, embedding_type='rx', ansatz_chromosome=ansatz, parameters=parameters, measure=False)
            
    # Check whether i calculate the inner product with just the real parts or should i include the imaginary ones as well
    def inner_product(self, state_vector_psi, state_vector_phi):
        return inner(state_vector_psi, state_vector_phi)

    def fidelity(self, state_vector_psi, state_vector_phi):
        return 1 - torch.einsum('i,i->i',state_vector_psi, state_vector_phi)**2


    def euclidian_distance(self, state_vector_psi, state_vector_phi):
        return sqrt(2 - 2*self.inner_product(state_vector_psi.real, state_vector_phi.real))

    # Decide which measure should be used in order to evaluate the

    def is_equal(self, ansatz_a, ansatz_b):
        print(type(ansatz_a))
        ansatz_a = ansatz_a.tolist()
        ansatz_b = ansatz_b.tolist()
        input_test = torch.rand(self.n_tests, self.n_qubits)
        params_a = self.generate_random_parameters(ansatz_a)
        params_b = self.generate_random_parameters(ansatz_b)
        
        output_a = [self.test_ansatz(ansatz_a, params_a, state_vector) for state_vector in input_test]
        output_b = [self.test_ansatz(ansatz_b, params_b, state_vector) for state_vector in input_test]
        output_pair = list(zip(output_a, output_b))
        
        fidelity_score = torch([self.fidelity(state_psi, state_phi) for (state_psi, state_phi) in output_pair])
        
        score = torch.mean(fidelity_score)
        
        return self.threshold_value > score[0]
    

        