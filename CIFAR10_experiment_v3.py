from core.ansatz_optimization_problem import QuantumCircuitOptimization, QuantumCircuitSampling
from core.ansatz_mutation_crossover import AnsatzCrossover, AnsatzMutation, RemoveEquivalentAnsätze, RepairAnsatz
from torchvision import datasets, transforms
from core.patch_making_v2 import PatchExtraction, sample_data, SampledDataset4Training, sampling
from torch.utils.data import Dataset, DataLoader, Subset
from core.ga_v2 import GeneticAlgorithm
import time

if __name__ == '__main__':
    population_size = 6
    crossover_rate = 0.8
    mutation_rate = 0.3
    generations = 6
    seed = 42

    # Circuit ansatz hyperparameters
    n_tests = 4
    equivalence_ratio = 0.7
    n_qubits = 4
    max_gates = 40
    gate_options = [None, 'pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard', 'ctrl', 'trgt']
    max_layers = 10

    # Model and training hyperparameters
    input_size = 20
    target_classes = list(range(0,10))
    n_classes = len(target_classes)
    mode = 'frozen'
    batch_size = 30
    train_ratio = 0.7
    patch_size = 2
    n_channels = 3

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        PatchExtraction(patch_size)])

    sample_size = 120

    full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


    sampled_training_dataset = sampling(full_trainset, target_classes, sample_size)
    CIFAR10_training_dataset = SampledDataset4Training(sampled_training_dataset, target_classes)


    ansatz_optimization_problem = QuantumCircuitOptimization(n_qubits=n_qubits,
                                                                n_gates=max_gates,
                                                                possible_gates=gate_options,
                                                                n_layers=max_layers,
                                                                patch_size=patch_size,
                                                                max_gates=max_gates,
                                                                input_size=input_size,
                                                                n_classes=n_classes,
                                                                mode=mode,
                                                                dataset=CIFAR10_training_dataset,
                                                                batch_size=batch_size,
                                                                train_ratio=train_ratio,
                                                                n_channels=n_channels)

    duplicate_ansatz = RemoveEquivalentAnsätze(n_tests, equivalence_ratio, n_qubits)

    repairer = RepairAnsatz(n_qubits=n_qubits, max_layers=max_layers)

    genetic_algo = GeneticAlgorithm(crossover_rate=crossover_rate,
                                    mutation_rate=mutation_rate,
                                    no_generations=generations,
                                    population_size=population_size,
                                    sampling=QuantumCircuitSampling(n_qubits),
                                    fitness=ansatz_optimization_problem,
                                    crossover=AnsatzCrossover(crossover_rate),
                                    mutation=AnsatzMutation(mutation_rate),
                                    repair=repairer,
                                    duplicates=duplicate_ansatz)
    
    file_path = 'results/resultados_CIFAR10_v2.txt'
    weights_filepath = 'weights/CIFAR10'

    
    start_time = time.perf_counter()
    result, best_solution_axis, iterations = genetic_algo.run_algorithm(weights_filepath, file_path)
    end_time = time.perf_counter()
    
    
    try:
        with open(file_path, 'at') as file:
            file.write(f'\n\nThis experiment took {(end_time-start_time)/60.0} minutes')
    except IOError as e:
        print(f'Error saving file: {e}')