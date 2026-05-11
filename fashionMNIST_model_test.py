from core.ansatz_optimization_problem import QuantumCircuitOptimization, QuantumCircuitSampling
from core.ansatz_mutation_crossover import AnsatzCrossover, AnsatzMutation, RemoveEquivalentAnsätze, RepairAnsatz
from torchvision import datasets, transforms
from core.patch_making_v2 import PatchExtraction, sample_data, SampledDataset4Training, sampling
from torch.utils.data import Dataset, DataLoader, Subset
from core.ga_v2 import GeneticAlgorithm
from core.genetic_hqcnn import test_model, HybridModel
import time
import torch
import torch.nn as nn

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
    batch_size = 12
    train_ratio = 0.7
    patch_size = 2
    n_channels = 1

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        PatchExtraction(patch_size)])

    sample_size = 36

    full_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


    sampled_training_dataset = sampling(full_testset, target_classes, sample_size)
    fashionMNIST_testset = SampledDataset4Training(sampled_training_dataset, target_classes)

    
    output_file_path = 'results/resultados_fashionMNIST_testset_v1.txt'
    weights_filepath = 'weights_v2/fashionMNIST_GA_gen_#6_individual_#2_v2.pth'
    
    circuit_ansatz = [['hadamard', 't', 'pauli_x', 'pauli_z'], ['rx_gate', 'ctrl_0', 'trgt_0', 'rz_gate'], ['hadamard', 't', 'pauli_x', 'pauli_z'], ['rz_gate', 'rx_gate', 'rz_gate', 'rx_gate'], ['rz_gate', 'rz_gate', 'ry_gate', 'rz_gate'], ['ctrl_0', 'trgt_0', 'ctrl_1', 'trgt_1'], ['t', 'hadamard', 'pauli_z', 'pauli_x'], ['rx_gate', 'ctrl_0', 'trgt_0', 'rz_gate'], ['rz_gate', 'rz_gate', 'ry_gate', 'rz_gate']]
    
    test_hqcnn = HybridModel(n_qubits, patch_size, circuit_ansatz, n_classes, input_size, n_channels, mode)
    
    optimizer = torch.optim.Adam(test_hqcnn.parameters(), lr=0.1, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    test_loader = DataLoader(fashionMNIST_testset, batch_size, shuffle=True)
    
    test_hqcnn.load_state_dict(torch.load(weights_filepath, weights_only=True))

    avg, acc, prec, rec, f1, auc = test_model(test_hqcnn, test_loader, loss_fn, output_file_path)
    