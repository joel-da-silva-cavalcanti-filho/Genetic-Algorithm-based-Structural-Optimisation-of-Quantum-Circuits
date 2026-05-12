import numpy as np
import random


def save_generation(gen_number, population, scores, file_path):
    try:
    # Open the file in write mode ('w') and assign it to the variable 'file'
        with open(file_path, 'at') as file:
            # Use the write() method to put content into the file
            file.write(f'Generation #{gen_number+1}\n\n')
            [file.write(f'Individual #{index+1}: \n{str(population[index])}\n\n') for index in range(len(population))]
            file.write('\n')
            file.write(f'Gen #{gen_number+1} Scores\n')
            [file.write(f'Individual #{index+1}s score: {scores[index]}\n') for index in range(len(scores))]
            file.write('\n\n')
        print(f"Successfully saved text to {file_path}")
    except IOError as e:
        print(f"Error saving file: {e}")

class GeneticAlgorithm():
    def __init__(self, crossover_rate: float, mutation_rate: float, no_generations: int, population_size: int, sampling, fitness, crossover, mutation, repair, duplicates):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.no_generations = no_generations
        self.population_size = population_size

        self.Sampling = sampling
        self.Fitness = fitness
        self.Crossover = crossover
        self.Mutation = mutation
        self.Repair = repair
        self.eliminate_duplicates = duplicates

    ## Population deve ter dimensões (population_size, solution_size)

    def initialize_population(self):
        return self.Sampling._do(self.population_size)
  
    def define_fitness(self, population, iteration, weight_filepath) -> np.array:
        return np.array([self.Fitness._evaluate(individual, iteration, population.index(individual), weight_filepath) for individual in population])

    def duplicate_threshold(self, offspring_a, offspring_b):
        if self.eliminate_duplicates.is_equal(offspring_a, offspring_b):
            drop_index = random.randint(len(offspring_b))
            offspring_b.pop(drop_index)
        
        return offspring_a, offspring_b
    
    def select_parents(self, population: list):
        random_generator = np.random.default_rng(seed=42)
        indices = random_generator.choice(len(population), 2, replace=False)
        parent_a = population[indices[0]]
        parent_b = population[indices[1]]
        
        if len(parent_a) < len(parent_b):
            parent_a, parent_b = parent_b, parent_a
        return parent_a, parent_b

    def choose_parent(self) -> int:
        options = [0, 1]
        return random.choice(options)
    
    def crossover(self, parent_a, parent_b):
        return self.Crossover._do(parent_a, parent_b)

    def mutate_offspring(self, offspring) -> np.array:
        return self.Mutation._do(offspring)
    
    ## Use fitness values for elitism or something
    
    def generate_new_population(self, population):
        new_population = []
        for individuals in range(int(self.population_size/2)):
            parent_a, parent_b = self.select_parents(population)
            offspring_a, offspring_b = self.mutate_offspring(self.crossover(parent_a, parent_b))
            
            new_population.append(offspring_a)
            new_population.append(offspring_b)
                
        return self.repair_population(new_population)
    
    def repair_population(self, population):
        return [self.Repair._do(individual) for individual in population]


    def run_algorithm(self, weight_filepath, file_path, start_iteration=0, warmup=None, warmup_best_fitness_value=0.0) -> tuple:
        iterations = start_iteration
        if warmup is None:
            population = self.initialize_population()
        else:
            population = warmup
            current_best_fitness_value = warmup_best_fitness_value
        all_generations = []
        iteration_axis = []
        best_chromossome_axis = []
        
        # 0 != 2
        while iterations != self.no_generations:
            
            print(f'Generation #{iterations+1}')
            print(population)  
            
            if iterations == 0:
                iteration_axis.append(iterations)
                old_fitness_values = self.define_fitness(population, iterations, weight_filepath)
                save_generation(iterations, population, old_fitness_values, file_path)
                

                current_best_individual_position = np.argmax(old_fitness_values)
                current_best_fitness_value = np.max(old_fitness_values)
                current_best_individual = population[current_best_individual_position]
                
                population = self.generate_new_population(population)

            else:
                new_fitness_values = self.define_fitness(population, iterations, weight_filepath)
                #save_generation(iterations, population, new_fitness_values, file_path)

                new_best_individual_position = np.argmax(new_fitness_values)
                new_best_fitness_value = np.max(new_fitness_values)
                new_best_individual = population[new_best_individual_position]
            

                if new_best_fitness_value > current_best_fitness_value:
                    current_best_fitness_value = new_best_fitness_value
                    current_best_individual = new_best_individual

                result_tuple = (current_best_individual, current_best_fitness_value)
                best_chromossome_axis.append(current_best_fitness_value)
            
                random_position = random.randint(0, self.population_size - 1)
                population[random_position] = current_best_individual
                save_generation(iterations, population, new_fitness_values, file_path)
                
            
            iterations +=1
            if iterations != self.no_generations or iterations != 1:
                population = self.generate_new_population(population)             
        
        return result_tuple, best_chromossome_axis, iteration_axis