"""
MOEA/D (Multiobjective Evolutionary Algorithm based on Decomposition) Implementation
Based on Zhang and Li (2007) - IEEE Transactions on Evolutionary Computation

This implementation includes:
- Weighted Sum decomposition
- Tchebycheff (Chebyshev) decomposition
- Neighborhood-based evolution
- Basic evolutionary operators
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
import random
from scipy.spatial.distance import cdist


class MOEAD:
    """
    MOEA/D algorithm implementation with multiple decomposition methods.
    """
    
    def __init__(self, 
                 n_objectives: int,
                 n_variables: int, 
                 population_size: int,
                 neighborhood_size: int = 20,
                 decomposition: str = 'tchebycheff',
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1,
                 bounds: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize MOEA/D algorithm.
        
        Args:
            n_objectives: Number of objectives
            n_variables: Number of decision variables
            population_size: Size of the population (number of subproblems)
            neighborhood_size: Number of neighboring subproblems
            decomposition: Decomposition method ('weighted_sum' or 'tchebycheff')
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            bounds: List of (min, max) bounds for each variable
        """
        self.n_objectives = n_objectives
        self.n_variables = n_variables
        self.population_size = population_size
        self.neighborhood_size = min(neighborhood_size, population_size)
        self.decomposition = decomposition
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # Default bounds [0, 1] for each variable if not provided
        if bounds is None:
            self.bounds = [(0.0, 1.0)] * n_variables
        else:
            self.bounds = bounds
            
        # Initialize algorithm components
        self.weight_vectors = self._generate_weight_vectors()
        self.neighborhoods = self._compute_neighborhoods()
        self.population = self._initialize_population()
        self.ideal_point = np.full(n_objectives, np.inf)
        
        # Storage for non-dominated solutions
        self.external_population = []
        
    def _generate_weight_vectors(self) -> np.ndarray:
        """
        Generate uniformly distributed weight vectors for decomposition.
        Uses the systematic approach for generating weight vectors.
        """
        if self.n_objectives == 2:
            # For 2 objectives, create uniform distribution
            weights = np.zeros((self.population_size, self.n_objectives))
            for i in range(self.population_size):
                weights[i, 0] = i / (self.population_size - 1)
                weights[i, 1] = 1.0 - weights[i, 0]
        else:
            # For more objectives, use random generation (can be improved with systematic methods)
            weights = np.random.random((self.population_size, self.n_objectives))
            # Normalize to sum to 1
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            
        return weights
    
    def _compute_neighborhoods(self) -> List[List[int]]:
        """
        Compute neighborhoods based on Euclidean distance between weight vectors.
        """
        distances = cdist(self.weight_vectors, self.weight_vectors)
        neighborhoods = []
        
        for i in range(self.population_size):
            # Get indices of nearest neighbors (including self)
            neighbor_indices = np.argsort(distances[i])[:self.neighborhood_size]
            neighborhoods.append(neighbor_indices.tolist())
            
        return neighborhoods
    
    def _initialize_population(self) -> np.ndarray:
        """
        Initialize population randomly within bounds.
        """
        population = np.zeros((self.population_size, self.n_variables))
        
        for i in range(self.population_size):
            for j in range(self.n_variables):
                min_val, max_val = self.bounds[j]
                population[i, j] = random.uniform(min_val, max_val)
                
        return population
    
    def weighted_sum(self, objectives: np.ndarray, weight_vector: np.ndarray) -> float:
        """
        Weighted Sum decomposition method.
        
        Args:
            objectives: Array of objective values
            weight_vector: Weight vector for this subproblem
            
        Returns:
            Scalar fitness value
        """
        return np.sum(weight_vector * objectives)
    
    def tchebycheff(self, objectives: np.ndarray, weight_vector: np.ndarray) -> float:
        """
        Tchebycheff (Chebyshev) decomposition method.
        
        Args:
            objectives: Array of objective values
            weight_vector: Weight vector for this subproblem
            
        Returns:
            Scalar fitness value
        """
        # Avoid division by zero
        weight_vector = np.maximum(weight_vector, 1e-6)
        return np.max(weight_vector * np.abs(objectives - self.ideal_point))
    
    def evaluate_subproblem(self, objectives: np.ndarray, subproblem_index: int) -> float:
        """
        Evaluate a solution for a specific subproblem using the chosen decomposition.
        
        Args:
            objectives: Array of objective values
            subproblem_index: Index of the subproblem
            
        Returns:
            Scalar fitness value for the subproblem
        """
        weight_vector = self.weight_vectors[subproblem_index]
        
        if self.decomposition == 'weighted_sum':
            return self.weighted_sum(objectives, weight_vector)
        elif self.decomposition == 'tchebycheff':
            return self.tchebycheff(objectives, weight_vector)
        else:
            raise ValueError(f"Unknown decomposition method: {self.decomposition}")
    
    def simulated_binary_crossover(self, parent1: np.ndarray, parent2: np.ndarray, 
                                 eta: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulated Binary Crossover (SBX) operator.
        
        Args:
            parent1, parent2: Parent solutions
            eta: Distribution index for SBX
            
        Returns:
            Two offspring solutions
        """
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        for i in range(self.n_variables):
            if random.random() <= self.crossover_rate:
                min_val, max_val = self.bounds[i]
                
                # Ensure parents are within bounds
                y1 = max(min_val, min(max_val, parent1[i]))
                y2 = max(min_val, min(max_val, parent2[i]))
                
                if abs(y1 - y2) > 1e-14:
                    if y1 > y2:
                        y1, y2 = y2, y1
                    
                    # Calculate beta
                    rand = random.random()
                    beta = 2.0 * rand if rand <= 0.5 else 1.0 / (2.0 * (1.0 - rand))
                    beta = beta ** (1.0 / (eta + 1.0))
                    
                    # Calculate offspring
                    offspring1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    offspring2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    # Ensure bounds
                    offspring1[i] = max(min_val, min(max_val, offspring1[i]))
                    offspring2[i] = max(min_val, min(max_val, offspring2[i]))
        
        return offspring1, offspring2
    
    def polynomial_mutation(self, solution: np.ndarray, eta: float = 20.0) -> np.ndarray:
        """
        Polynomial mutation operator.
        
        Args:
            solution: Solution to mutate
            eta: Distribution index for mutation
            
        Returns:
            Mutated solution
        """
        mutated = solution.copy()
        
        for i in range(self.n_variables):
            if random.random() <= self.mutation_rate:
                min_val, max_val = self.bounds[i]
                
                # Calculate mutation
                rand = random.random()
                if rand < 0.5:
                    delta = (2.0 * rand) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / (eta + 1.0))
                
                mutated[i] = solution[i] + delta * (max_val - min_val)
                mutated[i] = max(min_val, min(max_val, mutated[i]))
        
        return mutated
    
    def update_ideal_point(self, objectives: np.ndarray):
        """
        Update the ideal point (best value for each objective).
        
        Args:
            objectives: New objective values to consider
        """
        self.ideal_point = np.minimum(self.ideal_point, objectives)
    
    def optimize(self, objective_function: Callable, max_generations: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the MOEA/D optimization algorithm.
        
        Args:
            objective_function: Function that takes a solution and returns objective values
            max_generations: Maximum number of generations to run
            
        Returns:
            Tuple of (population, objective_values)
        """
        # Evaluate initial population
        objective_values = np.zeros((self.population_size, self.n_objectives))
        for i in range(self.population_size):
            objective_values[i] = objective_function(self.population[i])
            self.update_ideal_point(objective_values[i])
        
        # Main evolutionary loop
        for generation in range(max_generations):
            for i in range(self.population_size):
                # Select parents from neighborhood
                neighborhood = self.neighborhoods[i]
                parent_indices = random.sample(neighborhood, 2)
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                
                # Generate offspring
                offspring1, offspring2 = self.simulated_binary_crossover(parent1, parent2)
                offspring1 = self.polynomial_mutation(offspring1)
                offspring2 = self.polynomial_mutation(offspring2)
                
                # Evaluate offspring
                for offspring in [offspring1, offspring2]:
                    offspring_objectives = objective_function(offspring)
                    self.update_ideal_point(offspring_objectives)
                    
                    # Update neighboring subproblems
                    for j in neighborhood:
                        # Check if offspring improves subproblem j
                        current_fitness = self.evaluate_subproblem(objective_values[j], j)
                        offspring_fitness = self.evaluate_subproblem(offspring_objectives, j)
                        
                        if offspring_fitness < current_fitness:
                            self.population[j] = offspring.copy()
                            objective_values[j] = offspring_objectives.copy()
            
            # Print progress
            if generation % 10 == 0:
                print(f"Generation {generation}/{max_generations}")
        
        return self.population, objective_values