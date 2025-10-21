"""
Simple MOEA/D Problem Solver
Define your problem in a dictionary and get results!
"""

import numpy as np
import matplotlib.pyplot as plt
from moead import MOEAD
from test_problems import TestProblems


def run_moead_simple(objective_function, 
                    n_objectives: int,
                    n_variables: int,
                    bounds: list,
                    population_size: int = 100,
                    max_generations: int = 200,
                    decomposition: str = 'tchebycheff',
                    neighborhood_size: int = 20,
                    crossover_rate: float = 0.9,
                    mutation_rate: float = 0.1,
                    plot_results: bool = True,
                    plot_title: str = "MOEA/D Results"):
    """
    Simple interface to run MOEA/D with custom settings.
    
    Args:
        objective_function: Function that takes a solution array and returns objective values
        n_objectives: Number of objectives
        n_variables: Number of decision variables
        bounds: List of (min, max) tuples for each variable
        population_size: Population size (default: 100)
        max_generations: Maximum generations (default: 200)
        decomposition: 'weighted_sum' or 'tchebycheff' (default: 'tchebycheff')
        neighborhood_size: Neighborhood size (default: 20)
        crossover_rate: Crossover probability (default: 0.9)
        mutation_rate: Mutation probability (default: 0.1)
        plot_results: Whether to plot results (default: True)
        plot_title: Title for the plot
        
    Returns:
        Dictionary with 'population', 'objectives', and 'algorithm' results
    """
    
    print(f"Running MOEA/D with {decomposition} decomposition")
    print(f"Problem: {n_objectives} objectives, {n_variables} variables")
    print(f"Population size: {population_size}, Generations: {max_generations}")
    print("=" * 60)
    
    # Initialize MOEA/D
    moead = MOEAD(
        n_objectives=n_objectives,
        n_variables=n_variables,
        population_size=population_size,
        neighborhood_size=neighborhood_size,
        decomposition=decomposition,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        bounds=bounds
    )
    
    # Run optimization
    print("Starting optimization...")
    final_population, final_objectives = moead.optimize(
        objective_function=objective_function,
        max_generations=max_generations
    )
    
    print("Optimization completed!")
    
    # Display results
    print(f"\nResults:")
    print(f"  - Final population size: {len(final_population)}")
    print(f"  - Objective ranges:")
    for i in range(n_objectives):
        obj_min = np.min(final_objectives[:, i])
        obj_max = np.max(final_objectives[:, i])
        print(f"    Objective {i+1}: [{obj_min:.4f}, {obj_max:.4f}]")
    
    # Plot results if requested
    if plot_results:
        plot_optimization_results(final_objectives, plot_title)
    
    return {
        'population': final_population,
        'objectives': final_objectives,
        'algorithm': moead
    }


def plot_optimization_results(objectives: np.ndarray, title: str):
    """
    Plot optimization results.
    
    Args:
        objectives: Array of objective values
        title: Plot title
    """
    
    if objectives.shape[1] == 2:
        # 2D plot for 2 objectives
        plt.figure(figsize=(10, 6))
        plt.scatter(objectives[:, 0], objectives[:, 1], alpha=0.7, s=30, 
                   color='blue', label='MOEA/D Solutions')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
    elif objectives.shape[1] == 3:
        # 3D plot for 3 objectives
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], 
                  alpha=0.7, s=30, color='blue')
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        ax.set_title(title)
        
    else:
        # For more than 3 objectives, show pairwise plots
        n_obj = objectives.shape[1]
        n_plots = min(6, n_obj * (n_obj - 1) // 2)  # Limit number of plots
        
        if n_plots > 1:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes = [axes]
        
        fig.suptitle(title)
        
        plot_idx = 0
        for i in range(n_obj):
            for j in range(i+1, n_obj):
                if plot_idx >= n_plots:
                    break
                    
                ax = axes[plot_idx]
                ax.scatter(objectives[:, i], objectives[:, j], alpha=0.7, s=20)
                ax.set_xlabel(f'Objective {i+1}')
                ax.set_ylabel(f'Objective {j+1}')
                ax.grid(True, alpha=0.3)
                plot_idx += 1
                
            if plot_idx >= n_plots:
                break
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# DEFINE YOUR PROBLEM HERE - Just modify this dictionary!
# =============================================================================

problem = {
    # Problem definition
    'name': 'ZDT1',
    'objective_function': TestProblems.zdt1,  # Your function here
    'n_objectives': 2,
    'n_variables': 10,
    'bounds': [(0.0, 1.0)] * 10,  # [(min, max), (min, max), ...]
    
    # Algorithm settings
    'population_size': 100,
    'max_generations': 200,
    'decomposition': 'tchebycheff',  # 'weighted_sum' or 'tchebycheff'
    
    # Optional settings (can be omitted for defaults)
    'neighborhood_size': 20,
    'crossover_rate': 0.9,
    'mutation_rate': 0.1,
    
    # Plot settings
    'plot_title': 'MOEA/D Results - ZDT1'
}

# =============================================================================
# RUN OPTIMIZATION - Don't modify this section!
# =============================================================================

if __name__ == "__main__":
    
    print(f"🚀 Solving Problem: {problem['name']}")
    print("=" * 60)
    
    # Extract required parameters
    required_params = {
        'objective_function': problem['objective_function'],
        'n_objectives': problem['n_objectives'],
        'n_variables': problem['n_variables'],
        'bounds': problem['bounds'],
        'population_size': problem['population_size'],
        'max_generations': problem['max_generations'],
        'decomposition': problem['decomposition'],
        'plot_title': problem['plot_title']
    }
    
    # Add optional parameters if specified
    optional_params = ['neighborhood_size', 'crossover_rate', 'mutation_rate']
    for param in optional_params:
        if param in problem:
            required_params[param] = problem[param]
    
    # Run MOEA/D
    results = run_moead_simple(**required_params)
    
    # Show summary
    print("\n")
    print(f"🎉 Problem '{problem['name']}' solved successfully!")
    print(f"📊 Found {len(results['objectives'])} Pareto optimal solutions")
    
    # Show best solution for each objective
    objectives = results['objectives']
    print(f"\n🏆 Best solutions:")
    for i in range(problem['n_objectives']):
        best_idx = np.argmin(objectives[:, i])
        best_value = objectives[best_idx, i]
        print(f"   Objective {i+1} minimum: {best_value:.6f}")
    
    print(f"\n📈 Pareto front plotted above!")
    
    # Optional: Save results to variables for further analysis
    pareto_front = results['objectives']
    pareto_solutions = results['population']
    
    print(f"\nResults saved to variables:")
    print(f"- pareto_front: {pareto_front.shape} array of objective values")
    print(f"- pareto_solutions: {pareto_solutions.shape} array of decision variables")