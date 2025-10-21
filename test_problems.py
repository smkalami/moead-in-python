"""
Test Problems for MOEA/D
Common multiobjective benchmark functions for testing optimization algorithms.
"""

import numpy as np
import math
from typing import Callable


class TestProblems:
    """Collection of standard multiobjective test problems."""
    
    @staticmethod
    def zdt1(x: np.ndarray) -> np.ndarray:
        """
        ZDT1 test function (2 objectives, n variables)
        
        Pareto front: f2 = 1 - sqrt(f1), where f1 ∈ [0,1]
        
        Args:
            x: Decision variables (should be in [0,1])
            
        Returns:
            Array of objective values [f1, f2]
        """
        n = len(x)
        
        # First objective
        f1 = x[0]
        
        # Helper function
        g = 1 + 9 * np.sum(x[1:]) / (n - 1)
        
        # Second objective
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        
        return np.array([f1, f2])
    
    @staticmethod
    def zdt2(x: np.ndarray) -> np.ndarray:
        """
        ZDT2 test function (2 objectives, n variables)
        Non-convex Pareto front: f2 = 1 - (f1)^2
        
        Args:
            x: Decision variables (should be in [0,1])
            
        Returns:
            Array of objective values [f1, f2]
        """
        n = len(x)
        
        # First objective
        f1 = x[0]
        
        # Helper function
        g = 1 + 9 * np.sum(x[1:]) / (n - 1)
        
        # Second objective
        h = 1 - (f1 / g) ** 2
        f2 = g * h
        
        return np.array([f1, f2])
    
    @staticmethod
    def zdt3(x: np.ndarray) -> np.ndarray:
        """
        ZDT3 test function (2 objectives, n variables)
        Discontinuous Pareto front
        
        Args:
            x: Decision variables (should be in [0,1])
            
        Returns:
            Array of objective values [f1, f2]
        """
        n = len(x)
        
        # First objective
        f1 = x[0]
        
        # Helper function
        g = 1 + 9 * np.sum(x[1:]) / (n - 1)
        
        # Second objective
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        
        return np.array([f1, f2])
    
    @staticmethod
    def dtlz1(x: np.ndarray, n_objectives: int = 3) -> np.ndarray:
        """
        DTLZ1 test function (scalable objectives)
        
        Args:
            x: Decision variables (should be in [0,1])
            n_objectives: Number of objectives (default: 3)
            
        Returns:
            Array of objective values
        """
        n = len(x)
        k = n - n_objectives + 1
        
        # Calculate g function
        g = 100 * (k + np.sum([(xi - 0.5)**2 - np.cos(20*np.pi*(xi - 0.5)) for xi in x[-k:]]))
        
        # Calculate objectives
        objectives = np.zeros(n_objectives)
        
        for i in range(n_objectives):
            objectives[i] = 0.5 * (1 + g)
            
            # Product term
            for j in range(n_objectives - 1 - i):
                objectives[i] *= x[j]
            
            # Sine term (for objectives > 1)
            if i > 0:
                objectives[i] *= (1 - x[n_objectives - 1 - i])
        
        return objectives
    
    @staticmethod
    def schaffer_n1(x: np.ndarray) -> np.ndarray:
        """
        Schaffer's function N.1 (2 objectives, 1 variable)
        Simple test function for quick validation
        
        Args:
            x: Decision variable (single value or array with one element)
            
        Returns:
            Array of objective values [f1, f2]
        """
        if isinstance(x, np.ndarray):
            x_val = x[0] if len(x) > 0 else 0
        else:
            x_val = x
            
        f1 = x_val ** 2
        f2 = (x_val - 2) ** 2
        
        return np.array([f1, f2])
    
    @staticmethod
    def get_problem(name: str) -> Callable:
        """
        Get a test problem by name.
        
        Args:
            name: Name of the test problem
            
        Returns:
            Test problem function
        """
        problems = {
            'zdt1': TestProblems.zdt1,
            'zdt2': TestProblems.zdt2,
            'zdt3': TestProblems.zdt3,
            'dtlz1': TestProblems.dtlz1,
            'schaffer': TestProblems.schaffer_n1
        }
        
        if name.lower() not in problems:
            raise ValueError(f"Unknown problem: {name}. Available: {list(problems.keys())}")
        
        return problems[name.lower()]


class ProblemConfig:
    """Configuration for standard test problems."""
    
    @staticmethod
    def get_config(problem_name: str) -> dict:
        """
        Get recommended configuration for a test problem.
        
        Args:
            problem_name: Name of the test problem
            
        Returns:
            Dictionary with recommended settings
        """
        configs = {
            'zdt1': {
                'n_variables': 10,
                'n_objectives': 2,
                'bounds': [(0.0, 1.0)] * 10,
                'population_size': 100,
                'max_generations': 200,
                'pareto_front_description': 'Convex, f2 = 1 - sqrt(f1)'
            },
            'zdt2': {
                'n_variables': 10,
                'n_objectives': 2,
                'bounds': [(0.0, 1.0)] * 10,
                'population_size': 100,
                'max_generations': 200,
                'pareto_front_description': 'Non-convex, f2 = 1 - f1^2'
            },
            'zdt3': {
                'n_variables': 10,
                'n_objectives': 2,
                'bounds': [(0.0, 1.0)] * 10,
                'population_size': 100,
                'max_generations': 200,
                'pareto_front_description': 'Disconnected/discontinuous'
            },
            'dtlz1': {
                'n_variables': 7,  # For 3 objectives: k=5, total=7
                'n_objectives': 3,
                'bounds': [(0.0, 1.0)] * 7,
                'population_size': 150,
                'max_generations': 300,
                'pareto_front_description': '3D triangular simplex'
            },
            'schaffer': {
                'n_variables': 1,
                'n_objectives': 2,
                'bounds': [(-10.0, 10.0)],
                'population_size': 50,
                'max_generations': 100,
                'pareto_front_description': 'Simple convex, quick test'
            }
        }
        
        if problem_name.lower() not in configs:
            raise ValueError(f"Unknown problem: {problem_name}. Available: {list(configs.keys())}")
        
        return configs[problem_name.lower()]