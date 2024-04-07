import math
from jax import numpy as jnp
from functools import partial
from jax import jit
import random
from evox import Problem, jit_class


@jit_class
class TSP(Problem):

    def __init__(self, adj_matrix):
        super().__init__()
        self.adj_matrix = adj_matrix
        self.num_cities = len(adj_matrix)

    def evaluate(self, state, paths):
        fitness = jnp.sum(self.adj_matrix[paths[:, :-1], paths[:, 1:]], axis=1)
        fitness += self.adj_matrix[paths[:, -1], paths[:, 0]]
        return fitness, state

