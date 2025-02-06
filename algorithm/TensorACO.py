import time
import jax
import jax.numpy as jnp
import numpy as np
import math
from jax import jit, vmap
from functools import partial

from evox import Algorithm, jit_class, State


@jit_class
class TensorACO(Algorithm):
    """
    TensorACO_AdaIR in Evox framework
    Tensor implementation of the Ant Colony Optimization (ACO) algorithm with Adaptive Independent Roulette.

        Args:
            distances (jnp.array): The distance matrix representing the problem.
            n_ants (int): The number of ants to use in the ACO algorithm.
            n_best (int): The number of best ants to use for updating pheromone.
            n_iterations (int): The number of iterations to run the ACO algorithm.
            decay (float): The pheromone decay rate.
            alpha (float): The weight for pheromone in the transition probability.
            beta (float): The weight for distance in the transition probability.

    """

    def __init__(
            self,
            distances,
            node_count=100,
            n_ants=100,
            n_best=10,
            decay=0.5,
            alpha=1,
            beta=2,
            init_lr=1.5
    ):
        """
        Initialize the TensorACO class.
        """
        self.distances = distances
        self.node_count = node_count
        self.n_ants = n_ants
        self.n_best = n_best
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.all_inds = jnp.arange(len(distances))
        self.lr = init_lr
        self.iteration = 0

    def setup(self, key):
        """
        Initialize the state for the ACO algorithm.

        Args:
            PRNGKey (int): The seed for the random number generator.

        Returns:
            dict: The initial state.
        """
        pheromone = jnp.ones(self.distances.shape) / len(self.distances)
        return State(
            pheromone=pheromone,
            key=key,
            best_path=jnp.zeros_like(self.all_inds),
            best_length=np.inf,
            population=jnp.zeros((self.n_ants, self.node_count))
        )

    def ask(self, state):
        self.lr = self._one_cycle_lr(self.iteration)
        self.iteration += 1
        key, new_key, start_key = jax.random.split(state.key, 3)
        key_batch = jax.random.split(key, self.n_ants)
        pmat = state.pheromone ** self.alpha * ((1.0 / self.distances) ** self.beta)
        start_batch = jax.random.randint(start_key, (self.n_ants,), 0, self.node_count)
        paths = vmap(self._path_construction, in_axes=(0, None, 0))(start_batch, pmat, key_batch)
        return paths, state.update(population=paths, key=new_key)

    def tell(self, state, fitness):
        """
        Update the state after each iteration of the ACO algorithm.

        Args:
            all_paths (jnp.ndarray): The paths found by all ants.
            all_lengths (jnp.ndarray): The lengths of the paths found by all ants.
            state (dict): The current state dictionary.

        Returns:
            dict: The updated state dictionary.
        """
        all_paths = state.population
        all_lengths = fitness
        sorted_indices = jnp.argsort(all_lengths)
        top_indices = sorted_indices[:self.n_best]
        top_paths = all_paths[top_indices]
        top_lengths = all_lengths[top_indices]
        shortest_path, shortest_length = all_paths[top_indices[0]], all_lengths[top_indices[0]]

        new_best_path, new_best_length = \
            jax.lax.cond(shortest_length < state.best_length,
                         lambda x: (shortest_path, shortest_length),
                         lambda x: (state.best_path, state.best_length),
                         None)

        update_pheromone_stack = vmap(self._ant_path_tensorization, in_axes=(0, 0))(top_paths, top_lengths)
        delta_pheromone = jnp.sum(update_pheromone_stack, axis=0)

        new_pheromone = state.pheromone * self.decay + delta_pheromone
        return state.update(pheromone=new_pheromone, best_path=new_best_path, best_length=new_best_length)

    def _cyclical_learning_rate(self, iteration, step_size=0.05, base_lr=0.9, max_lr=2):
        cycle = math.floor(1 + iteration / (2 * step_size))
        x = abs(iteration / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
        return lr

    def _cosine_annealing_lr(self, epoch, num_epochs=2000, base_lr=2, max_lr=0.9):
        cos_inner = math.pi * (epoch % (num_epochs // 2))
        cos_inner /= num_epochs // 2
        lr = base_lr + 0.5 * (max_lr - base_lr) * (1 + math.cos(cos_inner))
        return lr

    def _one_cycle_lr(self, iteration, total_iterations=2000, max_lr=0.9, pct_start=0.999, annealing_pct=0.999):
        cycle_length = int(total_iterations * pct_start)
        annealing_length = int(total_iterations * annealing_pct)
        cos_inner = (np.pi * (iteration % cycle_length)) / cycle_length
        lr = max_lr * (np.cos(cos_inner) + 1) / 2
        if iteration >= cycle_length + annealing_length:
            decay_fraction = (iteration - cycle_length - annealing_length) / (
                    total_iterations - cycle_length - annealing_length)
            lr *= (1 - decay_fraction)
        return lr

    def _ant_path_tensorization(self, path, length):
        """
        Calculate the pheromone update for a given path.

        Args:
            path (jnp.ndarray): The path found by an ant.
            length (float): The length of the path.

        Returns:
            jnp.ndarray: The pheromone update for the path.
        """
        edge_indices_x = jnp.array(path[:-1])
        edge_indices_y = jnp.array(path[1:])
        edge_indices_x = jnp.append(edge_indices_x, path[-1])
        edge_indices_y = jnp.append(edge_indices_y, path[0])

        edge_indices_x_all = jnp.append(edge_indices_x, edge_indices_y)
        edge_indices_y_all = jnp.append(edge_indices_y, edge_indices_x)

        update_pheromone = jnp.zeros_like(self.distances)
        update_pheromone = update_pheromone.at[edge_indices_x_all, edge_indices_y_all].set(1. / length,
                                                                                           unique_indices=True)
        return update_pheromone

    def _path_construction(self, start, pmat, key):
        """
        Generate a path for a single ant.

        Args:
            start (int): The starting node for the ant.
            pmat (jnp.ndarray): The pheromone matrix.
            key (jax.random.PRNGKey): The random number generator key.

        Returns:
            jnp.ndarray: The path found by the ant.
        """

        def _fun(i, state):
            path, prev, visit = state
            move = self._selection(pmat[prev], key, visit)
            path = path.at[i + 1].set(move)
            visit = visit.at[move].set(0)
            return path, move, visit

        path = jnp.ones_like(self.all_inds) * start
        visit = jnp.ones_like(self.all_inds)
        visit = visit.at[start].set(0)
        path, last_move, _ = jax.lax.fori_loop(0, len(self.distances) - 1, _fun, (path, start, visit))
        return path

    def _selection(self, pmat, key, visited):
        """
        Pick the next move for an ant based on the pheromone matrix and visited nodes.

        Args:
            pmat (jnp.ndarray): The probability matrix row corresponding to the current node.
            key (jax.random.PRNGKey): The random number generator key.
            visited (jnp.ndarray): The array indicating the visited nodes.

        Returns:
            int: The index of the next node to visit.
        """
        row = pmat * visited
        rmat = jax.random.uniform(key, (len(row),)) ** self.lr
        move = jnp.argmax(row * rmat)
        return move
