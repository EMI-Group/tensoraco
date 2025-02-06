import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from functools import partial
import os


class TensorACORW:
    """
    TensorACO_RW
    Tensor implementation of the Ant Colony Optimization (ACO) algorithm with Roulette Wheel.

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
            n_ants,
            n_best,
            n_iterations,
            decay=0.5,
            alpha=1,
            beta=2
    ):
        """
        Initialize the TensorACO class.
        """
        self.distances = jnp.array(distances)
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.all_inds = jnp.arange(len(distances))

    def setup(self, PRNGKey):
        """
        Initialize the state for the ACO algorithm.

        Args:
            PRNGKey (int): The seed for the random number generator.

        Returns:
            dict: The initial state.
        """
        pheromone = jnp.ones(self.distances.shape) / len(self.distances)
        key = jax.random.PRNGKey(PRNGKey)
        return {'pheromone': pheromone, 'key': key, 'best_path': jnp.zeros_like(self.all_inds), 'best_length': np.inf}

    def run(self, PRNGKey):
        """
        Run the ACO algorithm.

        Args:
            PRNGKey (int): The seed for the random number generator.

        Returns:
            tuple: The best path and its length.
        """
        state = self.setup(PRNGKey)
        record_length = []
        start_time = time.time()

        for i in range(self.n_iterations):
            all_paths, all_lengths, state = self.ant_system_tensorization(state)
            state = self.update(all_paths, all_lengths, state)
            print(state['best_length'])
            record_length.append(state['best_length'])

        print("time:", time.time() - start_time)

        # Save argument, record length, and time to file
        dict = {'distances': self.distances, 'n_ants': self.n_ants, 'n_best': self.n_best,
                'n_iterations': self.n_iterations, 'decay': self.decay, 'alpha': self.alpha,
                'beta': self.beta, 'record_length': record_length, 'time': time.time() - start_time}
        file_name = f'result{self.n_ants}_{PRNGKey}_{time.strftime("%m-%d-%H-%M", time.localtime())}.npy'
        np.save(file_name, dict)
        file = open(file_name + '.txt', 'w')
        print(dict, file=file)
        file.close()

        return state['best_path'], state['best_length']

    @partial(jit, static_argnums=(0,))
    def update(self, all_paths, all_lengths, state):
        """
        Update the state after each iteration of the ACO algorithm.

        Args:
            all_paths (jnp.ndarray): The paths found by all ants.
            all_lengths (jnp.ndarray): The lengths of the paths found by all ants.
            state (dict): The current state dictionary.

        Returns:
            dict: The updated state dictionary.
        """
        sorted_indices = jnp.argsort(all_lengths)
        top_indices = sorted_indices[:self.n_best]
        top_paths = all_paths[top_indices]
        top_lengths = all_lengths[top_indices]
        shortest_path, shortest_length = all_paths[top_indices[0]], all_lengths[top_indices[0]]

        state['best_path'], state['best_length'] = \
            jax.lax.cond(shortest_length < state['best_length'],
                         lambda x: (shortest_path, shortest_length),
                         lambda x: (state['best_path'], state['best_length']),
                         None)

        update_pheromone_stack = vmap(self.ant_path_tensorization, in_axes=(0, 0))(top_paths, top_lengths)
        delta_pheromone = jnp.sum(update_pheromone_stack, axis=0)

        state['pheromone'] = state['pheromone'] * self.decay + delta_pheromone
        return state

    @partial(jit, static_argnums=(0,))
    def ant_path_tensorization(self, path, length):
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

    @partial(jit, static_argnums=(0,))
    def ant_system_tensorization(self, state):
        """
        Generate paths for all ants in the current iteration.

        Args:
            state (dict): The current state dictionary.

        Returns:
            tuple: A tuple containing the paths, lengths, and updated state.
        """
        key, new_key = jax.random.split(state['key'])
        key_batch = jax.random.split(key, self.n_ants)
        state['key'] = new_key
        pmat = state['pheromone'] ** self.alpha * ((1.0 / self.distances) ** self.beta)
        ant_ids = jnp.arange(self.n_ants)
        paths = vmap(self.path_construction, in_axes=(0, None, 0))(ant_ids, pmat, key_batch)
        dists = jnp.sum(self.distances[paths[:, :-1], paths[:, 1:]], axis=1)
        dists += self.distances[paths[:, -1], paths[:, 0]]
        return paths, dists, state

    @partial(jit, static_argnums=(0,))
    def path_construction(self, start, pmat, key):
        """
        Generate a path for a single ant.

        Args:
            start (int): The starting node for the ant.
            pmat (jnp.ndarray): The pheromone matrix.
            key (jax.random.PRNGKey): The random number generator key.

        Returns:
            jnp.ndarray: The path found by the ant.
        """

        def fun(i, state):
            path, prev, visit = state
            move = self.selection(pmat[prev], key, visit)
            path = path.at[i + 1].set(move)
            visit = visit.at[move].set(0)
            return path, move, visit

        path = jnp.ones_like(self.all_inds) * start
        visit = jnp.ones_like(self.all_inds)
        visit = visit.at[start].set(0)
        path, last_move, _ = jax.lax.fori_loop(0, len(self.distances) - 1, fun, (path, start, visit))
        return path

    @partial(jit, static_argnums=(0,))
    def selection(self, pmat, key, visited):
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
        norm_row = row / jnp.sum(row)
        move = jax.random.choice(key, self.all_inds, p=norm_row)
        return move


# Example usage
if __name__ == '__main__':
    base_path = os.path.dirname(os.path.abspath(__file__))  
    file_path = os.path.join(base_path, "..", "problem", "pcb442.npy") 
    distances = np.load(file_path) 
    n_ants = 442
    n_best = 100
    n_iterations = 100
    decay = 0.5
    alpha = 1
    beta = 2
    prng_key_seed = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    for i in range(10):
        prng_key = prng_key_seed[i]
        # print("PRNG_key:", prng_key)
        print("Iteration: ", i)
        taco = TensorACORW(distances, n_ants, n_best, n_iterations, decay, alpha, beta)
        answer = taco.run(prng_key)
        print(f"Shortest path: {answer}")
