import numpy as np
import random

import time

time_rec = []

class AntColonyOptimizer:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):
        """
        A simple ACO implementation.
        """
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            start_time = time.time()
            all_paths = self.gen_all_paths()
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.spread_pheronome(all_paths, self.n_best)
            self.pheromone *= self.decay
            #print iteration result
            print("Iteration #{}: Best path length = {}".format(i, all_time_shortest_path[1]))
            print("time:", time.time() - start_time)
            time_rec.append(time.time() - start_time)
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / dist
                self.pheromone[move[::-1]] += 1.0 / dist

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = (pheromone ** self.alpha) * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move


def np_choice(a, size, p=None):
    return np.array(random.choices(a, weights=p, k=size))


# Example usage
np.random.seed(7)
distances = np.load('../problem/pcb442.npy')
print(distances)
aco = AntColonyOptimizer(distances, n_ants=159, n_best=16, n_iterations=1000, decay=0.5, alpha=1, beta=2)
shortest_path = aco.run()
print(f"Shortest path: {shortest_path}")
