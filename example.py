import time
import jax
import jax.numpy as jnp
import algorithm
from algorithm import TensorACO


def run_tensoraco(algorithm, key):
    state = TensorACO.setup(algorithm, key)

    for i in range(100):

        offspring, state = TensorACO.ask(algorithm, state)
        state = TensorACO.tell(algorithm, offspring, state)
        print("The distance is ", state['best_length'])


if __name__ == '__main__':

    algorithm = algorithm.TensorACO(
        distances = jnp.load('problem/pcb442.npy'),
        n_ants = 442,
        n_best = 100,
        decay = 0.5,
        alpha = 1,
        beta = 2
    )
    key = jax.random.PRNGKey(42)
    start_time = time.time()
    run_tensoraco(algorithm, key)
    end_time = time.time()
    print(f"time: {end_time - start_time} s")