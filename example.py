import jax
import jax.numpy as jnp
import algorithm
import problem
import os
import evox


if __name__ == '__main__':

    base_path = os.path.dirname(os.path.abspath(__file__))  
    file_path = os.path.join(base_path, "problem", "pcb442.npy") 

    algorithm = algorithm.TensorACO(
        distances=jnp.load(file_path),
        node_count=442,
        n_ants=442,
        n_best=100,
        decay=0.5,
        alpha=1,
        beta=2
    )

    problem = problem.TSP(
        jnp.load(file_path)
    )
    monitor = evox.monitors.StdSOMonitor()

    key = jax.random.PRNGKey(42)

    workflow = evox.workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor
    )

    state = workflow.init(key)

    for i in range(10):
        state = workflow.step(state)
        monitor.flush()
        best_fitness = monitor.get_best_fitness()
        print("Iteration:", i, "Best Fitness:", best_fitness)
