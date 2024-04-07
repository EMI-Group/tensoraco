import jax
import jax.numpy as jnp
import algorithm
import problem

import evox

if __name__ == '__main__':

    algorithm = algorithm.TensorACO(
        distances=jnp.load('problem/pcb442.npy'),
        node_count=442,
        n_ants=442,
        n_best=100,
        decay=0.5,
        alpha=1,
        beta=2
    )

    problem = problem.TSP(
        jnp.load('problem/pcb442.npy')
    )
    monitor = evox.monitors.StdSOMonitor()

    key = jax.random.PRNGKey(42)

    workflow = evox.workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor
    )

    state = workflow.init(key)

    for i in range(100):
        state = workflow.step(state)
        monitor.flush()
        print(monitor.get_best_fitness())
