<h1 align="center">
  <a href="https://github.com/EMI-Group/evox">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./assets/evox_logo_light.png">
      <img alt="EvoX Logo" height="50" src="./assets/evox_logo_light.png">
  </picture>
  </a> 
  <br>
</h1>

<p align="center">
🌟 TensorACO: Tensorized Ant Colony Optimization for GPU Acceleration 🌟
</p>


<p align="center">
  <a href="http://arxiv.org/abs/2404.04895">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="TensorACO Paper on arXiv">
  </a>
</p>
Tensorized Ant Colony Optimization (TensorACO) enhances the convergence speed and efficient of large-scale Traveling Salesman Problems (TSP) by incorporating GPU acceleration.  By tensorizing the ant system and path, TensorACO capitalizes on GPU parallelism for accelerated computation. Additionally, the Adaptive Independent Roulette (AdaIR) method enhances the performance by a dynamically strategy. TensorACO is compatible with the <a href="https://github.com/EMI-Group/evox">EvoX</a> framework.

## Key Features

---

- **GPU Acceleration** 💻: Leverages GPUs for enhanced computational capabilities.

- **Large-Scale Optimization** 📈: Ideal for large ant colony sizes and large city sizes.

- **Real-World Applications** 🌐: Suited for TSP, a central challenge in combinatorial optimization, is characterized by the theoretical complexity and practical relevance in routing and logistics.

## Requirements

---

TensorACO requires:

- Python (version >= 3.7)
- evox (version == 0.8.0)
- JAX (version == 0.4.34)
- jaxlib (version == 0.4.34)
- (Optional) CUDA and cuDNN for GPU acceleration

## Example Usage

---

```python
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



```

## Community & Support

---

- Engage in discussions and share your experiences on [GitHub Discussion Board](https://github.com/EMI-Group/evox/discussions).
- Join our QQ group (ID: 297969717).

## Citing TensorACO

---

If you use TensorACO in your research and want to cite it in your work, please use:

```
@article{tensoraco,
  title = {{Tensorized} {Ant} {Colony} {Optimization} {for} {GPU} {Acceleration}},
  author = {Yang, Luming and Jiang, Tao and Cheng, Ran},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
  year = {2024}
}
```
