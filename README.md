# paper_implementation

Flow / MeanFlow implementation and analysis notebooks.

## Repository layout

- [`flow_basic.ipynb`](./flow_basic.ipynb) — basic flow / MeanFlow practice notebook
- [`mean_flow_experiment1.ipynb`](./mean_flow_experiment1.ipynb) — MeanFlow dynamics, geometry, representation, and generalization diagnostics
- [`meanflow_minimal/`](./meanflow_minimal/) — minimal conditional MeanFlow + DiT implementations
  - [`meanflow_dit_mnist.ipynb`](./meanflow_minimal/meanflow_dit_mnist.ipynb) — MNIST practice
  - [`meanflow_dit_fashion_mnist.ipynb`](./meanflow_minimal/meanflow_dit_fashion_mnist.ipynb) — FashionMNIST practice
  - [`mean_flow_ablation.ipynb`](./meanflow_minimal/mean_flow_ablation.ipynb) — ablation of the implementation choices

## Minimal MeanFlow + DiT

The minimal implementation keeps the core pipeline small and inspectable:

`class-conditioned image -> small DiT -> MeanFlow JVP objective -> one-step generation`

The FashionMNIST run already separates major clothing classes with a small DiT and one-step conditional MeanFlow generation.

### FashionMNIST — step 7000

![Conditional MeanFlow FashionMNIST step 7000](./meanflow_minimal/assets/fashion_mnist_step_07000.jpg)
