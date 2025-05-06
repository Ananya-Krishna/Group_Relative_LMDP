# Group Relative Linear Markov Decision Processes

This repository contains the implementation of **Aggregated Feature LMDPs** and **Group Relative LMDPs**, two methods to incorporate symmetry into the provable linear MDP framework for the tabular and discrete environments. Both methods exploit structured redundancy to reduce effective feature dimension and accelerate learning without sacrificing regret guarantees.

---

## Repository Overview

- **`aggregated_feature_lmdp`**: Folder containing experimentation and visualizations for AF-LMDP setting
  - **`lincomblock.py`**: Code with linear combinatorial lock environment
  - **`experiment.py`**: Runs linear combinatorial lock environment experiments
  - **`experiment_gridworld.py`**: Self contained code to run Gridworld experiments (tabular setting with linear)
  - **`agent.py`**: Features all policies and functions called in lincomblock
  - **`batch_run.py`**: Runs all experiments
- **`requirements.txt`**: Dependencies required to run the project.
- **`RL_Linear_Markov_Decision_Processes.ipynb`**: GR-LMDP Code
- **`README.md`**: This document.

---

## Requirements

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `numpy`
- `numba`
- `matplotlib`

---

## Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ananya-Krishna/Group_Relative_LMDP.git
cd Group_Relative_LMDP
```

### Step 2: Running Experiments

#### AF-LMDP
```bash
cd aggregated_feature_lmdp
```
Run the `batch_run.py` notebook to conduct all experiments:

```bash
python batch_run.py
```

#### GR-LMDP
To explore the group relative approach, use the `RL_Linear_Markov_Decision_Processes.ipynb` notebook:

```bash
jupyter notebook RL_Linear_Markov_Decision_Processes.ipynb
```

#### Visualize Results
All scripts include code for graphs and visualizations.

---

### Results 

Aggregated-Feature LMDPs (AF-LMDPs) utilize symmetry in the environment by pooling features over group orbits, mapping equivalent state–action pairs to shared embeddings. This reduces the feature dimension from \(d\) to \(d_G\lld\), while preserving the information necessary for optimal control. Experimentally, AF-LMDPs achieved the lowest cumulative regret in small, highly structured environments such as the 4x4 GridWorld, where rotational and reflectional symmetries were fully exploited. These results confirm that group-invariant feature compression enhances both computational efficiency and sample efficiency. However, in environments like the linear combinatorial lock—where reward requires sustained exploration—AF-LMDPs were less effective due to limited structural regularity to exploit.

Group-Relative LMDPs (GR-LMDPs) improve computational efficiency by replacing exact dynamic programming with a Monte Carlo estimator of the value function. This approach reduces the computational complexity from exponential to polynomial time while maintaining unbiasedness. GR-LMDPs demonstrated strong performance in both small and large GridWorlds, consistently outperforming exact methods in terms of cumulative reward accumulation. In all experiments, GR-LMDPs enabled significantly faster learning curves, confirming that stochastic sampling is a viable and theoretically sound alternative to exact evaluation in large-scale episodic MDPs.

Looking forward, these findings suggest promising avenues for integrating symmetry-aware design into more general classes of reinforcement learning algorithms. Future work may also explore automatic symmetry discovery and its interaction with exploration strategies in unstructured or partially observable environments.

## Citation

If you find this work useful, please cite:

```
@article{krishna2025grouprelativelmdp,
  title={Dynamic Curvature Optimization for Hyperbolic GCNs},
  author={Krishna, Ananya and Simon, Valentina},
  journal={Yale University},
  year={2025}
}
```

---

## License
This project is licensed under the MIT License.

