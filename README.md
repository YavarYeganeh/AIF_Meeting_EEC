This repository provides the implementation and experiments associated with the paper **"Energy-Efficient Control of Parallel and Identical Machines Using Active Inference"**, authored by **Yavar Taheri Yeganeh**, **Mohsen Jafari**, and **Andrea Matta**. The paper was presented at the **2023 Active Inference Meeting**.

### Overview

The project investigates the application of **active inference** in controlling energy consumption for **parallel and identical machines** in manufacturing systems. The proposed approach integrates **deep learning** and **active inference** to develop **energy-efficient control (EEC) agents**. These agents optimize power consumption by balancing productivity and energy efficiency under uncertain and dynamic conditions.

The key challenges addressed in this project include:
- Managing power consumption across machines dynamically.
- Handling uncertainty in the environment using **active inference** principles.
- Using a hybrid horizon approach to manage short-term and long-term planning efficiently.

### Key Contributions
- **Deep Active Inference Agent:** We developed an agent capable of making energy-efficient control decisions in real time by predicting system states and adapting to the stochastic nature of machine operations.
- **Multi-Step Transitions:** To handle complex planning and delayed responses, the agent uses multi-step transitions that allow it to predict long-term outcomes and optimize accordingly.
- **Hybrid Horizon Approach:** The agent combines both short-term and long-term planning strategies to ensure optimal energy use without sacrificing throughput.
  
### Methodology

The agent's decision-making process is based on the **Free Energy Principle (FEP)**, leveraging **deep variational models** to estimate system behavior and minimize variational free energy. The model integrates multiple decision-making modules:
- **Encoder, Transition, Decoder Modules**: These neural networks help the agent learn state transitions, predict outcomes, and adjust its actions accordingly.
- **Monte Carlo Tree Search (MCTS):** Used to explore and evaluate potential future action sequences and optimize the policy.
  
### Performance


## Citation
Yeganeh, Y. T., Jafari, M., & Matta, A. (2024). Active Inference Meeting Energy-Efficient Control of Parallel and Identical Machines. arXiv preprint arXiv:2406.09322.
```
@article{yeganeh2024active,
  title={Active Inference Meeting Energy-Efficient Control of Parallel and Identical Machines},
  author={Yeganeh, Yavar Taheri and Jafari, Mohsen and Matta, Andrea},
  journal={arXiv preprint arXiv:2406.09322},
  year={2024}
}
```

## Contact

For inquiries or collaboration, please reach out to **yavar.taheri@polimi.it** or **yavar.yeganeh@gmail.com**.
