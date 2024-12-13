## Local-Cloud Inference Offloading for LLMs in Multi-Modal, Multi-Task, Multi-Dialogue Settings

[![python](https://img.shields.io/badge/Python_3.10-306998?logo=python&logoColor=FFD43B)](https://www.python.org/downloads/release/python-31012/)
[![License: MIT](https://img.shields.io/badge/license-MIT-750014.svg)](https://opensource.org/licenses/MIT) 

---

## 👨‍🍳 Use Case: Kitchen Activity Assistance with LLM

Imagine a scenario where an intelligent assistant helps you track activities in your kitchen:

- **User:** *Where did I put my dishes?*
- **RL Agent:** Processes first-person and overhead camera views -> Sends to Cloud LLM.
- **Cloud LLM:** *I see that you are in the kitchen. There are dishes on the counter next to the sink, likely just placed there after washing. Would you like me to turn on the dishwasher so you can put them away?*

<img src="figures/Local-Cloud.png" alt="overview" width="400" style="display: block; margin: auto;" />

---

## 🔥 Our Framework

We present **LCIO** (Local-Cloud Inference Offloading), a framework designed to optimize **response quality**, **latency**, and **cost efficiency** for LLMs in **multi-modal, multi-task, and multi-dialogue settings**. LCIO dynamically adapts to diverse conversational demands across tasks such as assistance, query, recommendation, and message editing. To enhance performance, we propose resource-constrained RL, which selects the best LLMs and modalities for inference, balancing quality, latency, and cost. RCRL also integrates user prompt associations with multi-modal data to effectively manage task connections in decision-making.

<img src="figures/RL.png" alt="overview" width="400"/>

---

## 🖥️ Prerequisites

Install the required packages via:
```bash
pip install -r requirements.txt
```

Alternatively, ensure the following dependencies are installed:
```plaintext
python == 3.10.12
numpy == 1.26.4
torch == 2.2.1
gymnasium == 0.28.1
stable_baselines3 == 2.2.1
scikit-learn == 1.5.1
```

---

## 📚 M4AI Dataset

We introduce **M4AI**, a comprehensive dataset capturing the **four multi-** elements **all in one** dataset:
1. **Multi-modal:** Includes three different view images.
2. **Multi-task:** Features four distinct tasks.
3. **Multi-dialogue:** Contains sequences of 2–5 dialogues.
4. **Multi-LLM:** Incorporates four LLMs tailored for different purposes.

<img src="figures/M4A1.png" alt="overview" width="400"/>

---

## 🗂️ Folder Structure
```
LCIO/
│   README.md
│   requirements.txt    
│
└─── main/
    │   main.py
    │   models.py
    └─── data/
        └─── M4A1_Dataset.json
```

- **`main/`**: Contains the primary codebase.
  - `main.py`: Entry point of the framework.
  - `models.py`: Includes model definitions such as `PPOLagrangian`, `A2CLagrangian`, and `DQNLagrangian`.
- **`data/`**: Stores the M4AI dataset.

---

## 🏃‍♂️‍➡️ Run Code

Run the framework with the following command:
```bash
python main.py --device='cuda:0'
```

---

## 🙏 Acknowledgement

The images in the M4A1 dataset are derived from the [ActionSense](https://action-sense.csail.mit.edu/) dataset.

