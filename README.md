## Local-Cloud Inference Offloading for LLMs in Multi-Modal, Multi-Task, Multi-Dialogue Settings

[![python](https://img.shields.io/badge/Python_3.10-306998?logo=python&logoColor=FFD43B)](https://www.python.org/downloads/release/python-31012/)
[![License: MIT](https://img.shields.io/badge/license-MIT-750014.svg)](https://opensource.org/licenses/MIT) 

-----------------------------------------------------------------------------------------------

## 👨‍🍳 Use Case: Kitchen Activity Assistance with LLM

![overview](Fig/Local-Cloud.png)

- **User:** Where did I put my dishes?.
- **RL:** First-person view and overhead view -> Cloud LLM
- **Cloud LLM:** I see that you are currently in the kitchen, and there are some dishes on the counter next to the sink. It appears that you may have just put them there after washing them. Would you like me to turn on the dishwasher so you can put them away? 


## 🔥 Our Framework

![overview](Fig/RL.png)

We introduce LCIO, a local-cloud LLM inference offloading system designed to optimize response quality, latency, and usage costs in multi-modal, multi-task, and multi-dialogue scenarios. LCIO dynamically adapts to diverse conversational demands across tasks such as assistance, query, recommendation, and message editing. To enhance performance, we propose resource-constrained RL, which selects the best LLMs and modalities for inference, balancing quality, latency, and cost. RCRL also integrates user prompt associations with multi-modal data to effectively manage task connections in decision-making.

## 🖥️ Prerequisites
Please download packages via `pip install -r requirements.txt` or below
```
* python == 3.10.12
* numpy==1.26.4
* torch==2.2.1
* gymnasium==0.28.1
* stable_baselines3==2.2.1
* sklearn==1.5.1
```

## 📚 M4AI Dataset

We propose and generate a new dataset termed M4A1, which considers multi-modal, multi-task, multi-dialogue, and multi-LLM characteristics, encapsulating these four ``multi'' elements all in one dataset. It includes (i) three different view images, (ii) four distinct tasks, (iii) two to five sequential dialogues, and (iv) four LLMs for different purposes.

![overview](Fig/M4A1.png)

## 🗂️ Folder Structure
```
DCFL/
│   README.md
│   requirements.txt    
│
└─── main/
    │   main.py
    │   models.py
    └─── data/
        └─── M4A1_Dataset.json
```

- `main/` the main folder
-- `main.py` is our framework code
-- `models.py` is models' code, including PPOLagrangian, A2CLagrangian, and DQNLagrangian
- `data/` stores the M4A1 dataset

## 🏃‍♂️‍➡️ Run Code
```
python main.py --device='cuda:0'
```

## 🙏 Acknowledgement

The original implementations of DDPM and UNet are sourced from [labml_nn](https://nn.labml.ai/diffusion/ddpm/index.html).




