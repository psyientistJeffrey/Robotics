Cold trained **Mujoco humanoid-v5 model** within Openai gym environment, using **SB3 PPO2** and custom wrappers, including *callback*, *updating live-plot*, and *reward functions*.

The preliminary result was far from ideal, partialy due to ill-designed reward function and non-adaptive learning rate. It was also subject to limited computation, lack of pretraining data, inherent limited dof of the humanoid model.

#### 📁 Project Structure
```
Robotics/
├── implementation/          # Training scripts, custom wrappers, PPO2 config all in one.
│   └── spicy_vanilla.py     (# because its dedicated for a spicy walking humanoid, with basic config.)
│
├── Results/                 # Trained models, reward curves, logs
│   ├── shin_sprinter.zip
│   └── spicy_walker.zip
│
├── Eval/
│   └── analysis.ipynb       # model eval and detailed movement analysis
│
├── README.md                # Project overview and usage instructions
```
