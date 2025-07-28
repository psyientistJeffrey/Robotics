Cold trained **Mujoco humanoid-v5 model** within Openai gym environment, using **SB3 PPO2** and custom wrappers, including *callback*, *updating live-plot*, and *reward functions*.

#### 📁 Project Structure
```
Robotics/
├── implementation/          # Training scripts, custom wrappers, PPO2 config all in one.
│   └── spicy_vanilla.py     (# because its dedicated for a spicy walking humanoid, with basic config.)
├── Results/                 # Trained models, reward curves, logs
│   ├── shin_sprinter.zip
│   └── spicy_walker.zip    
├── Eval/
│   └── analysis.ipynb       # model eval and detailed movement analysis
├── README.md                # Project overview and usage instructions
```
