# NHL Hitz 2003 Reinforcement Learning AI
<p align="center">
    <img alt="NHL Hitz 2003 Logo" src="assets/nhl-hitz-logo.jpg" height="200">
</p>

__This is a reinforcement learning artificial intelligence model built to play the GameCube version of NHL Hitz 2003 using the Dolphin Emulator.__

Note: This was built on a Linux system, and may not work with other operating systems without some changes (mainly the window focusing).

## 👾 Getting Started 👾
You need to set up the Python evnironment to be able to run the model.
1. Clone this repository
```
git clone https://github.com/mattanikiej/NHL-Hitz-RL-AI
```

2.  Enter repository
```
cd NHL-Hitz-RL-AI
```

3. Install correct python version: 3.12
    * if using anaconda, you can create a new environment by running these commands in terminal
    ```
    conda create --name nhl-hitz python=3.12
    conda activate nhl-hitz
    ```
    * you can also use pyenv, some other python environment handler, or just install python 3.12 to your machine
    * This probably will work with other versions of python, but it hasn't been tested

4. Install necessary packages
```
pip install -r requirements.txt
```


## 🐬 Install Dolphin Emulator for GameCube and Wii 🐬
The game runs on the Dolphin emulator for GameCube and Wii games, and needs to be installed for it to work.

[Dolphin](https://github.com/dolphin-emu/dolphin) can be installed in many different ways. Initially I was going to try to script this within dolphin so I used this fork [here](https://github.com/Felk/dolphin). Both versions should work, as long as you can run dolphin from the terminal.

## 🏒 Obtain NHL Hitz 2003 ROM 🏒
The model is trained to play NHL Hitz 2003 for the GameCube. The ROM that I have is a ```.rvz``` file. However, it should work with all formats. 

After obtaining your legally obtained ROM, make sure that dolphin can run it. Additionally, you can try this command that the environment uses to open the game: (Replace ```path/to/your/rom``` with the path to your rom location)

```
dolphin-emu  --exec path/to/your/rom
```

## 🤖 Run Pretrained Model 🤖
1. Download the latest model from the releases tab

2. Run the `run_pretrained_model.py` script with your model path:
```
python run_pretrained_model.py --model-path path/to/your/model.zip
```

### 📝 Arguments for `run_pretrained_model.py` 📝

The `run_pretrained_model.py` script supports the following command-line arguments:

| Argument      | Type   | Default           | Description                                                                                   |
|---------------|--------|-------------------|-----------------------------------------------------------------------------------------------|
| `--model-path`| str    | 'saved_models/hawksai-model'| Path to the saved PPO model file (.zip format)                                               |
| `--steps`     | int    | 500               | Number of steps for the model to take. Use -1 for infinite steps.                            |

**Example usage:**
```
# Run with default settings (hawksai-model for 500 steps)
python run_pretrained_model.py

# Run a specific model for 1000 steps
python run_pretrained_model.py --model-path saved_models/my_model --steps 1000

# Run a model indefinitely
python run_pretrained_model.py --model-path saved_models/my_model --steps -1
```



## 🦾 Train Your Own AI 🦾

1. Run ```train.py``` file with default settings
```
python train.py
```

2. To continue training you can pass the ```--train-pretrained``` argument, along with the ```--session_id``` argument to continue training a pretrained model.
```
python train.py --session_id=YOUR_MODEL --train-pretrained=True
```

### 📝 Arguments for `train.py` 📝

The `train.py` script supports the following command-line arguments:

| Argument            | Type   | Default                | Description                                                                                   |
|---------------------|--------|------------------------|-----------------------------------------------------------------------------------------------|
| `--session_id`      | str    | Random 5-char string   | Unique identifier for the training session. Used for saving checkpoints and model files.       |
| `--train-pretrained`| bool   | False                  | If `True`, continues training from a pretrained model specified by `session_id`.              |
| `--save-model`      | bool   | True                   | If `True`, saves the final model after training.                                              |

**Example usage:**
```
python train.py --session_id=hawksai --train-pretrained=True --save-model=True
```

### 🤓 Tips For Training 🤓
Unless you have a very powerful computer, and a lot (and I mean A LOT) of time, I would recommend the following changes:
* decrease ```train_steps``` to reduce time
* decrease ```batch_size``` and/ or ```n_steps``` to decrease memory load
* decrease ```n_epochs``` to speed up training
* decrease ```action_frequency``` to increase the amount of actions it takes per second


## 📁 Repository Directory Structure 📁

Here's a quick overview of the main files and folders in this repository:

```
NHL-Hitz-RL-AI/
├── assets/                             # Contains images and media files
├── callbacks/                          # Custom callback implementations for training
│   ├── episode_metrics_logger.py
│   └── reward_breakdown_callback.py
├── checkpoints/                        # Training checkpoints directory
├── saved_models/                       # Directory for saved trained models
├── save_states/                        # Game state saves
├── nhl_hitz_env.py                     # Main environment implementation
├── train.py                            # Training script
├── run_pretrained_model.py             # Script to run pretrained models
├── configs.py                          # Configuration settings
├── memory_constants.py                 # Memory address constants
├── check_env.py                        # Environment verification script
├── requirements.txt                    # Python dependencies
```


## 💡 Built With 💡
This AI couldn't have been done without these amazing projects. Please check them out and support them!

### [Dolphin](https://github.com/Felk/dolphi)

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)

### [Dolphin Memory Engine](https://github.com/randovania/py-dolphin-memory-engine)


# 🥅 Thanks for visiting! Check out these awesome communities: 🥅
[![Discord Banner 2 Dolphin](https://invidget.switchblade.xyz/SUWqhYUVb4)](https://discord.gg/SUWqhYUVb4) 

[![Discord Banner 2 Reinforcement Learning](http://invidget.switchblade.xyz/pV8k2v6Fes)](https://discord.gg/pV8k2v6Fes)


<p align="center">
    <img alt="NHL Hitz 2003 GameCube Box Art" src="assets/boxart.jpg" height="300" >
</p>