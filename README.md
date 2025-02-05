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
    * you can also use pyenv, some other python environment handler, or just insall python 3.12 to your machine
    * This probably will work with other versions of python, but it hasn't been tested

4. Install necessary packages
```
pip install -r requirements.txt
```


## Install Dolphin Emulator for GameCube and Wii
The game runs on the Dolphin emulator for GameCube and Wii games, and needs to be installed for it to work.

[Dolphin](https://github.com/dolphin-emu/dolphin) can be installed in many different ways. Initially I was going to try to script this within dolphin so I used this fork [here](https://github.com/Felk/dolphin). Both versions should work, as long as you can run dolphin from the terminal.

## Obtain NHL Hitz 2003 ROM
The model is trained to play NHL Hitz 2003 for the GameCube. The ROM that I have is a ```.rvz``` file. However, it should work with all formats. 

After obtaining your legally obtained ROM, make sure that dolphin can run it. Additionally, you can try this command that the environment uses to open the game: (Replace ```path/to/your/rom``` with the path to your rom location)

```
dolphin-emu  --exec path/to/your/rom
```

## 🤖 Run Pretrained Model 🤖
Note: The pretrained models I have are too large to host on GitHub, so you will need to train your own first.
1. Enter ```src/``` directory
```
cd src
```
2. Run ```run_pretrained_model.py``` file
```
python run_pretrained_model.py
```
3. To stop the run, press ```CTRL + C``` in the terminal

## 🦾 Train Your Own AI 🦾

1. Run ```train.py``` file
```
python train.py
```

### Tips For Training
Unless you have a very powerful computer, and a lot (and I mean A LOT) of time, I would recommend the following changes:
* decrease ```train_steps``` to reduce time
* decrease ```batch_size``` and/ or ```n_steps``` to decrease memory load
* increase ```action_frequency``` or decrease ```n_epochs``` to speed up training


## 💡 Built With 💡
This AI couldn't have been done without these amazing projects. Please check them out and support them!

### [Dolphin](https://github.com/Felk/dolphi)

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)

### [Dolphin Memory Engine](https://github.com/randovania/py-dolphin-memory-engine)


# Thanks for visiting! Continue the discussion in these awesome communities:
[![Discord Banner 2 Dolphin](https://invidget.switchblade.xyz/SUWqhYUVb4)](https://discord.gg/SUWqhYUVb4) 

[![Discord Banner 2 Reinforcement Learning](http://invidget.switchblade.xyz/pV8k2v6Fes)](https://discord.gg/pV8k2v6Fes)


<p align="center">
    <img alt="NHL Hitz 2003 GameCube Box Art" src="/assets/boxart.jpg" height="200" >
</p>