from uuid import uuid4
import math
import subprocess
import time
import os
import signal

import numpy as np
from Xlib import display, X
import pyautogui as pgui
import mss.linux as mss
from PIL import Image

from gymnasium import Env, spaces

import dolphin_memory_engine as dme

import memory_constants as mem

class NHLHitzGymEnv(Env):
    """
    Gymnasium environment to be used by the model
    """
    def __init__(self, config=None):
        """
        Constructor for NHLHitzGymEnv
        
        :param config (dict): configuration settings for the environment
        """
        # check a config was passed in
        if config is None:
            raise Exception("Config needs to be set for NHLHitzGymEnv. Check configs.py for structure")
        
        # load in config values
        self.action_frequency = config['action_frequency']
        self.initial_state = config['state']
        self.window_name = config['window_name']
        self.dolphin_x = config['dolphin_x']
        self.dolphin_y = config['dolphin_y']

        self.id = str(uuid4())[:5]

        # initialize actions
        self.movement_actions = [
            ["right", False],
            ["up", False],
            ["left", False],
            ["down", False],
        ]

        self.button_actions = [
            "A",
            "B",
            "X",
            "Y",
            "Z",
            None
        ]

        # set gym attributes
        # space [8 directions, len(button_actions) buttons to press]
        self.action_space = spaces.MultiDiscrete([8, len(self.button_actions)])
        
        self.reward_range = (-math.inf, math.inf)
        
        # observation is all frames since previous action
        self.obs_shape = (self.dolphin_y, self.dolphin_x, self.action_frequency)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)

        # initialize reward structure
        self.rewards = {
            'hit': 0,
            'pass': 0,
            'shot': 0,
            'goal': 0,

            'opponent_goal': 0,
            'missed_pass': 0
        }

        self.reward_weights = {
            'hit': 1,
            'pass': 1,
            'shot': 2,
            'goal': 50,

            'opponent_goal': -1,
            'missed_pass': -1
        }

        self.total_rewards = 0

        # initialize dolphin emulator
        command = [
            "dolphin-emu",
            "--exec", "/home/macio/isos/NHL Hitz 2003 (USA).rvz",
            "--no-python-subinterpreters"
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.dolphin_pid = process.pid

        print(f"Dolphin Emulator started with PID {process.pid}.")

        # allow time for dolphin window to start up
        time.sleep(3)

        # hook into dolphin memory
        start = time.time()
        while not dme.is_hooked(): 
            dme.hook()
            end = time.time()
            if end - start > 5:
                print(f"hooking {end - start:.2f}")
                start = end
            
        # initialize window and keyboard capture
        self.window = self.get_dolphin_window()
        self.sct = mss.MSS()
            
        # start the game from initial state
        self.resets = 0
        self.reset()


    def step(self, action):
        """
        Updates an environment with actions returning the next agent observation, 
        the reward for taking that actions, if the environment has terminated or 
        truncated due to the latest action and information from the environment 
        about the step, i.e. metrics, debug info.

        https://gymnasium.farama.org/api/env/

        :param action (actType): action the model selected to run

        :return: (ObsType), (SupportsFloat), (bool), (bool), (dict)
        """
        self.act(action)

        self.steps += 1

        obs, _ = self.render()
        reward_gain = self.update_rewards()
        truncated = self.check_period()

        return obs, reward_gain, False, truncated, {}


    def reset(self, seed=None):
        """
        Resets the environment to an initial state, required before calling step. 
        Returns the first agent observation for an episode and information, 
        i.e. metrics, debug info.

        https://gymnasium.farama.org/api/env/

        :return: (ObsType), (dict)
        """
        # increment resets
        self.resets += 1

        # reset steps
        self.steps = 0

        # reset rewards
        for reward in self.rewards:
            self.rewards[reward] = 0

        self.total_rewards = 0

        # reset game state
        self.press_key("f1")  # load state
        self.press_key("A")  # skip intro
        

        return self.render()


    def render(self):
        """
        Renders the environments to help visualise what the agent see, 

        https://gymnasium.farama.org/api/env/

        :return: (list[int])
        """
        obs = np.zeros(self.obs_shape, dtype=np.uint8)
        for i in range(self.action_frequency):
            obs[:, :, i] = self.capture_dolphin()
        
        return obs, {}


    def close(self):
        """
        Closes the environment, important when external software is used, 
        i.e. dolphin for rendering

        https://gymnasium.farama.org/api/env/
        """
        print(f"Closing Dolphin Emulator with PID {self.dolphin_pid}.")
        os.kill(self.dolphin_pid, signal.SIGTERM)


    def act(self, action):
        """
        Sends the given action to the emulator

        :param action (actType): action to send to dolphin
        """
        movement_action, button_action = action

        self.movement_key_press(movement_action)

        # ensure button action is not None
        if button_action != len(self.button_actions) - 1:
            self.button_key_press(button_action)


    def check_period(self):
        """
        Checks if the period has ended

        :return: (bool)
        """
        if dme.read_byte(mem.PERIOD) > 1:
            return True
        return False
    

    def get_dolphin_window(self):
        """
        Finds the dolphin window and brings it into focus.
        Returns None if window not found

        :return: (Window)
        """
        d = display.Display()
        root = d.screen().root

        # Get all windows
        root_children = root.query_tree().children

        for window in root_children:
            # Get window title
            title = window.get_wm_name()
            if title and self.window_name in title:
                
                # Focus the window
                window.set_input_focus(X.RevertToNone, X.CurrentTime)
                d.sync()

                # Raise window to front
                window.configure(stack_mode=X.Above)
                d.sync()

                window_id = window.id
                window = d.create_resource_object('window', window_id)

                return window
    

    def focus_window(self):
        """
        Brings the dolphin window into focus
        """
        d = display.Display()
        self.window.set_input_focus(X.RevertToNone, X.CurrentTime)
        d.sync()

        self.window.configure(stack_mode=X.Above)
        d.sync()


    def press_key(self, key):
        """
        Presses the given key

        :param key (Key): key to press
        """
        # print(f"Pressing {key}")
        self.focus_window()
        pgui.keyDown(key)
        time.sleep(0.001)
        pgui.keyUp(key)


    def button_key_press(self, action):
        """
        Presses the given button key

        :param actino (float): action to take
        """

        self.press_key(self.button_actions[action])


    def movement_key_press(self, action):
        """
        releases unused movement keys and presses necessary keys

        :param action (float): action to take
        """
        self.focus_window()

        # i is left over from abysmal code and don't want to fix this since it works
        i = action

        # move right
        if i == 0:
            if not self.movement_actions[0][1]:
                self.movement_actions[0][1] = True
                pgui.keyDown("right")

            for j in range(len(self.movement_actions)):
                if j != 0:
                    if self.movement_actions[j][1]:
                        self.movement_actions[j][1] = False
                        pgui.keyUp(self.movement_actions[j][0])

        # move up-right
        elif i == 1:
            if not self.movement_actions[0][1]:
                self.movement_actions[0][1] = True
                pgui.keyDown("right")

            if not self.movement_actions[1][1]:
                self.movement_actions[1][1] = True
                pgui.keyDown("up")

            for j in range(len(self.movement_actions)):
                if j != 0 and j != 1:
                    if self.movement_actions[j][1]:
                        self.movement_actions[j][1] = False
                        pgui.keyUp(self.movement_actions[j][0])

        # move up
        elif i == 2:
            if not self.movement_actions[1][1]:
                self.movement_actions[1][1] = True
                pgui.keyDown("up")

            for j in range(len(self.movement_actions)):
                if j != 1:
                    if self.movement_actions[j][1]:
                        self.movement_actions[j][1] = False
                        pgui.keyUp(self.movement_actions[j][0])

        # move up-left 
        elif i == 3:
            if not self.movement_actions[1][1]:
                self.movement_actions[1][1] = True
                pgui.keyDown("up")

            if not self.movement_actions[2][1]:
                self.movement_actions[2][1] = True
                pgui.keyDown("left")

            for j in range(len(self.movement_actions)):
                if j != 1 and j != 2:
                    if self.movement_actions[j][1]:
                        self.movement_actions[j][1] = False
                        pgui.keyUp(self.movement_actions[j][0])

        # move left
        elif i == 4:
            if not self.movement_actions[2][1]:
                self.movement_actions[2][1] = True
                pgui.keyDown("left")

            for j in range(len(self.movement_actions)):
                if j != 2:
                    if self.movement_actions[j][1]:
                        self.movement_actions[j][1] = False
                        pgui.keyUp(self.movement_actions[j][0])

        # move down-left
        elif i == 5:
            if not self.movement_actions[2][1]:
                self.movement_actions[2][1] = True
                pgui.keyDown("left")

            if not self.movement_actions[3][1]:
                self.movement_actions[3][1] = True
                pgui.keyDown("down")

            for j in range(len(self.movement_actions)):
                if j != 2 and j != 3:
                    if self.movement_actions[j][1]:
                        self.movement_actions[j][1] = False
                        pgui.keyUp(self.movement_actions[j][0])

        # move down
        elif i == 6:
            if not self.movement_actions[3][1]:
                self.movement_actions[3][1] = True
                pgui.keyDown("down")

            for j in range(len(self.movement_actions)):
                if j != 3:
                    if self.movement_actions[j][1]:
                        self.movement_actions[j][1] = False
                        pgui.keyUp(self.movement_actions[j][0])

        # move down-right
        elif i == 7:
            if not self.movement_actions[3][1]:
                self.movement_actions[3][1] = True
                pgui.keyDown("down")

            if not self.movement_actions[0][1]:
                self.movement_actions[0][1] = True
                pgui.keyDown("right")

            for j in range(len(self.movement_actions)):
                if j != 3 and j != 0:
                    if self.movement_actions[j][1]:
                        self.movement_actions[j][1] = False
                        pgui.keyUp(self.movement_actions[j][0])
    

    def capture_dolphin(self):
        """
        Captures the dolphin window
        """
        geometry = self.window.get_geometry()
        padding = 20
        x, y, width, height = geometry.x, geometry.y, geometry.width, geometry.height

        monitor = {
            "top": y + padding,
            "left": x + padding,
            "width": width - 2*padding,
            "height": height - 2*padding
        }
    
        screenshot = self.sct.grab(monitor)

        # Save the screenshot
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img = img.convert("L")
        img_resized = img.resize((self.obs_shape[1], self.obs_shape[0]), Image.LANCZOS)
        return np.array(img_resized, dtype=np.uint8)


    def update_rewards(self):
        """
        Updates the rewards based on the memory values
        """
        self.rewards = {
            'hit': dme.read_byte(mem.HITS),
            'pass': dme.read_byte(mem.COMPLETED_PASSES),
            'shot': dme.read_byte(mem.SHOTS),
            'goal': dme.read_byte(mem.GOALS),

            'opponent_goal': dme.read_byte(mem.OPPONENT_GOALS),
            'missed_pass': dme.read_byte(mem.TOTAL_PASSES) - dme.read_byte(mem.COMPLETED_PASSES)
        }

        new_rewards = 0
        for reward in self.rewards:
            new_rewards += self.rewards[reward] * self.reward_weights[reward]

        reward_gain = new_rewards - self.total_rewards
        self.total_rewards += new_rewards

        return reward_gain
