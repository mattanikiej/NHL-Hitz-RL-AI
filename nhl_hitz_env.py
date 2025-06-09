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
import cv2

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
        self.initial_state = config['state']
        self.window_name = config['window_name']
        self.dolphin_x = config['dolphin_x']
        self.dolphin_y = config['dolphin_y']
        self.frame_stack = config['frame_stack']

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
        self.obs_shape = (self.dolphin_y, self.dolphin_x, 3 * self.frame_stack)
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
            'hit': 5,              # Encourage physical play
            'pass': 1,             # Very small reward for passing
            'shot': 10,            # Encourage shooting
            'goal': 100,           # Big reward for scoring
            'opponent_goal': -150, # Big penalty for conceding
            'missed_pass': -3      # Penalty for missed passes
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

        # Initialize frame buffer
        self.frame_buffer = [np.zeros((self.dolphin_y, self.dolphin_x, 3), dtype=np.uint8) for _ in range(self.frame_stack)]


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

        reward_gain = self.update_rewards()
        truncated = self.check_period()

        # Capture new frame and update buffer
        new_frame = self.capture_dolphin()

        self.frame_buffer.pop(0)
        self.frame_buffer.append(new_frame)

        obs = self._get_obs()

        reward_breakdown = self.rewards.copy()
        for reward in reward_breakdown:
            reward_breakdown[reward] = reward_breakdown[reward] * self.reward_weights[reward]
        info = {'reward_breakdown': reward_breakdown}


        return obs, reward_gain, False, truncated, info


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
        
        initial_frame = self.capture_dolphin()
        self.frame_buffer = [initial_frame.copy() for _ in range(self.frame_stack)]
        return self._get_obs(), {}


    def render(self):
        """
        Renders the environments to help visualise what the agent see, 

        https://gymnasium.farama.org/api/env/

        :return: (list[int])
        """
        return self._get_obs(), {}


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


    def press_key(self, key):
        """
        Presses the given key

        :param key (Key): key to press
        """
        pgui.keyDown(key)
        time.sleep(0.001)
        pgui.keyUp(key)


    def button_key_press(self, action):
        """
        Presses the given button key

        :param action (float): action to take
        """

        self.press_key(self.button_actions[action])


    def movement_key_press(self, action):
        """
        Handles movement key presses based on the action value.
        Action values 0-7 correspond to 8 directional movements:
        0: right, 1: up-right, 2: up, 3: up-left, 4: left, 5: down-left, 6: down, 7: down-right

        Args:
            action (int): Integer between 0-7 representing the movement direction
        """
        # Define movement mappings for each direction
        movement_map = {
            0: ["right"],           # right
            1: ["right", "up"],     # up-right
            2: ["up"],              # up
            3: ["up", "left"],      # up-left
            4: ["left"],            # left
            5: ["left", "down"],    # down-left
            6: ["down"],            # down
            7: ["right", "down"]    # down-right
        }

        keys_to_press = set(movement_map[action])

        # Release keys that are pressed but not needed
        for i, (key, is_pressed) in enumerate(self.movement_actions):
            if is_pressed and key not in keys_to_press:
                pgui.keyUp(key)
                self.movement_actions[i][1] = False

        # Press keys that are needed and not already pressed
        for i, (key, is_pressed) in enumerate(self.movement_actions):
            if key in keys_to_press and not is_pressed:
                pgui.keyDown(key)
                self.movement_actions[i][1] = True


    def capture_dolphin(self):
        """
        Captures the dolphin window and returns an RGB numpy array.
        """
        t0 = time.time()
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
        img_array = np.frombuffer(screenshot.rgb, dtype=np.uint8).reshape((screenshot.height, screenshot.width, 3))
        img_resized = cv2.resize(img_array, (self.obs_shape[1], self.obs_shape[0]), interpolation=cv2.INTER_LINEAR)

        return img_resized


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
        self.total_rewards = new_rewards

        return reward_gain


    def _get_obs(self):
        return np.concatenate(self.frame_buffer, axis=2, dtype=np.uint8)
