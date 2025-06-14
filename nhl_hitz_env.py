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
    Gymnasium-compatible environment for NHL Hitz 2003 using the Dolphin Emulator.
    
    This environment interacts with the emulator, captures game frames, sends actions, and computes rewards for reinforcement learning agents.
    
    Key features:
    - Supports frame stacking for temporal observations.
    - Allows configurable action frequency (agent acts every N frames).
    - Custom reward structure encouraging passing before shooting.
    """
    def __init__(self, config=None):
        """
        Initialize the NHLHitzGymEnv environment.

        Args:
            config (dict): Configuration settings for the environment. Must include:
                - 'action_frequency' (int): Number of frames between agent actions.
                - 'state' (str): Path to initial emulator state.
                - 'window_name' (str): Name of the Dolphin window.
                - 'dolphin_x' (int): Width of the emulator window.
                - 'dolphin_y' (int): Height of the emulator window.
                - 'frame_stack' (int): Number of frames to stack for observations.
        Raises:
            Exception: If config is not provided.
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
        self.frame_stack = config['frame_stack']

        self.steps = 0

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
            'shot': 5,             # Encourage shooting
            'goal': 100,           # Big reward for scoring
            'opponent_goal': -100, # Big penalty for conceding
            'missed_pass': -3      # Penalty for missed passes
        }

        # initialize consecutive passes
        self.consecutive_passes = 0
        self.consecutive_pass_bonus = 5  # Bonus reward weight for consecutive passes before a shot
        self.max_consecutive_passes = 5  # Max number of consecutive passes to reward

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
        Take a step in the environment.

        The agent's action is only executed every `action_frequency` frames; otherwise, the previous action is repeated or no action is taken.
        Captures a new frame, updates the frame buffer, computes the reward, and returns the observation and info.

        Args:
            action (tuple): The action selected by the agent (movement, button).

        Returns:
            obs (np.ndarray): The stacked observation after the step.
            reward (float): The reward for this step.
            terminated (bool): Whether the episode has ended (always False here).
            truncated (bool): Whether the episode was truncated (e.g., period ended).
            info (dict): Additional info, including reward breakdown.
        """
        self.act(action)

        self.steps += 1

        reward_gain = self.update_rewards()
        truncated = self.check_period()

        # advance 5 frames
        for _ in range(5):
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
        Reset the environment to the initial state.

        Loads the initial emulator state, resets counters and rewards, and returns the initial stacked observation.

        Args:
            seed (int, optional): Random seed (unused).

        Returns:
            obs (np.ndarray): The initial stacked observation.
            info (dict): Additional info (empty dict).
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
        Render the current environment observation.

        Returns:
            obs (np.ndarray): The current stacked observation.
            info (dict): Additional info (empty dict).
        """
        return self._get_obs(), {}


    def close(self):
        """
        Close the environment and terminate the Dolphin Emulator process.
        """
        print(f"Closing Dolphin Emulator with PID {self.dolphin_pid}.")
        os.kill(self.dolphin_pid, signal.SIGTERM)


    def act(self, action):
        """
        Send the given action to the emulator.

        Args:
            action (tuple): (movement_action, button_action) to send to Dolphin.
        """
        movement_action, button_action = action

        self.movement_key_press(movement_action)

        # ensure button action is not None
        if button_action != len(self.button_actions) - 1:
            self.button_key_press(button_action)


    def check_period(self):
        """
        Check if the current period in the game has ended.

        Returns:
            bool: True if the period has ended, False otherwise.
        """
        if dme.read_byte(mem.PERIOD) > 1:
            return True
        return False
    

    def get_dolphin_window(self):
        """
        Find and focus the Dolphin Emulator window by name.

        Returns:
            window (Xlib.display.Window): The Dolphin window object, or None if not found.
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
        Simulate a key press and release using pyautogui.

        Args:
            key (str): The key to press.
        """
        pgui.keyDown(key)
        time.sleep(0.001)
        pgui.keyUp(key)


    def button_key_press(self, action):
        """
        Press the specified button action in the emulator.

        Args:
            action (int): Index of the button action to press.
        """

        self.press_key(self.button_actions[action])


    def movement_key_press(self, action):
        """
        Handle movement key presses based on the action value.

        Action values 0-7 correspond to 8 directional movements:
            0: right, 1: up-right, 2: up, 3: up-left, 4: left, 5: down-left, 6: down, 7: down-right

        Args:
            action (int): Integer between 0-7 representing the movement direction.
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
        Capture the current frame from the Dolphin Emulator window and return as an RGB numpy array.

        Returns:
            np.ndarray: The captured and resized RGB frame.
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
        img_array = np.frombuffer(screenshot.rgb, dtype=np.uint8).reshape((screenshot.height, screenshot.width, 3))

        img_resized = cv2.resize(img_array, (self.obs_shape[1], self.obs_shape[0]), interpolation=cv2.INTER_LINEAR)

        return img_resized


    def update_rewards(self):
        """
        Update the reward structure based on current game memory values.

        Implements a bonus for shooting after consecutive passes, up to a maximum streak. Resets the bonus on missed pass or shot.

        Returns:
            float: The reward gained since the last update.
        """
        # Read current memory values
        hits = dme.read_byte(mem.HITS)
        completed_passes = dme.read_byte(mem.COMPLETED_PASSES)
        shots = dme.read_byte(mem.SHOTS)
        goals = dme.read_byte(mem.GOALS)
        opponent_goals = dme.read_byte(mem.OPPONENT_GOALS)
        total_passes = dme.read_byte(mem.TOTAL_PASSES)
        missed_passes = total_passes - completed_passes

        # Calculate deltas
        prev_rewards = self.rewards.copy()
        self.rewards = {
            'hit': hits,
            'pass': completed_passes,
            'shot': shots,
            'goal': goals,
            'opponent_goal': opponent_goals,
            'missed_pass': missed_passes
        }

        # Track events
        pass_delta = self.rewards['pass'] - prev_rewards['pass']
        shot_delta = self.rewards['shot'] - prev_rewards['shot']
        missed_pass_delta = self.rewards['missed_pass'] - prev_rewards['missed_pass']

        # Update consecutive passes
        if pass_delta > 0:
            # get the number of passes since the last shot or missed pass
            self.consecutive_passes = min(self.consecutive_passes + pass_delta, self.max_consecutive_passes)
        if missed_pass_delta > 0 or shot_delta > 0:
            # reset bonus if a shot or missed pass occurs
            self.consecutive_passes = 0

        # Calculate reward
        new_rewards = 0
        for reward in self.rewards:
            if reward == 'shot' or reward == 'pass':
                # Add bonus for consecutive passes
                bonus = self.consecutive_pass_bonus * self.consecutive_passes
                new_rewards += self.rewards[reward] * (self.reward_weights[reward] + bonus)
            else:
                new_rewards += self.rewards[reward] * self.reward_weights[reward]

        reward_gain = new_rewards - self.total_rewards
        self.total_rewards = new_rewards

        return reward_gain


    def _get_obs(self):
        """
        Get the current stacked observation for the agent.

        Returns:
            np.ndarray: The stacked frames as a single observation (shape: height x width x 3*frame_stack).
        """
        return np.concatenate(self.frame_buffer, axis=2, dtype=np.uint8)
