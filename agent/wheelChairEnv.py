import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import serial
import serial.serialutil
import glob
import time

TARGET = np.array([10, 10])
VELOCITY = 1

class WheelChairEnv(gym.Env):
    """Wheelchair environment compatible with SB3, with simulation or Teensy hardware."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, simulation=False, port=None, baudrate=115200, timeout=1):
        super().__init__()
        self.simulation = simulation

        # Observation space: left, right, lift positions
        self.observation_space = spaces.MultiDiscrete([11, 11, 11])

        # Flattened action space: 3x3x3 → 27 discrete actions
        self.action_space = spaces.Discrete(27)

        self.state = np.zeros(3, dtype=np.int64)

        # --------------------------
        # Hardware setup
        # --------------------------
        self.ser = None
        if not self.simulation:
            if port is None:
                ports = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/tty.usbserial*')
                if not ports:
                    print("No Teensy serial device found. Switching to simulation mode.")
                    self.simulation = True
                else:
                    port = ports[0]

            if not self.simulation:
                try:
                    self.ser = serial.Serial(port, baudrate, timeout=timeout)
                    time.sleep(2)  # allow Teensy to initialize
                    print(f"Connected to Teensy on {port}")
                except serial.serialutil.SerialException:
                    print(f"Failed to open serial port {port}. Running in simulation mode.")
                    self.simulation = True
                    self.ser = None

    # =======================
    # Encode / Decode Actions
    # =======================
    def encode_action(self, action_array):
        """Convert [left, right, lift] in {-1,0,1} to flat index 0-26"""
        left, right, lift = action_array + 1
        return left * 9 + right * 3 + lift

    def decode_action(self, action_idx):
        """Convert flat index 0-26 → [left, right, lift] in {-1,0,1}"""
        left = action_idx // 9
        right = (action_idx % 9) // 3
        lift = action_idx % 3
        return np.array([left - 1, right - 1, lift - 1], dtype=np.int64)

    # =======================
    # Step function
    # =======================
    def step(self, action):
        self.step_count += 1
        truncated = self.step_count >= 100
        
        self.prev_state = self.state.copy()

        # Decode flat action
        left, right, lift = self.decode_action(action)

    # --------------------------
    # Hardware mode: Teensy simulator
    # --------------------------
        if not self.simulation and self.ser:
            # Send command
            self.ser.write(bytes([self._map_actuator(left),
                                self._map_actuator(right),
                                self._map_actuator(lift)]))

            # ---- Read exactly 3 bytes ----
            resp = bytearray()
            start_time = time.time()
            while len(resp) < 3:
                resp += self.ser.read(3 - len(resp))
                # Timeout after 0.1s
                if time.time() - start_time > 0.1:
                    break

            if len(resp) == 3:
                self.state = np.array(list(resp), dtype=np.int64)
            else:
                # fallback if incomplete read
                print(f"Serial warning: expected 3 bytes, got {len(resp)}. Using previous state")
                self.state = self.prev_state.copy()


        # --------------------------
        # Simulation mode
        # --------------------------
        if self.simulation:
            self.state += np.array([left, right, lift]) * VELOCITY
            # Clip if in simulation
            self.state = np.clip(self.state, 0, 10).astype(np.int64)


        # --------------------------
        # Reward shaping
        # --------------------------


        # Distance-based shaping (main signal)
        prev_dist = np.linalg.norm(self.prev_state[:2] - TARGET)
        new_dist  = np.linalg.norm(self.state[:2] - TARGET)
        reward = (prev_dist - new_dist)

        # Success bonus
        if (self.state[:2] == TARGET).all():
            reward = 10.0
            terminated = True
        else:
            terminated = False
        print(self.state)
        return self.state, reward, terminated, truncated, {}

    # =======================
    # Actuator mapping
    # =======================
    def _map_actuator(self, act):
        """Map -1,0,1 → Teensy byte"""
        if act == -1: return 0x00  # down
        if act == 0:  return 0x01  # stop
        if act == 1:  return 0x02  # up
        raise ValueError(f"Invalid actuator command: {act}")

    # =======================
    # Reset / Render
    # =======================
    def reset(self, seed=None, options=None):
        self.step_count = 0
        if options and "init_state" in options:
            self.state = np.array(options["init_state"], dtype=np.int64)
        else:
            self.state = np.array([random.randint(0, 10) for _ in range(3)], dtype=np.int64)

        # Push the new state to the Teensy
        if not self.simulation and self.ser:
            left_byte = self.state[0]
            right_byte = self.state[1]
            main_byte = self.state[2]
            self.ser.write(bytes([0xFF, left_byte, right_byte, main_byte]))

        return self.state.astype(int), {}

    def render(self):
        print(f"Agent Position: {self.state}")

    def close(self):
        if self.ser:
            self.ser.close()
