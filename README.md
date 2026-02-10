## RL-Training

RL-Training contains two directories:

### Agent  
- **train.py**  
  Master script that uses SB3 to train an agent in the Wheelchair Environment.
  (Note there is no agent.py as SB3 usees a built in agent depnding on learning model)
- **trainingCallBack**
  Class that's passed to model to log resulting q-table and episode data
  

- **wheelChairEnv.py**  
  Contains code for the Wheelchair environment:
  - **Action state:**  
    3 controls `{ -1, 0, 1 }` for the left paddle, right paddle, and main lift actuators
  - **Observation state:**  
    Uses `main.cpp` to simulate an environment where we record the position of the three actuators `[0, 10]`

### Controller
- **src/main.cpp**  
  Functions as an environment simulator:
  - Contains information about the observation state
  - Interfaces with `wheelChairEnv.py`’s `step` function to:
    - Read in actions
    - Update the environment
    - Transmit the new environment back to `wheelChairEnv.py`
  - **Goal:**  
    Make `main.cpp` actually send the commands to the Niko robot and return updated hull sensing
