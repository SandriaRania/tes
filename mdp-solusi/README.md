![version](https://img.shields.io/badge/version-release_1-blue)

# MDP: Markov Decision Process
Begin with reading the code of conduct and the problem description below, 
then follow the instructions presented thereafter.
For some bonus, go to [mdp-bonus](../mdp-bonus).

# Code of conduct
* This lab work is individual
  * Any doubt/question should be discussed with TA in-person during lab session.
    Note that there is no online lab session.
* You may open any resources, including generative AI but it should be used sparingly.
  You are expected to write down the code yourself. 
  Do not copy paste code from the chatbot's responses.
* Breaking any rule above will **cancel all the labs** worth at least 21 points.
  Note that here, the burden of proving innocent is on the presumably offender (the student).
* Submit by either one of these two deadlines specified below. 
  * For those who attend the lab session in-person: 
    * Deadline: 19 Feb at 17:30 WIB (= 16:40 WIB + 50 minutes extra time)
    * By attending, we mean coming to the lab by 15:10 WIB,
      taking some reasonable break (eg for praying Asar),
      coming back, then spending the remaining time in the lab.
    * Make sure you get the signatures `Attendance: OK` from TA on your PR for your attendance.
      There will be several attendance check points, hence multiple signatures.
  * For those who **do not** attend the lab session in-person: 
    * Deadline: 19 Feb at 16:40 WIB (at the end of the lab session)
  * Late submission will not be accepted.
  

# The problem: A frog on lily pads
![frog_on_lilypads.png](./frog_on_lilypads.png)

Credits to `Levin, 2009` for the inspiration and the picture of the frog on lily pads.

A certain frog lives in a pond with a number lily pads, say two: east and west lily pads.
A long time ago, he found two coins at the bottom of the pond and brought one up to each lily pad. 
Given the source of the coins, we should not assume that they are fair! 

Every morning, the frog decides whether to jump by tossing the current lily pad's coin. 
If the coin lands heads up, the frog jumps toward the other lily pad. 
Because of the morning wind over the pond, the jumping frog may land on the other lily pad only with some probability.
The jumping frog may also land on where he was (just before jumping) with some probability.
If the coin lands tails up, the frog aims to stay where he is,
but again the morning wind may blow him to the other lily pad with some probability.

After executing his decision, the frog spends the whole day hunting mosquitos from the current lily pad.
The number of daily catches varies with some probabilities, which are different on different lily pads.
The frog losses energy when there is no successful catch, worse if he jumps then no catch.
He may also loss energy even after eating mosquitos because 
they are so tiny that the intake energy does not compensate the energy for hunting them.
This often happens in cold weather.
The frog rewards himself based on the stochastic net energy, ie. IN minus OUT, at the end of each day.
It may be -1, 0, 1, or 2 joules.

The frog wants you to help him quantify the quality of his current coin tossing policy.
He also asks you to determine whether his current coin tossing policy is optimal. 
If not, then he expects you to tell him the optimal ones.
Note that any decision (based on coin tossing) is taken only in the morning.
Only the morning wind is strong enough to affect the frog's location.
The frog sleeps during the night.

# Setup the Python environment
```
conda create --name mdp python=latest
conda activate mdp
pip install -r requirements.txt
```

# Study the code structure
The code is structured into 3 main files: `env_lilypad.py`, `agent_frog.py`, and `xprmt.py`,
which reflect major entities in MDP modeling and experiment/simulation for RL.
Each main file has its own config YML file. 
There is also a utility `util.py`, which should not be modified.

The entry point for experiment is `xprmt.py` along with its example config `xprmt_example.yml`.
Running `xprmt.py` generates `xprmt_log.yml`.
We provide an example `xprmt_log.yml`, which was generated using our solution code and example configs.
An example terminal output can be seen in `xprmt_out.txt`.

The `xprmt.py` already contains the complete code.
It may be helpful to review `xprmt.py`.
Do not modify the `xprmt.py`.
You may copy the `xprmt_example.yml`, then modify the copy as necessary.

It may also be helpful to review `env_lilypad.py` and `agent_frog.py`, eventhough they are incomplete.
Your job is to complete several methods in those 2 files (see the next section for more details).
The method signatures are (hopefully) self-explained and intuitive.
You may copy the `env_lilypad_example.yml` and `agent_frog_example.yml`, then modify their copies as necessary.

# Complete the following 9 methods
1. Method `get_one_step_state_transition_matrix_of_a_policy` in `agent_frog.py`.
2. Method `get_reward_mat` in `env_lilypad.py` 
3. Method `get_reward_vector_of_a_policy` in `agent_frog.py`.
4. Method `get_pstar_matrix_for_unichain` in `agent_frog.py`. 
5. Method `evaluate_a_policy_wrt_average_reward` in `agent_frog.py`.
6. Method `evaluate_a_policy_wrt_discounted_reward` in `agent_frog.py`.
7. Method `get_all_stationary_deterministic_policies` in `agent_frog.py`.
8. Method `exhaustive_search_over_all_stationary_deterministic_policies` in `agent_frog.py`.
9. Method `sample_nextstate` in `env_lilypad.py`

# Grading
We will run your `agent_frog.py` and `env_lilypad.py` using many different RNG seeds, 
then compare the resulting `xprmt_log.yml` against that of our solution code.
Those various seeds are used to generate several versions of agent and environment configs.
They are also used to randomize the agent-environment interaction during experiments.
