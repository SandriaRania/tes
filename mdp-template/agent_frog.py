import itertools
import numpy as np
import util as u
from typing import Any

class AgentFrog():
    def __init__(self, cfg, S, A):
        assert cfg['policy_name'] == 'coin_tossing'
        self.S, self.A = S, A; self.nS = nS = len(S); self.nA = nA = len(A)
        jump_prob = cfg['policy_coin_tossing']['head_prob']; assert len(jump_prob) == nS
        self.pi_mat = np.vstack([[1 - jump_prob[s], jump_prob[s]] for s in range(nS)])
        assert self.pi_mat.shape == (nS, nA)
        assert self.pi_mat.sum() == nS
    
    def policy(self, s):
        a = u.sample_discrete(pvals=self.pi_mat[s])
        return a
    
    def get_one_step_state_transition_matrix_of_a_policy(self, p: dict, pi_mat: np.ndarray) -> np.ndarray:
        P_pi = np.full((self.nS, self.nS), np.nan)
        ##### Begin: write your code below #####
        # TODO complete this chunk
        ##### End of your code #####
        return P_pi
    
    def get_reward_vector_of_a_policy(self, r_mat: np.ndarray, pi_mat: np.ndarray) -> np.ndarray:
        assert (self.nS, self.nA) == r_mat.shape
        r_vec = np.full(self.nS, np.nan)
        ##### Begin: write your code below #####
        # TODO complete this chunk
        ##### End of your code #####
        return r_vec
    
    def get_pstar_matrix_for_unichain(self, P_pi: np.ndarray) -> np.ndarray:
        nS = self.nS; assert (nS, nS) == P_pi.shape
        Pstar_mat = np.full(nS, np.nan)
        ##### Begin: write your code below #####
        # TODO complete this chunk
        ##### End of your code #####
        return Pstar_mat
    
    def evaluate_a_policy_wrt_average_reward(self, P_pi: np.ndarray, r_pi: np.ndarray, unused=None) -> np.ndarray:
        assert self.nS == r_pi.shape[0]
        v_gain = np.full(self.nS, np.nan)
        ##### Begin: write your code below #####
        # TODO complete this chunk
        ##### End of your code #####
        return v_gain
    
    def evaluate_a_policy_wrt_discounted_reward(self, P_pi: np.ndarray, r_pi: np.ndarray, gamma: float) -> np.ndarray:
        assert self.nS == r_pi.shape[0]
        v_gamma = np.full(self.nS, np.nan)
        ##### Begin: write your code below #####
        # TODO complete this chunk
        ##### End of your code #####
        return v_gamma

    def get_all_stationary_deterministic_policies(self) -> list[np.ndarray]:
        pi_mat_list = []
        ##### Begin: write your code below #####
        # TODO complete this chunk
        ##### End of your code #####
        return pi_mat_list
    
    def exhaustive_search_over_all_stationary_deterministic_policies(self, 
            p0: list[float], p: dict, r_mat: np.ndarray, value_fn: Any, gamma: float) -> tuple[list[np.ndarray], float]:
        optimal_policies = []; optimal_value = None
        ##### Begin: write your code below #####
        # TODO complete this chunk
        ##### End of your code #####
        return optimal_policies, optimal_value
    
