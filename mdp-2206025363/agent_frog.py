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
        for s in range(self.nS):
            for s_next in range(self.nS):
                P_pi[s, s_next] = sum(
                    pi_mat[s, a] * p.get((s, a, s_next), 0)
                    for a in range(self.nA)
                )
        ##### End of your code #####
        return P_pi
    
    def get_reward_vector_of_a_policy(self, r_mat: np.ndarray, pi_mat: np.ndarray) -> np.ndarray:
        assert (self.nS, self.nA) == r_mat.shape
        r_vec = np.full(self.nS, np.nan)
        ##### Begin: write your code below #####
        # TODO complete this chunk
        for s in range(self.nS):
            r_vec[s] = np.sum(pi_mat[s] * r_mat[s])
        ##### End of your code #####
        return r_vec
    
    def get_pstar_matrix_for_unichain(self, P_pi: np.ndarray) -> np.ndarray:
        nS = self.nS; assert (nS, nS) == P_pi.shape
        Pstar_mat = np.full((nS, nS), np.nan)
        ##### Begin: write your code below #####
        # TODO complete this chunk
        eigenvalues, eigenvectors = np.linalg.eig(P_pi.T)
        stationary_vector = eigenvectors[:, np.isclose(eigenvalues, 1)].real[:, 0]
        stationary_distribution = stationary_vector / np.sum(stationary_vector)
        Pstar_mat[:] = stationary_distribution
        ##### End of your code #####
        return Pstar_mat
    
    def evaluate_a_policy_wrt_average_reward(self, P_pi: np.ndarray, r_pi: np.ndarray, unused=None) -> np.ndarray:
        assert self.nS == r_pi.shape[0]
        v_gain = np.full(self.nS, np.nan)
        ##### Begin: write your code below #####
        # TODO complete this chunk
        nS = self.nS
        A = np.eye(nS) - P_pi + np.ones((nS, nS)) 
        b = r_pi
        v_gain = np.linalg.lstsq(A, b, rcond=None)[0] 
        ##### End of your code #####
        return v_gain
    
    def evaluate_a_policy_wrt_discounted_reward(self, P_pi: np.ndarray, r_pi: np.ndarray, gamma: float) -> np.ndarray:
        assert self.nS == r_pi.shape[0]
        v_gamma = np.full(self.nS, np.nan)
        Pgamma_pi = np.linalg.inv(np.identity(self.nS) - gamma*P_pi)
        v_gamma = np.matmul(Pgamma_pi, r_pi)
        return v_gamma

    def get_all_stationary_deterministic_policies(self) -> list[np.ndarray]:
        pi_mat_list = []
        ##### Begin: write your code below #####
        # TODO complete this chunk
        for policy in itertools.product(range(self.nA), repeat=self.nS):
            pi_mat = np.zeros((self.nS, self.nA))
            for s, a in enumerate(policy):
                pi_mat[s, a] = 1  
            pi_mat_list.insert(0, pi_mat)
        ##### End of your code #####
        return pi_mat_list
    
    def exhaustive_search_over_all_stationary_deterministic_policies(self, 
            p0: list[float], p: dict, r_mat: np.ndarray, value_fn: Any, gamma: float) -> tuple[list[np.ndarray], float]:
        optimal_policies = []; optimal_value = None; value_wrt_p0_list = []
        pi_mat_list = self.get_all_stationary_deterministic_policies()
        for pi_mat in pi_mat_list:
            P_pi = self.get_one_step_state_transition_matrix_of_a_policy(p, pi_mat)
            r_pi = self.get_reward_vector_of_a_policy(r_mat, pi_mat)
            v = value_fn(P_pi, r_pi, gamma)
            v_wrt_p0 = np.dot(v, p0)
            value_wrt_p0_list.append(v_wrt_p0.item())
        optimal_value = max(value_wrt_p0_list)
        optimal_policies = [pi_mat_list[pidx] for pidx, gain in enumerate(value_wrt_p0_list) if gain == optimal_value]
        return optimal_policies, optimal_value
