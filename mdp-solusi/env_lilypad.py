import numpy as np
import util as u

class EnvLilypad():
    def __init__(self, cfg):
        self.S, self.A, self.R = cfg['state_set'], cfg['action_set'], cfg['reward_set']
        self.nS, self.nA, self.nR = len(self.S), len(self.A), len(self.R)
        self.p0 = [cfg['initial_state_distrib'][s] for s in self.S]; assert len(self.p0) == self.nS
        self.p = {tuple(map(int, k.split(','))): v for k, v in cfg['state_transition_distrib'].items()}
        self.r = {tuple(map(int, k.split(','))): v for k, v in cfg['reward_distrib'].items()}
        assert sorted(set([k[0] for k in self.p.keys()])) == self.S
        assert sorted(set([k[1] for k in self.p.keys()])) == self.A
        assert sorted(set([k[2] for k in self.p.keys()])) == self.S
        assert sorted(set([k[0] for k in self.r.keys()])) == self.S
        assert sorted(set([k[1] for k in self.r.keys()])) == self.A
        assert sorted(set([k[2] for k in self.r.keys()])) == self.S
        assert sorted(set([k[3] for k in self.r.keys()])) == self.R
        assert sum(self.r.values()) == self.nS*self.nA*self.nS
        
    def get_initialstate(self):
        s0 = u.sample_discrete(self.p0)
        return s0

    def step(self, s, a):
        snext = self.sample_nextstate(s, a)
        return snext
                
    def sample_nextstate(self, s: int, a: int) -> tuple[int, int]:
        next_state_probs = [self.p[(s, a, s_next)] for s_next in range(self.nS)]
        snext = u.sample_discrete(next_state_probs)
    
        reward_probs = [self.r[(s, a, snext, r)] for r in self.R]
        reward_index = u.sample_discrete(reward_probs)
        rnext = self.R[reward_index]
    
        return snext, rnext

    
    def get_reward_mat(self) -> np.ndarray:
        r_mat = np.full((self.nS, self.nA), np.nan)
        
        for s in range(self.nS):
            for a in range(self.nA):
                expected_reward = 0.0
                for s_next in range(self.nS):
                    reward_given_sas = sum(r * self.r[(s, a, s_next, r)] for r in self.R)
                    expected_reward += self.p[(s, a, s_next)] * reward_given_sas
                r_mat[s, a] = expected_reward

        return r_mat
    