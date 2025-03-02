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
        snext, rnext = None, None
        ##### Begin: write your code below #####
        # TODO complete this chunk
        nstate, probs = zip(*[(k[2], v) for k, v in self.p.items() if k[0] == s and k[1] == a])
        probs = np.array(probs) / sum(probs)  
        snext = np.random.choice(nstate, p=probs)
        rnext = sum(k[3] * v for k, v in self.r.items() if k[0] == s and k[1] == a and k[2] == snext)
        ##### End of your code #####
        return snext, rnext # the nextstate and the reward value (instead of the reward idx)
    
    def get_reward_mat(self) -> np.ndarray:
        r_mat = np.full((self.nS, self.nA), np.nan)
        ##### Begin: write your code below #####
        # TODO complete this chunk
        r_mat = np.zeros((self.nS, self.nA))
        for s in self.S:
            for a in self.A:
                exp_reward = 0.0  

                for s_next in self.S:  
                    exp_r = 0.0  

                    for r_val in self.R: 
                        key = (s, a, s_next, r_val)
                        if key in self.r:
                            exp_r += r_val * self.r[key] 

                    if (s, a, s_next) in self.p:  
                        exp_reward += self.p[(s, a, s_next)] * exp_r  
            r_mat[s, a] = exp_reward
        ##### End of your code #####
        return r_mat
    