import sys, random
import numpy as np, yaml
import util as u
from env_lilypad import EnvLilypad
from agent_frog import AgentFrog

def main():
    assert len(sys.argv) == 2, f'Usage: python xprmt.py xprmt_config.yml'
    xcfg = u.read_cfg(sys.argv[1]); log = {'xcfg': xcfg}
    np.random.seed(xcfg['seed']); random.seed(xcfg['seed'])

    # Init the agent and env
    env = EnvLilypad(xcfg['env'])
    agent = AgentFrog(xcfg['agent'], S=env.S, A=env.A)
    P_pi = agent.get_one_step_state_transition_matrix_of_a_policy(env.p, agent.pi_mat)
    r_pi = agent.get_reward_vector_of_a_policy(env.get_reward_mat(), agent.pi_mat)
    Pstar_pi = agent.get_pstar_matrix_for_unichain(P_pi)
    log['P_pi'] = P_pi.tolist(); print(f'P_pi: {log["P_pi"]}')
    log['r_pi'] = r_pi.tolist(); print(f'r_pi: {log["r_pi"]}')
    log['pstar_pi'] = Pstar_pi[0].tolist(); print(f'pstar_pi: {log["pstar_pi"]}')

    # Evaluate the specified policy based on the average reward criterion
    v_gain = agent.evaluate_a_policy_wrt_average_reward(P_pi, r_pi)
    log['v_gain'] = v_gain.tolist(); print(f'v_gain: {v_gain}')

    # Evaluate the specified policy based on the discounted reward criteria
    log['v_gamma'] = {}
    for gamma in [gamma for gamma in xcfg['discount_factors'] if gamma is not None]:
        v_gamma = agent.evaluate_a_policy_wrt_discounted_reward(P_pi, r_pi, gamma)
        log['v_gamma'][gamma] = v_gamma.tolist()
    print(f"v_gamma: {log['v_gamma']}")

    # Exhaustive search over all stationary deterministic (SD) policies
    log['optimal'] = {gamma: {} for gamma in xcfg['discount_factors']}
    for gamma in xcfg['discount_factors']:
        if gamma is None: value_fn = agent.evaluate_a_policy_wrt_average_reward
        else: value_fn = agent.evaluate_a_policy_wrt_discounted_reward
        optimal_policies, optimal_value = \
            agent.exhaustive_search_over_all_stationary_deterministic_policies(
                env.p0, env.p, env.get_reward_mat(), value_fn, gamma)
        log['optimal'][gamma]['optimal_value'] = optimal_value
        log['optimal'][gamma]['optimal_policies'] = [pi.tolist() for pi in optimal_policies]
        print(f'Optimal wrt gamma:{gamma} \n optimal_value: {optimal_value} \n {optimal_policies}')

    # Simulate the agent-environment interaction
    sasr_list = [] # a sequence of state, action, nextstate, nextreward
    s = env.get_initialstate(); tmix = None
    for t in range(xcfg['xprmt_episode_length']):
        a = agent.policy(s)
        snext, rnext = env.step(s, a)
        
        Pt_pi = np.linalg.matrix_power(P_pi, t)
        sasr_list.append((s, a, snext, rnext))
        print(f'{t}: {s}, {a}, {snext}, {rnext} \n {Pt_pi}')

        if tmix is None and np.allclose(Pt_pi, Pstar_pi):
            tmix = t
    log['tmix'] = tmix; print(f'tmix: {tmix}'); assert tmix is not None
    v_gain_empirical = sum([sasr[-1] for sasr in sasr_list])/len(sasr_list)
    log['v_gain_empirical_from_s0'] = v_gain_empirical
    print(f'v_gain_empirical_from_s0: {v_gain_empirical}')
    print(f'v_gain_theoretical_from_s0: {v_gain[0]}')

    # Write all the logs
    with open('xprmt_log.yml', 'w') as f:
        yaml.dump(log, f, width=float('inf'))

if __name__ == '__main__':
    main()
