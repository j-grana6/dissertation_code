import numpy as np
import copy
from tools import prob_model_given_data_times, gen_logn_fact, prob_model_no_attacker
from itertools import chain



def Direct_Sample(CyberNet, data, num_samples, T, s0):
    """
    Returns P(data|attacker) by Monte Carlo Sampling

    CyberNet : CyberNet Instance

    data : list
        Output of gen_data

    num_samples : int
        How many Monte Carlo samples

    T : int
        Time window

    s0 : dict
        Initial states of nodes
    """
    net = copy.deepcopy(CyberNet)
    # The states of nodes will be changing so we want to make sure
    # we do not change the input  network
    logn_fact = gen_logn_fact(data)
    #Precompute log(n!) for various n.  
    n = 1
    nodes_to_change = [nd for nd in net.node_names if s0[nd] == 'normal' ]
    nodes_no_change = [nd for nd in net.node_names if s0[nd] == 'infected']
    prob_no_attacker = prob_model_no_attacker(net, data, T, logn_fact)
    numattackers = len(nodes_no_change)
    prob_mod = lambda x : prob_model_given_data_times(net, data, x, T,
                                                logn_fact, s0)
    # Only input is not the infection times
    probs = []
    while n < num_samples:
        t = 0
        for nd in net.node_names:
            net.node_dict[nd].state = s0[nd]
        times = {nd: 0 for nd in nodes_no_change}
        # Corresponds to correct order
        while t<T :
            infected = [nd.name for nd in net.nodes if nd.state =='infected']
            at_risk = set(chain(*[net.node_dict[nd].sends_to for nd in infected])) - set(infected)
            if len(at_risk) == 0:
                break
            at_risk_ix = [net.node_names.index(nd) for nd in at_risk]
            mt_rates = np.sum(net.get_mal_trans()[:, at_risk_ix], axis=0)
            #print at_risk, mt_rates, infected, n
            r_rate = np.sum(mt_rates)
            t += np.random.exponential(scale=1./r_rate)
            # Sample time of next infection
            if t<T:
                next_infected = np.random.choice(list(at_risk), p = mt_rates/float(sum(mt_rates)))
                # Sample node to be infected
                times[next_infected] = t
                net.node_dict[next_infected].state = 'infected'
        #print times, n
        probs.append(prob_mod(times))
        n+=1
    # prob_mod returns log prob so we need to exponentiate to get the mean    
    e_probs = np.exp(probs)
    return np.log(np.mean(e_probs))
