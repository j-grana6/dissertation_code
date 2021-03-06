import numpy as np
import operator

def gen_data(T, CyberNet, t0):
    """

    Parameters
    ----------

    T : int
        Length of time

    CyberNet : CyberNet instance:
        A CyberNet

    t0 : dict
        A dictionary keyed by node names and values initial state

    """
    ##Make initial state list
    s0 = []
    for nd in CyberNet.node_names:
        s0.append(t0[nd])
    n_nodes = len(CyberNet.nodes)
    # Number of nodes
    reaction_times = []
    # empty list to collect reaction times
    reaction_sender = []
    # empty list to collect reaction sender
    reaction_receiver = []
    # collect the receiving nodes
    msgs_sent = []
    # Record the type of messages sent
    n_reactions = 0
    # initialize number of reactions
    t = 0
    # initialize infection time to T
    infect_times = dict(zip([x.name for x in CyberNet.nodes], [T]*n_nodes))
    for key, val in t0.iteritems():
        if val == 'infected':
            infect_times[key] = 0
    state = s0
    # Set states
    for s in range(len(state)):
        CyberNet.nodes[s].state = state[s]
    # Indicates wheter in the last time step, a node changed its
    # state. 
    state_change = 0
    t_rates = CyberNet.get_all_trans()
    while t < T:
        if t == 0 or state_change == 1:
            # If we are just starting, get the correct
            # transmission rates.  If there is a state
            # change, get the new transmission rates.
            # If the state of the system didn't change.
            # use the same transmission rate and skip
            # this block.

            # The transmission matrix that corresponds
            # to that state.
            t_rates = CyberNet.get_all_trans()   
            r_rate = np.sum(t_rates)
            # Reaction rate
        t += np.random.exponential(scale=1 / r_rate)
        if t > T:
            break
        reaction_times.append(t)
        # Draw the next time and append.
        # Marginally faster than t - log(random.random())/r_rate
        draw = r_rate*np.random.random()
        # Random number to determine sender and receiver
        reaction_ix = np.argmin(np.cumsum(t_rates) < draw)
        # argmin returns the index of the **first** element the
        # cum sum that is less than draw.  Therefore, the random
        # number was between ix-1 and ix of the cumsum, which is the
        # area of the distribution associated with a draw of reaction
        # given by reaction_ix
        sender_ix, receiver_ix = int(reaction_ix)/int(n_nodes),\
            reaction_ix % n_nodes
        # This is the location in the matrix of the reaction
        # The first term is the sending node index, the second
        # term is the receiving node index.
        sender, receiver = CyberNet.nodes[sender_ix], CyberNet.nodes[receiver_ix]
        # Get the actual sender and receiver nodes
        sndr_state_ix = sender.states.index(state[sender_ix])
        # What state is the sender in (from which sending distribution
        # do we draw?)
        msg_distribution = np.cumsum(sender.rates[receiver.name][sndr_state_ix])
        msg_ix = np.argmin(msg_distribution <
                            np.random.random() * msg_distribution[-1])
        # Determine the index of the message to send
        # Note that this data generating algorithm calls 2 random numbers.
        # One to determine the sender-receiver and the other to determine
        # the message to send.  Theoretically, these steps can be combined
        # and we can use only 1 random number but with more boilerplate
        msg = sender.messages[msg_ix]
        # The message string
        reaction_sender.append(sender.name)
        reaction_receiver.append(receiver.name)
        msgs_sent.append(msg)
        # Record the transmission
        receiver.react(msg)
        if state == CyberNet.get_state():
            # If the message is not malicious or the node was already
            # infected, this will hold.
            state_change = 0
        else:
            # The only time this happens is if a node gets infected
            state_change = 1
            infect_times[receiver.name] = t
            state = CyberNet.get_state()
            # Update state
        n_reactions += 1
    transmissions = {}
    for node in CyberNet.nodes:
        for o_node in node.sends_to:
            key = node.name+'-'+str(o_node)
            from_node = np.asarray(reaction_sender) == node.name
            to_o_node = np.asarray(reaction_receiver) == o_node
            times = from_node * to_o_node
            transmissions[key] = np.asarray(reaction_times)[times]

    return (np.asarray(msgs_sent), np.asarray(reaction_times),
            np.asarray(reaction_sender),
            np.asarray(reaction_receiver), n_reactions, transmissions,
            infect_times)



def prob_model_given_data_times(CyberNet, data, infect_times, T, logn_fact, s0):  
    """
    Returns  P(data | infect times)

    Parameters
    ----------

    CyberNet : CyberNet instance

    data : list
        Output of gendata

    infect_times: dict
    
    T : float
        Total running time

    logn_fact : list

    s0: dict

    """
    eps = 0
    transmissions = data[-2]
    # First order the infections
    attackers = [nd for nd in s0.keys() if s0[nd] =='infected']
    sorted_infect = sorted(infect_times.iteritems(),
                           key=operator.itemgetter(1))
    not_infected = [nd for nd in CyberNet.node_names \
                    if nd not in infect_times.keys()]

    # Reset the state of the nodes
    for nd in CyberNet.nodes:
        if nd.name in attackers:
            nd.state='infected'
        else:
            nd.state='normal'

    time_minus_1 = 0
    prob_no_infect_data = 0
    # Contribution to likelihood of nodes that do not get infected
    for node in not_infected:
        _node_inst = CyberNet.node_dict[node]
        # We need the node instance here.
        norm_ix = _node_inst.states.index('normal')
        # Loop through nodes to get (Log) probability of all messages
        # emitted 
        for o_node in _node_inst.sends_to:
            rate =  np.sum(_node_inst.rates[o_node][norm_ix, :])
            num_msgs = len(transmissions[node+'-'+o_node])
            prob_msgs = (num_msgs *
                    np.log(rate * T) -
                    logn_fact[num_msgs] -
                    rate * T)
            prob_no_infect_data += prob_msgs
            
    prob_data = prob_no_infect_data
    for node, time in sorted_infect:
        _node_inst = CyberNet.node_dict[node]
        norm_ix = _node_inst.states.index('normal')
        infect_ix = _node_inst.states.index('infected')
        for o_node in _node_inst.sends_to:
            if time == 0:
                # Because of problems with 0 in logs
                # we use this for nodes that are initially
                # infected
                num_msgs = len(transmissions[node+'-'+o_node])
                prob_msgs = (num_msgs *
                    np.log(
                    np.sum(_node_inst.rates[o_node][infect_ix, :]) * T) -
                    logn_fact[num_msgs] -
                    np.sum(_node_inst.rates[o_node][infect_ix, :]) * T)
                prob_data += prob_msgs
            else:
                num_before =  np.searchsorted(
                    transmissions[node+'-'+o_node],time)
                # Number of reactions before
                num_after = len(transmissions[node+'-'+o_node]) - num_before
                # Number of reactions after infection
                if num_before == 0 :
                    prob_before = - np.sum(_node_inst.rates[o_node][norm_ix, :]) * \
                                    min(T, time)
                else:
                    prob_before = (num_before *
                            np.log(eps +
                            np.sum(_node_inst.rates[o_node][norm_ix, :]) *
                            min(T, time)) -
                            logn_fact[num_before] -
                            np.sum(_node_inst.rates[o_node][norm_ix, :]) *
                            min(T, time))
                # prob before is the probability of node sending num_before
                # messages to o_node before it gets infected. 
                if num_after == 0:
                    prob_after = - np.sum(_node_inst.rates[o_node][infect_ix, :]) * \
                            (T-time)
                else:
                    prob_after = ( num_after *
                            np.log(eps +
                            np.sum(_node_inst.rates[o_node][infect_ix, :]) *
                            (T- time)) -
                            logn_fact[num_after] -
                            np.sum(_node_inst.rates[o_node][infect_ix, :]) *
                            (T-time))
                prob_data += prob_before + prob_after


    return prob_data


def prob_model_no_attacker(CyberNet, data, T, logn_fact):
    """
    Calculates the probability of the data when no node is
    initially infected.  It is just the probability of each
    sequence of observations.
    """
    total_prob = 0
    for node in CyberNet.nodes:
        # For each node
        for rec in node.sends_to:
        # For each possible receiver
            normal_ix = node.states.index('normal')
            clean_ix = node.messages.index('clean')
            rate = node.rates[rec][normal_ix, clean_ix]
            num_sent = np.sum((data[2] == node.name) * (data[3] == rec))
            logprob = -rate * T + num_sent * (np.log(rate * T))  \
                - logn_fact[num_sent]
            total_prob += logprob
    return total_prob

def gen_logn_fact(data):
    return np.hstack((np.zeros(1), np.cumsum(np.log(np.arange(1, len(data[0])+2,1)))))

