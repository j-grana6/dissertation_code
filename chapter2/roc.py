from tools import gen_data, prob_model_no_attacker, gen_logn_fact
import numpy as np
from direct_sample import Direct_Sample
import random
from matplotlib import pyplot as plt
from multiprocessing import Pool
from functools import partial


def get_likelihoods(seed, num_pos, num_neg, CyberNet, s0, T, truenet=None,  directsamps=1000):
    """
    seed : int
        Random seed for Monte Carlo and data generation

    See plot_roc_parallel for more info

    """
    if truenet == None:
        truenet = CyberNet
    no_a_s0 = dict(zip(CyberNet.node_names, ['normal'] * len(CyberNet.nodes)))
    np.random.seed(seed)
    random.seed(seed)
    infected_lhoods = []
    # Will store the lhood w attacker and lhood difference
    clean_lhoods = []
    datalen = 0
    for i in range(num_pos):
        seed +=1
        np.random.seed(seed)
        random.seed(seed)
        # Augment the seed by 1 to generate new sample
        data = gen_data(T, truenet, s0)
        if len(data[0]) > datalen:
            datalen = len(data[0])
            logn_fact = gen_logn_fact(data)
            # Only generate these as needed
        p_data_attacker = Direct_Sample(CyberNet, data, directsamps, T, s0)
        p_no_attacker = prob_model_no_attacker(CyberNet, data, T, logn_fact)
        infected_lhoods.append((p_data_attacker, p_no_attacker))
    for j in range(num_neg):
        seed += 1
        np.random.seed(seed)
        random.seed(seed)
        # Code uses both
        data = gen_data(T, truenet, no_a_s0)
        if len(data[0]) > datalen:
            datalen = len(data[0])
            logn_fact = gen_logn_fact(data)
            # Only generate these as needed
        p_data_attacker = Direct_Sample(CyberNet, data, directsamps, T, s0)
        p_no_attacker = prob_model_no_attacker(CyberNet, data, T, logn_fact)
        clean_lhoods.append((p_data_attacker, p_no_attacker))
    return np.asarray(infected_lhoods), np.asarray(clean_lhoods)
    

def get_lr_roc_pts(infect_res, clean_res, lhood_ratio_step):
    """
    infect_res : array
        array with columns P(data | attacker) and P(data | no attacker) when there
        is an attacker

    clean res : array
        array with columns P(data | attacker) and P(data | no attacker) when there
        is not an attacker

    step : float
        Threshhold step
    """
    roc_pts = []
    infect_lhood_diff  = infect_res[:, 0] - infect_res[:, 1]
    clean_lhood_diff = clean_res[:, 0] - clean_res[:, 1]
    threshmin = min( min(clean_lhood_diff), min(infect_lhood_diff)) +.05
    threshmax = max( max(clean_lhood_diff), max(infect_lhood_diff)) +.05
    for step in np.arange(threshmin, threshmax, lhood_ratio_step):
        tps = np.sum(infect_lhood_diff > step)
        tps_rate = float(tps) / float( len(infect_lhood_diff) )
        fps = np.sum(clean_lhood_diff > step)
        fps_rate = float(fps) / float( len ( clean_lhood_diff))
        roc_pts.append((fps_rate, tps_rate))
    return roc_pts


def get_anomaly_roc_pts(infect_res, clean_res, lhood_step):
    """
    See above
    """
    lhood_infect = np.asarray(infect_res)[:,1]
    lhood_clean = np.asarray(clean_res)[:, 1]
    roc_pts = []
    threshmin = min( min(lhood_infect), min(lhood_clean)) +.05
    threshmax = max( max(lhood_infect), max(lhood_clean)) +.05
    for step in np.arange(threshmin, threshmax, lhood_step):
        tps = np.sum(lhood_infect < step)
        tps_rate = float(tps) / float( len(lhood_infect) )
        fps = np.sum(lhood_clean < step)
        fps_rate = float(fps) / float( len(lhood_clean))
        roc_pts.append((fps_rate, tps_rate))
    return roc_pts



def plot_roc_parallel(num_pos, num_neg, CyberNet, s0, T,
                      truenet=None,  directsamps=1000, seeds=None, numcores=4):
    """
    Plots the ROC curve
    
    num_pos : int
        Number of true positive realizations of network activity per core

    num_neg :
        Number of true negative realizations of network activity per core

    CyberNet : CyberNet
        The CyberNet the defender uses to model the network


    s0 : dict
       Initial state of the net when there is an attacker

    T : int
        Observation Window

    truenet : CyberNet instance
        A net used to generate  data.  If model is misspecified,
        truenet is different from CyberNet
    

    direct_samps : int
        Number of samples to compute P(data|attacker)

    seeds : lsit
        List of seeds to pass to each process.  Difference between any
        two seeds must be greater than num_pos + num_neg
    
    numcores: int
        Number of cores to use
 
    """
    if truenet is None:
        truenet=CyberNet
    if seeds is None:
        firstseed = np.random.randint(10000)
        seeds = list(np.arange(firstseed, firstseed+numcores*(num_pos+num_neg), num_pos+num_neg))
    g = _likelihood_partial(num_pos=num_pos, num_neg = num_neg, CyberNet=CyberNet, s0=s0,
                   T=T, truenet=truenet, directsamps=directsamps)
    p=Pool(numcores)
    res = p.map(g, seeds)
    pos, neg = handle_parallel_res(res)
    our_roc_pts = np.asarray(get_lr_roc_pts(pos, neg, .02))
    anom_roc_pts = np.asarray(get_anomaly_roc_pts(pos, neg, .02))
    fig, ax = plt.subplots()
    ax.plot(our_roc_pts[:,0], our_roc_pts[:,1], label =' Likelihood Ratio Detector')
    ax.plot(anom_roc_pts[:,0], anom_roc_pts[:,1], label = 'Simple Anomaly Detector')
    plt.legend(loc=4)
    return fig, ax

def handle_parallel_res(res):
    pos = []
    neg = []
    numcores = len(res)
    for i in range(numcores):
        pos.append(res[i][0])
        neg.append(res[i][1])
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    pos = pos.reshape(-1, 2)
    neg = neg.reshape(-1, 2)
    return pos, neg

def _likelihood_partial(num_pos, num_neg, CyberNet, s0, T, truenet=None,  directsamps=1000):
    return partial(get_likelihoods, num_pos=num_pos, num_neg = num_neg, CyberNet=CyberNet, s0=s0,
                   T=T, truenet=truenet, directsamps=directsamps)
