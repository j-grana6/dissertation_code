"""
This script will test the level-k model.  The main function that needs  
testing is the P(no alarm) functions.  The rest of the functions do 
not "crunch numbers" but simply provide a convenient way to handle the   
level-k optimizations. This will be tested using monte carlo.

To test, we will create a two node network with one directed edge and
1 time periods. From such a simple model, we can then test that the
optimization is working via the results of chapter 3.  
"""


from exfiltration import *
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from scipy.stats import norm

parameters = { 'tau': 2,
                'sigmamat' : np.array([[0., 1.], 
                                       [0., 0.]]),
              'r' : np.array(         [[0., 1.], 
                                       [0., 0.]]),

                'c' :  np.array(      [[0., 1.],
                                       [0., 0.]]),

                'b1': 1, # Catching Attacker
                'b2': 1, # Let network function
                'c1': 1, # False alarm
                'pi' : .1,
                'effort': 5,
             }

def test_p_no_alarm():
    model = LevelKModel(**parameters)
    threshold = 0 # this is log threshold
    astrat = {(0,1): np.array([.2,.2])}
    astratbelief = {(0,1): np.array([.2,.2])}
    pnoalarm = model.prob_no_alarm_lr(threshold, astrat, astratbelief) 
    normalactivity = np.random.normal(size=(2,1000000))
    totalactivity = normalactivity +.2
    lr = np.prod(np.exp(totalactivity**2/-2.) ,axis=0) / \
            np.prod(np.exp(normalactivity**2/-2.), axis=0)
    loglr = np.log(lr)
    assert_allclose(pnoalarm, 1 - np.sum(loglr<threshold)/1000000.,
             atol=0, rtol=.01)

def test_FOC():
    """
    This function will test that the attacker's strategy in two
    periods satisfies to FOC given in chapter 3.  
    """
    parameters = { 'tau': 2,
                'sigmamat' : np.array([[0., 1.], 
                                       [0., 0.]]),
              'r' : np.array(         [[0., 1.], 
                                       [0., 0.]]),

                'c' :  np.array(      [[0., 1.],
                                       [0., 0.]]),

                'b1': 1, # Catching Attacker
                'b2': 1, # Let network function
                'c1': 1, # False alarm
                'pi' : .1,
                'effort': 5,
             }
    model = LevelKModel(**parameters)
    alpha = (1-model.pi)*2
    beta = lambda strat: model.pi*(1+np.sum(strat))
    astratbelief = {(0,1): np.array([2,2])}
    p1b = astratbelief[(0,1)][0]
    p2b = astratbelief[(0,1)][1]
    W = lambda strat : -2*log(beta(strat)/alpha) - \
                p1b**2 -p2b**2 -2*strat[0]*p1b - 2*strat[1]*p2b
    foc1 = lambda strat : norm.cdf(W(strat)) - p1b*norm.pdf(W(strat))* \
                np.sum(strat)
    foc2 = lambda strat : norm.cdf(W(strat)) - p2b*norm.pdf(W(strat))* \
                np.sum(strat) 
    lrd = np.log(beta([2,2])/alpha)
    l1strat = model.solve_attacker_lr(lrd, astratbelief)
    assert_almost_equal(np.array([foc1(l1strat[0]),
                foc2(l1strat[0])]), np.zeros(2))
