"""
This will attempt to provide test coverage for the code.  It will
use a small CyberNet and test very simple cases where known solutions
exist.  
"""
from node import Node
from cybernet import CyberNet
from tools import *
import numpy as np
from scipy.stats import poisson
from scipy import integrate
from scipy.misc import factorial
from direct_sample import Direct_Sample


def test_w_and_without():
    """
    If we specify that no note is initially infected at t=0, then
    there is no attacker present and therefore the function that
    computes P(data|no attacker) should return the same value as the
    function P(data|attacker, no node initially infected).
    
    Also, the prob_data_no_a should just be the log-value
    of a Poisson distribution
    """
    A = Node("A", ["B"], {"B": np.array([[1,0],[1,.1]])})
    B = Node("B", [], {})
    net = CyberNet([A,B])
    T=10
    data = gen_data(T, net, {"A": "normal", "B":"normal"})
    logn_fact = gen_logn_fact(data)
    pdata_no_a = prob_model_no_attacker(net, data, T, logn_fact)
    pdata_a = prob_model_given_data_times(net, data, {}, T, logn_fact,
                                          {"A": "normal",
                                           "B":"normal"})
    np.testing.assert_almost_equal(pdata_no_a, pdata_a)

    np.testing.assert_almost_equal(np.log(poisson.pmf(len(data[0]), 10)), pdata_a)

def test_montecarlo():
    """
    With a three node net, we can test Direct_Sample against numerical
    integration. 
    """
    A = Node("A", ["B"], {"B": np.array([[1,0],[1,.2]])})
    B = Node("B", ["C"], {"C": np.array([[1,0],[1,.4]])})
    C = Node("C", [], {})
    net = CyberNet([A,B,C])
    T=10
    data = gen_data(T, net, {"A": "infected", "B":"normal", "C": "normal"})
    dsres = Direct_Sample(net, data, 10000, 10, {"A": "infected",
        "B":"normal", "C":"Normal"})
    probfroma = np.log(poisson.pmf(np.sum(data[2]=="A"), 12))
    def integrand(zbar, T=T, data=data):
        fromb_times = data[1][data[2]=="B"]
        #total = len(fromb_times)
        numbefore = np.sum(fromb_times<=zbar)
        numafter = np.sum(fromb_times>zbar)
        pbefore = zbar**numbefore*np.exp(-zbar)/float(factorial([numbefore])[0])
        pafter = (1.4*(T-zbar))**numafter*np.exp(-1.4*(T-zbar))/float(factorial([numafter])[0])
        return pbefore*pafter*.2*np.exp(-.2*zbar)
    total = len(data[1][data[2]=="B"])
    num_integral = integrate.quad(integrand, 0,10, epsabs=.01) + \
               np.exp(-2)*10**total*np.exp(-10)/float(factorial([total])[0])
    np.testing.assert_allclose(np.log(num_integral[0]) + probfroma, dsres, atol=0, rtol=.01) #relative test

if __name__ == "__main__":
    test_w_and_without()
    test_montecarlo()
