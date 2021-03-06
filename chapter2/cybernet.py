"""
Constructs an SFT net from SFT's
"""
import itertools
import numpy as np


class CyberNet(object):
    """
    Creates an cyber-network from Node objects.

    The basic functionality of the CyberNet is to aggregate the information 
    in each Node instance and provide basic functions that facilitate
    simulation.  

    Parameters
    ----------

    nodes : list
         A list containing node instances to be included in the CyberNet.


    """

    def __init__(self, nodes):
        self.nodes = nodes
        self.node_names = [x.name for x in nodes]
        self.node_dict = dict([(node.name, node) for node in self.nodes])
        self.clean_trans, self.mal_trans, self.inf_trans = self._gen_trans_vect() 

    def get_state(self):
        """
        Returns the state of all nodes in the network
        """

        return [x.state for x in self.nodes]

    def _gen_trans_vect(self):
        """
        Returns 3 dictonaries, all keyed by the node name.

        The values of the dictionary with key "node" is a list, L,
        such that L[i] is the "clean", "malicious" or total rate of
        message transmission from "node" to node i, where i is given
        by the index of node_names.  
        """
        clean_rates = {}
        mal_rates = {}
        inf_rates = {}
        for nd in self.nodes:
            clean_r = []
            mal_r = []
            inf_r = []
            for nd2 in self.node_names:
                if nd2 in nd.sends_to:
                    clean_r.append(nd.rates[nd2][0][0])
                    mal_r.append(nd.rates[nd2][1][1])
                    inf_r.append(np.sum(nd.rates[nd2][1,:]))
                else:
                    clean_r.append(0)
                    mal_r.append(0)
                    inf_r.append(0)
            clean_rates[nd.name] = clean_r
            mal_rates[nd.name] =  mal_r
            inf_rates[nd.name] = inf_r
        return clean_rates, mal_rates, inf_rates
                    
    def get_mal_trans(self):
        """
        Returns np.array.

        Given the state of the nodes in the network it returns a
        matrix where the i'th row of the matrix is the rate at which
        node i sends malicious messageas to all other nodes.  The
        index is given in self.nodes (which is the same as
        self.node_names). 
        """
        mal_trans = []
        non_infect = list(np.zeros(len(self.nodes)))
        for nd in self.nodes:
            if nd.state =='normal':
                mal_trans.append(non_infect)
            else:
                mal_trans.append(self.mal_trans[nd.name])
        return np.asarray(mal_trans)

    def get_all_trans(self):
        """
        Returns np.array.

        Given the state of the nodes in the network it returns a
        matrix where the i'th row of the matrix is the rate at which
        node i sends messages to all other nodes.  The
        index is given in self.nodes (which is the same as
        self.node_names). 
        """
        all_trans = []
        for nd in self.nodes:
            if nd.state =='normal':
                all_trans.append(self.clean_trans[nd.name])
            else:
                all_trans.append(self.inf_trans[nd.name])
        return np.asarray(all_trans)
