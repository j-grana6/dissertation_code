"""
An OO approach to SFT's
"""
import numpy as np

class Node(object):
    """
    An SFT node.

    Parameters
    ----------

    name : str
        The name of the node.

    sends_to  : list of str
        A list whose elements are names of nodes that receive messages
        from the SFT instance.


    rates : dict
        A dictionary with keys being the elements of sends_to.  The
        entry is a pxq array where p = len(states) and q =
        len(messages).  The order is determined by the order of
        sends_to and messages. e.x. for some node, if sends_to is
        ['A', 'B'], states is ['normal', 'infected'] and messages are
        ['clean', 'malicious'] then rates can be:
        >>> {'A': [[1, 0],[1, .00001]] , 'B': [[1,0], [1, .1]]}
        which means that the SFT sends clean messages to 'A' at a rate
        of 1 when it is in the normal state and sends message no
        malicious messages.  When the SFT is in an infected state, it
        sends clean messages to 'A' at a rate of 1 and malicious
        messages at a rate of .00001. 
    """

    def __init__(self, name,  sends_to, rates):
        self.name = name
        self.states = ["normal", "infected"]
        self.sends_to = sends_to
        self.rates = rates
        self.messages = ["clean", "malicious"]
        self.state = None


    def react(self, message):
        """
        A function that changes the node's state upon receiving a
        message 
        
        Parameters
        ----------

        message : str
            Content of the message
        """
        if message.lower() == 'malicious':
            self.state = 'infected'
        elif message.lower() == 'clean':
            pass 
        else:
            raise ValueError('Unknown message type: {}!').format(message)
