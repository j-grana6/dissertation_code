import numpy as np
from pandas import rolling_sum
import pandas as pd
from collections import defaultdict
from scipy.optimize import fmin_slsqp
from scipy.stats import  norm, chi2, ncx2
import copy
import time
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm

class LevelKModel(object):
    """
    Provides the tools to solve the level-K attacker and defender model

    tau : int
        Time window

    sigmamat : 2-d array
        sigmamat[i,j] is the standard deviation of noise from node i
        to node j

    r : 2-d array
        Reward per unit of transfer to the attacker from i to j

    b1,b2,c1 : floats
        Parameters in defender's expected utility

    pi : float
        Exogenous probability of attacker

    c : 2-d matrix
         Cost to defender for unit of data transfer from i to j

    effort : per time period per edge limit on attacker

    
    """
    def __init__(self, tau, sigmamat, r,b1,b2,c1,pi,c, effort, **kwargs):
        self.tau = tau 
        self.sigmamat = sigmamat
        self.r = r 
        self.b1 = b1
        self.b2 = b2
        self.c1 = c1
        self.pi = pi
        self.positive_edges = [(np.nonzero(self.sigmamat)[0][i], np.nonzero(self.sigmamat)[1][i])
                               for i in range(np.sum(sigmamat>0))] 
                                        #List of edges with traffic
        self.edges_into_node = self.get_node_edge_into() 
        self.edges_outof_node = self.get_node_edge_out()
        self.sumlogsigma = np.sum([np.log(x) for x in np.nditer(sigmamat) if x >0])
        self.x0opt = np.random.random(self.tau * len(self.positive_edges))
        self.noattackerstrat = self.attacker_strat_from_res(np.zeros(self.tau*len(self.positive_edges)))
        self.c = c
        self.effort = effort
        
    def get_node_edge_out(self):
        outin = defaultdict(list)
        for e in self.positive_edges:
            outin[e[0]].append(e)
        return outin
    
    def get_node_edge_into(self):
        inout = defaultdict(list)
        for e in self.positive_edges:
            inout[e[1]].append(e)
        return inout
    
    def get_attacker_dict(self, attackerstrat):
        """
        Returns attackers strategy as a dict keyed by edges; it is
        easier to work with.
        """
        astratdict = {}
        for ix, elem in enumerate(self.positive_edges):
            astratdict[elem] = np.asarray(attackerstrat[ix*self.tau:self.tau*(ix+1)])
        return astratdict
    
    def general_constraint(self, x):
        astrat = self.get_attacker_dict(x)
        l1 = self.nonzero_constraint(x)
        l2 = self.effort_constraint(x)
        l3 = self.in_out_constraint(astrat)
        return list(l1) + list(l2)+ list(l3)
    
    def nonzero_constraint(self, x):
        return x
    
    
    def effort_constraint(self, x):
        """ 
        Total effort over the period
        """
        return self.effort - x
    

    def in_out_constraint(self, astrat):
        allbalances = []
        if len(self.edges_outof_node.keys()):
            return []
        for key, val in self.edges_outof_node.iteritems():
            if key != 0 : # No traversal for first node
                total_out_at_t = np.sum(np.asarray([astrat[e] for e in val]), axis=0)
                total_out_by_t = np.cumsum(total_out_at_t)
                total_in_at_t = np.sum(np.asarray(
                        [astrat[e] for e in self.edges_into_node[key]]), axis=0)
                total_in_by_t = np.cumsum(np.hstack((np.array([0]), total_in_at_t)))[:-1]
                allbalances.append(total_in_by_t - total_out_by_t)
        return list(np.hstack(tuple(allbalances)))
            


    def prob_no_alarm_simple(self, d, astrat=None):
        """
        Probability of no alarm under simple likelihood.
        
        Given by P(chi2_nk(mu) < -2(ln(d) + nk/2 ln(2pi) + k \sum_J ln(\sigma_j)))
        
        where mu is given by \sum_IJ (aij/sigma_j)^2
        """
        muhat = 0
        param_contribution = -2 * (d + self.tau * len(self.positive_edges)/2.*np.log(2*np.pi)
                                       + self.tau*self.sumlogsigma )
        for edge in self.positive_edges:
                muhat +=  np.sum(astrat[edge]**2) / float(self.sigmamat[edge])**2
        if muhat > 0:
            p_no_alarm = ncx2.cdf(param_contribution, self.tau * len(self.positive_edges), muhat)
        if muhat ==0:
            p_no_alarm = chi2.cdf(param_contribution, self.tau * len(self.positive_edges))
        return p_no_alarm
    
    def prob_no_alarm_lr(self, lrd, astrat=None, astratbelief=None):
        """
        Probability no alarm under likelihood ratio detector
        
        lrd : float
            Defender's threshold

        astrat : dict
            Attacker's activity along edges

        astratbelief : dict
            Belief defender uses to compute likelihood ratio
        """
        stdev = self.get_sd_from_belief(astratbelief)
        evalat = -2*lrd
        for key, val in astratbelief.iteritems():
            evalat += np.sum(astratbelief[key]**2 - 2*astratbelief[key]*astrat[key]) \
                / self.sigmamat[key]**2
        return  norm.cdf(evalat, loc=0, scale=stdev)
        
    def get_sd_from_belief(self, astratbelief):
        """
        Standard deviation used to comput p(alarm)

        astratbelief : dict
            Defender's belief of attacker's strategy.

        """
        var = 0
        for key, val in astratbelief.iteritems():
            var += np.sum(4*astratbelief[key]**2) / float(self.sigmamat[key]**2)
        return var**.5

    
    def attacker_expected_utility_simple(self, x, d=None):
        """
        Computes (negative) attacker's expected utility when defender uses
        threshold d and simple likelihood
        """
        astrat = self.get_attacker_dict(x)
        u = 0
        pnoalarm = self.prob_no_alarm_simple(d, astrat=astrat)
        reward = 0
        for edge in self.positive_edges:
            reward += self.r[edge] * np.sum(astrat[edge])
        return -pnoalarm*reward
        
    def attacker_expected_utility_lr(self, x, astratbelief=None,lrd=None):
        """
        Computes (negative) attacker's expected utility when defender uses
        threshold d and likelihood ratio.
        """
        astrat = self.get_attacker_dict(x)
        u = 0
        pnoalarm = self.prob_no_alarm_lr(lrd, astrat, astratbelief)
        reward = 0
        totaleffort = 0
        for edge in self.positive_edges:
            totaleffort += np.sum(astrat[edge])
            reward += self.r[edge] * np.sum(astrat[edge])
        return - pnoalarm*reward
                
            
    def defender_expected_utility_simple(self, d, astrat=None):
        """
        Computes defender's expected utility when he uses a simple
        likelihood
        """
        pnawa = self.prob_no_alarm_simple(d, astrat) 
        pnana = self.prob_no_alarm_simple(d, self.noattackerstrat)
        uncaught_cost = 0
        for edge in self.positive_edges:
            uncaught_cost += self.c[edge] * np.sum(astrat[edge])
        acost = self.pi * ( (1-pnawa)*self.b1 -  pnawa*uncaught_cost )
        noacost = (1-self.pi) * ( pnana*self.b2 - (1-pnana)*self.c1)
        eu = acost + noacost
        return - eu
    
    def defender_expected_utility_lr(self, lrd, astrat=None, astratbelief=None):
        """
        Computes defender's expected utility when he uses a likelihood
        ratio
        """
        pnawa = self.prob_no_alarm_lr(lrd, astrat, astratbelief) 
        pnana = self.prob_no_alarm_lr(lrd, self.noattackerstrat, astratbelief)
        uncaught_cost = 0
        for edge in self.positive_edges:
            uncaught_cost += self.c[edge] * np.sum(astrat[edge])
        acost = self.pi * ( (1-pnawa)*self.b1 - pnawa*uncaught_cost )
        noacost = (1-self.pi)* ( pnana*self.b2 - (1-pnana)*self.c1)
        eu = acost + noacost
        return - eu
        
    
    def attacker_strat_from_res(self, res):
        """
        Get's attacker dictionary from the result of an optimization.
        However, it rounds everything so that intensities close to 0
        become 0.
        """
        astar = {}
        for ix, edge in enumerate(self.positive_edges):
            astar[edge] = np.round(res[ix*self.tau: (ix + 1)*self.tau], decimals=3)
        return astar
    
    def array_from_astrat(self,astrat):
        """
        From an attacker's strategy in dictionary form, returns the
        attacker's strategy in array form
        """
        x=np.array([])
        for edge in self.positive_edges:
            x=np.hstack((x, astrat[edge]))
        return list(x)
    
    def solve_attacker_simple(self, d0):
        """
        Returns attacker's optimal strategy when the defender uses a
        simple likelihood
        """
        f = lambda x : self.attacker_expected_utility_simple(x, d=d0)
        res = fmin_slsqp(f, self.x0opt, f_ieqcons = self.general_constraint, full_output=True, disp=False)
        return res
    
    def solve_attacker_lr(self, lrd, abelief):
        """
        Returns attacker's optimal strategy when the defender uses a
        likelihood ratio.
        """
        for key, val in abelief.iteritems():
            abelief[key][abelief[key]<.0001] = 0.
        f = lambda x : self.attacker_expected_utility_lr(x, lrd=lrd, astratbelief=abelief)
        res = fmin_slsqp(f, self.x0opt, f_ieqcons = self.general_constraint, full_output=True, disp=False,
                        acc=10**-10, epsilon=10**-10)
        return res
    
    def solve_defender_lr(self, astratactual, astratbelief):
        """
        Returns defender's optimal strategy when he uses a likelihood
        ratio
        """
        for key, val in astratbelief.iteritems():
             astratbelief[key] = np.maximum(.02, val)
        f = lambda x : self.defender_expected_utility_lr(x, astratactual, astratbelief)
        res = fmin_slsqp(f, np.array([-5]), full_output=True, disp=False)
        return res
    
    def solve_defender_simple(self, astratactual):
        """ 
        Returns defender's optimal strategy when he uses a
        simple likelihood
        """
        f = lambda x : self.defender_expected_utility_simple(x, astrat=astratactual)
        res = fmin_slsqp(f, np.array([-250]), full_output=True, disp=False)
        return res
    
    def get_results_simple(self, d0list, maxlevel=3):
        """
        Returns a dataframe that gives the attacker and defender
        strategies for each level as well as expected utilities.  It
        also stores the input parameters.  

        d0list : list
            A list of level-0 defender thresholds

        maxlevel : int
            The highest level, k, to compute
        
        """
        frame = []
        flattenparams =  [self.tau]  \
                         + list(self.sigmamat.flatten())  \
                         + [self.b1, self.b2, self.c1]
        ix=0
        for d in d0list:
            ix +=1
            columns = [ "tau"]\
                  + ["sigma" +str(i) for i in range(self.sigmamat.shape[0]**2)] \
                  + ["b1", "b2", "c1", "d"]
            row = copy.copy(flattenparams)
            row.append(d)
            attackerres = self.solve_attacker_simple(d)
            row.append(-1*attackerres[1])
            columns = columns + ["L1AEU_V_DL0"] 
            row = row + list(attackerres[0])
            columns = columns +  ["L1A" +str(i) for i in range(self.tau*len(self.positive_edges))]
            attackerdict = self.attacker_strat_from_res(attackerres[0])
            defendereu = self.defender_expected_utility_simple(d, attackerdict)
            row.append(-1 * defendereu)
            columns = columns + ["L0DEU_V_AL1"]
            level = 2
            while level< maxlevel:
                defenderres = self.solve_defender_simple(attackerdict)
                row = row + list(-1 * defenderres[1])
                columns = columns + ["L" + str(level) + "DEU_V_AL" + str(level-1)]
                row = row + list(defenderres[0])
                columns = columns + ["L" +str(level) + "LRThresh"]
                ainput = self.array_from_astrat(attackerdict)
                attackereu = self.attacker_expected_utility_simple(ainput, 
                                                              defenderres[0])
                row.append(-1 * attackereu)
                columns = columns + ["L" +str(level-1) +"AEU_V_DL" + str(level)]
                attackerres = self.solve_attacker_simple(defenderres[0])
                attackerdict = self.attacker_strat_from_res(attackerres[0])
                row = row + list(-1 * attackerres[1]) 
                columns = columns + ["L"+str(level+1) + "AEU_V_DL" +str(level)]
                row = row + list(attackerres[0])
                columns = columns + ["L" + str(level +1) + "A" +str(i) for i in 
                            range(self.tau*len(self.positive_edges))]
                defendereu = self.defender_expected_utility_simple(defenderres[0], 
                                                            attackerdict)
                row = row + list(-1 * defendereu)
                columns = columns + ["L" + str(level) + "DEU_V_AL" + str(level +1)]
                level += 2
            frame.append(row)
        return pd.DataFrame(frame, columns=columns)
    
    def get_results_lr(self, d0list, a0, maxlevel=3, smooth=False, hetero=False):
        """
        Returns a dataframe with level-k results when the defender
        uses a likelihood ratio.  

        d0list : list
            Defender's level-0 threshold

        a0 : array
            Defender's level-0 belief of attacker's strategy in array
            form. 

        smooth : bool
            If true, the defender uses "smooth" beliefs over paths to
            exfiltration point
        
        hetero : bool
            If true, the defender uses "smooth" beliefs where the
            standard deviation along the right path is 1 and along the
            left path is 2.
        
        """
        a0 = self.attacker_strat_from_res(a0)
        frame = []
        flattenparams =  [self.tau]  \
                         + list(self.sigmamat.flatten())  \
                         + [self.b1, self.b2, self.c1]
        ix=0
        for d in d0list:
            ix +=1
            columns = [ "tau"]\
                  + ["sigma" +str(i) for i in range(self.sigmamat.shape[0]**2)] \
                  + ["b1", "b2", "c1", "d"]
            row = copy.copy(flattenparams)
            row.append(d)
            attackerres = self.solve_attacker_lr(d, a0)
            row.append(-1*attackerres[1])
            columns = columns + ["L1AEU_V_DL0"] 
            row = row + list(attackerres[0]) #strat
            columns = columns +  ["L1A" +str(i) for i in range(self.tau*len(self.positive_edges))]
            attackerdict = self.attacker_strat_from_res(attackerres[0])
            defendereu = self.defender_expected_utility_lr(d, attackerdict, a0)
            row.append(-1 * defendereu)
            columns = columns + ["L0DEU_V_AL1"]
            level = 2
            attackerdictactual = attackerdict
            if smooth:
                attackerdictbelief = self.get_smooth_belief(attackerdictactual)
            elif hetero:
                attackerdictbelief = self.get_smooth_belief_hetero(attackerdictactual)
            else:
                 attackerdictbelief = attackerdictactual
            while level< maxlevel:
                defenderres = self.solve_defender_lr(attackerdictactual, attackerdictbelief)
                row = row + list(-1 * defenderres[1])
                columns = columns + ["L" +str(level) +"DEU_V_AL" + str(level-1)]
                row = row + list(defenderres[0])
                columns = columns + ["L" +str(level) + "LRThresh"]
                ainput = self.array_from_astrat(attackerdictactual)
                attackereu = self.attacker_expected_utility_lr(ainput, attackerdictbelief, 
                                                              defenderres[0])
                row.append(-1 * attackereu)
                columns = columns + ["L" +str(level-1) +"AEU_V_DL" + str(level)]
                attackerres = self.solve_attacker_lr(defenderres[0], attackerdictbelief)
                attackerdictactual = self.attacker_strat_from_res(attackerres[0])
                row = row + list(-1 * attackerres[1])
                columns = columns + ["L"+str(level+1) + "AEU_V_DL" +str(level)]
                row = row + list(attackerres[0]) 
                columns = columns + ["L" + str(level +1) + "A" +str(i) for i in 
                            range(self.tau*len(self.positive_edges))]
                defendereu = self.defender_expected_utility_lr(defenderres[0], 
                                                               attackerdictactual, attackerdictbelief)
                row = row + list(-1 * defendereu)
                columns = columns + ["L" + str(level) + "DEU_V_AL" + str(level +1)]
                if smooth:
                    attackerdictbelief = self.get_smooth_belief(attackerdictactual)
                elif hetero:
                    attackerdictbelief = self.get_smooth_belief_hetero(attackerdictactual)
                else:
                    attackerdictbelief = attackerdictactual
                level += 2
            frame.append(row)
        return  pd.DataFrame(frame, columns=columns)


    
    def get_smooth_belief(self, astratdict):
        """
        This function smooths beliefs only in the 5 node network
        presented in the dissertation.  It is not general enough to
        handle arbitrary network topologies.  
        """
        total_exfiltrate = np.sum(astratdict[(3,4)])
        attackerdictbelief = copy.deepcopy(astratdict)
        attackerdictbelief[(3,4)] = np.zeros(15)
        attackerdictbelief[(3,4)][2:] = total_exfiltrate/13.
        attackerdictbelief[(0,1)] = np.ones(15) * total_exfiltrate/26.
        attackerdictbelief[(0,1)][13:] = 0
        attackerdictbelief[(0,2)] = np.ones(15) * total_exfiltrate/26.
        attackerdictbelief[(0,2)][13:] = 0
        attackerdictbelief[(1,3)] = np.ones(15) * total_exfiltrate/26.
        attackerdictbelief[(1,3)][0] = 0
        attackerdictbelief[(1,3)][-1] = 0
        attackerdictbelief[(2,3)] = np.ones(15) * total_exfiltrate/26.
        attackerdictbelief[(2,3)][0] = 0
        attackerdictbelief[(2,3)][-1] = 0
        return attackerdictbelief
    
    def get_smooth_belief_hetero(self, astratdict):
        """
        This function smooths beleifs when the standard deviation
        along the right path is 1 and along the left path is 2.  This
        is not general.
        """
        total_exfiltrate = np.sum(astratdict[(3,4)])
        attackerdictbelief = copy.deepcopy(astratdict)
        attackerdictbelief[(3,4)] = np.zeros(15)
        attackerdictbelief[(3,4)][2:] = total_exfiltrate/13.
        attackerdictbelief[(0,1)] = np.ones(15) * total_exfiltrate/(5.*13)
        attackerdictbelief[(0,1)][13:] = 0
        attackerdictbelief[(0,2)] = np.ones(15) * 4*total_exfiltrate/(5*13.)
        attackerdictbelief[(0,2)][13:] = 0
        attackerdictbelief[(1,3)] = np.ones(15) * total_exfiltrate/(5*13.)
        attackerdictbelief[(1,3)][0] = 0
        attackerdictbelief[(1,3)][-1] = 0
        attackerdictbelief[(2,3)] = np.ones(15) * 4*total_exfiltrate/(13*5.)
        attackerdictbelief[(2,3)][0] = 0
        attackerdictbelief[(2,3)][-1] = 0
        return attackerdictbelief

class LevelKPlotting(object):
    """
    Plotting results for level-k game

    lkmodel : LevelKModel
        Instance of level-k game

    lkresults : dataframe
        Output of either get_results_lr or get_results_simple
    """
    
    def __init__(self, lkmodel, lkresults):
        self.lkmodel = lkmodel
        self.lkresults = lkresults

    def get_attacker_strat_from_res(self, level, d0, against="d"):
        cs =  ["L" + str(level) + "A" +str(i) for i in range(self.lkmodel.tau*len(self.lkmodel.positive_edges))]
        sonly = self.lkresults[[against] + cs]
        strat = sonly[abs(sonly[against]-d0)<.01][cs].values[0]
        astratdict = self.lkmodel.attacker_strat_from_res(strat)
        return astratdict

    
    def plot_attacker_strategy(self, d0, edges, level, linewidth=1, 
                            against="d"):
        res = self.lkresults
        fig, axes = plt.subplots(5, figsize=(8,8))
        estr = ["Server", "Host 1", "Host 2", "Host 3", "Internet"]
        ix = 0
        cs = cm.inferno
        for edge in edges:
            ax = axes[ix]
            cix = 0
            activity = self.get_attacker_strat_from_res(level, d0)[edge]
            ax.plot(np.arange(1, len(activity)+1), activity, color="black")
            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel("Intensity", fontsize=12)
            ax.set_title(estr[edge[0]]+ " to " + estr[edge[1]]  , fontsize=12)
            ax.grid(True)
            ax.set_ylim(0, max(activity) + .25)
            ix +=1
        # for ix, ax in enumerate(axes):
        #     pos1 = ax.get_position() # get the original position 
        #     pos2 = [pos1.x0, pos1.y0 + .15-ix/15.,  pos1.width, pos1.height] 
        #     ax.set_position(pos2)
        fig.tight_layout()
        return fig, ax

    def deu_over_levels(self, maxlevel, d0, dumb=True):
        res = self.lkresults
        cols = []
        for i in np.arange(0,maxlevel,2):
            if i==0:
                cols.append("L"+str(i)+"DEU_V_AL"+str(i+1))
            else:
                if dumb:
                    cols.append("L"+str(i)+"DEU_V_AL"+str(i+1))
                else:
                    cols.append("L"+str(i)+"DEU_V_AL"+str(i-1))
        deu = res[cols][abs(res["d"]-d0)<.01]
        return deu

    def plot_deu_by_level(self, maxlevel, d0list, dumb=True):
        """
        Plots the defender's expected utility.
        """
        res = self.lkresults
        fig, ax = plt.subplots()
        ix = 0
        for d0 in d0list:
            if len(d0list) ==1:
                b = 1
            else:
                b = (255 - 255*ix/float(len(d0list)))/float(255)
            toplot = self.deu_over_levels(maxlevel, d0, dumb).values.T
            ax.plot(np.arange(0, maxlevel,2), toplot, color = cm.inferno(int(255*(1-b))), linewidth=1)
            ix +=1
        if len(d0list) > 1:
            ax2 = fig.add_axes([0.95, 0.12, 0.05, 0.78])
            cb = mpl.colorbar.ColorbarBase(ax2, cmap=plt.cm.inferno, 
                                   boundaries=d0list, 
                                   label="Initial Threshold")
        fig.suptitle("Defender of level-$k$ expected utility against an attacker of level $k+1$.", fontsize=12)
        ax.set_xlabel("$k$")
        ax.set_ylabel("Defender's Expected Utility")
        return fig, ax

    def compare_astrat(self, d, edges, levels=[11,13], title_in=None, linewidth=1, 
                            against="d"):
        res = self.lkresults
        fig, axes = plt.subplots(5, figsize=(8,8))
        estr = ["Server", "Host 1", "Host 2", "Host 3", "Internet"]
        ix = 0
        cs = cm.inferno
        for edge in edges:
            ax = axes[ix]
            cix = 0
            for l in levels:
                activity = self.get_attacker_strat_from_res(l, d)[edge]
                ax.plot(np.arange(1, len(activity)+1), activity, 
                                linewidth=linewidth, label="Level " + str(l) +" Attacker Strategy")
                ax.set_xlabel("Time Step", fontsize=12)
                ax.set_ylabel("Intensity", fontsize=12)
                ax.set_title(estr[edge[0]]+ " to " + estr[edge[1]]  , fontsize=12)
                ax.grid(True)
                cix += 1
            ix +=1
        fig.tight_layout()
        return fig, ax

    def aeu_over_levels(self, maxlevel, d0, dumb=True):
        res = self.lkresults
        cols = []
        for i in np.arange(1,maxlevel,2):
            if dumb:
                cols.append("L"+str(i)+"AEU_V_DL"+str(i+1))
            else:
                cols.append("L"+str(i)+"AEU_V_DL"+str(i-1))
        aeu = res[cols][abs(res["d"]-d0)<.01]
        return aeu

    def plot_aeu_by_level(self,  maxlevel, d, dumb=False):
        fig, ax = plt.subplots()
        toplot = self.aeu_over_levels(maxlevel, d, dumb=dumb).values.T
        ax.plot(np.arange(1, maxlevel,2), toplot, color = "black", linewidth=1)
        fig.suptitle("Attacker of level $k$ expected utility against a defender of level $k-1$.", fontsize=12)
        ax.set_xlabel("$k$")
        ax.set_ylabel("Attacker's Expected Utility")
        return fig, ax
