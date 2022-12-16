'''

Define a Gibbs sampler to approximate the posterior distribution of U.
@author: Benjamin Sischka

'''
import numpy as np
import math
import scipy.stats as stats
import scipy.ndimage.filters as flt
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import warnings
from copy import copy
from SBSGMpy.graphon import fctToMat, matToFct, colorBar

# Define a Sample Class
class Sample:

    def __init__(self, sortG, graphon=None, w_fct=None, N_fine=None, use_origFct=True):  # , Us_type=None, graphonEst=None, wEst_fct=None, graphonReal=None, wReal_fct=None
        # sortG = sorted extended graph, graphon = graphon used for posterior sampling,
        # w_fct = graphon function used for posterior sampling (alternatively to 'graphon'),
        # N_fine = fineness of the garphon discretization (for faster results),
        # use_origFct = logical whether to use original graphon function or discrete approx
        if sortG.sorting is None:
            warnings.warn('no specification about Us_type (see sortG.sorting), empirical degree ordering is used')
            print('UserWarning: no specification about Us_type (see sortG.sorting), empirical degree ordering is used')
            sortG.sort(Us_type='emp')
            warnings.warn('input graph is now sorted by empirical degree')
            print('UserWarning: input graph is now sorted by empirical degree')
        self.sortG = sortG
        self.labels = sortG.labels_()  # save for iteration -> allowing for re-identification
        if N_fine is None:
            N_fine = int(np.round((3.15 - 2 / (1 + np.exp(- 5e-3 * (self.sortG.N - 500)))) * self.sortG.N))
        if graphon is None:
            if w_fct is None:
                raise TypeError('no information about graphon')
            self.w_fct = w_fct if use_origFct else matToFct(mat=fctToMat(w_fct, size=N_fine))
            self.splitPos = None
        else:
            self.w_fct = graphon.fct if use_origFct else \
                matToFct(mat=graphon.mat if (graphon.byMat & (graphon.mat.shape[0] <= N_fine)) else fctToMat(graphon.fct, size=N_fine))
            if not w_fct is None:
                warnings.warn('function \'w_fct\' has not been used')
                print('UserWarning: function \'w_fct\' has not been used')
            try:
                self.splitPos = graphon.splitPos
            except AttributeError:
                self.splitPos = None
        self.U_MCMC, self.U_MCMC_all = None, None
        self.acceptRate = None
    def gibbs(self,steps=300,rep=10,proposal='logit_norm',sigma_prop=2,gamma=.5,splitPos=None,averageType='mean',returnAllGibbs=False,updateGraph=False,use_stdVals=None,printWarn=True):
        # steps = steps of Gibbs iterations, rep = number of repetitions/sequences, proposal = type of proposal (options: 'logit_norm', 'exclus_unif', 'uniform', 'mixture'),
        # sigma_prop = variance of sampling step (only for proposal='logit_norm' or 'mixture'), gamma = probability to propose new position within current group (only for proposal='mixture'),
        # splitPos = split positions according to the graphon (if there are any), averageType = type of averaging the Gibbs sampling sequence (options: 'mean', 'median', 'mode'), 
        # returnAllGibbs = logical whether to return all Gibbs sampling stages, updateGraph = logical whether input graph should be updated,
        # use_stdVals = logical whether to use standardized U's or as given from averaging directly, printWarn = logical whether to print warning when graph has been updated
        self.U_MCMC = np.zeros((rep, self.sortG.N))
        self.acceptRate = np.zeros(rep)
        if returnAllGibbs:
            self.U_MCMC_all = np.zeros((rep*steps, self.sortG.N))
        if splitPos is None:
            if self.splitPos is None:
                self.splitPos = np.array([0, 1])
                if ((proposal in ['exclus_unif', 'mixture']) or (averageType == 'mode')):
                    warnings.warn('splitPos has been automatically set to {0,1}')
                    print('UserWarning: splitPos has been automatically set to {0,1}')
        else:
            if not self.splitPos is None:
                warnings.warn('[].splitPos has been specified by argument \'splitPos\', splitPos attribute from input graphon has not been used')
                print('UserWarning: [].splitPos has been specified by argument \'splitPos\', splitPos attribute from input graphon has not been used')
            self.splitPos = splitPos
        u_t=self.sortG.Us_(self.sortG.sorting)
        for ival_i in range(len(self.splitPos)-1):
            u_t[np.logical_and(u_t >= self.splitPos[ival_i], u_t < self.splitPos[ival_i+1])] = np.minimum(np.maximum(u_t[np.logical_and(u_t >= self.splitPos[ival_i], u_t < self.splitPos[ival_i+1])], self.splitPos[ival_i]+1e-5), self.splitPos[ival_i+1]-1e-5)
        #u_t=np.minimum(np.maximum(self.sortG.Us_(self.sortG.sorting), 1e-5), 1-1e-5)
        for rep_step in range(rep):
            Decision = np.zeros(shape=[steps,self.sortG.N], dtype=bool)
            for step in range(steps):
                for k in np.random.permutation(np.arange(self.sortG.N)):
                    if proposal == 'logit_norm':
                        z_star_k=np.random.normal(loc=math.log(u_t[k]/(1-u_t[k])),scale=sigma_prop)
                        u_star_k=math.exp(z_star_k)/(1+math.exp(z_star_k))
                    elif proposal == 'exclus_unif':
                        ## uniform proposal over all other subintervals
                        ival_k = [(self.splitPos[i_k-1], self.splitPos[i_k]) for i_k in [np.min([np.max([np.searchsorted(self.splitPos,u_t[k]), 0]), len(self.splitPos) -1])]][0]
                        ival_diff = np.diff(ival_k)[0]
                        z_star_k=np.random.uniform(0,1 -ival_diff)
                        u_star_k=z_star_k + (ival_diff if (ival_k[0] <= z_star_k) else 0)
                    elif proposal == 'uniform':
                        u_star_k=np.random.uniform(0,1,1)
                    elif proposal == 'mixture':
                        if (u_t[k] >= self.splitPos[-1]):  # !!!
                            warnings.warn('u_k (=' + u_t[k].__str__() + ') is out of splitPos (=[' + ', '.join(self.splitPos.astype(str)) + '])')
                            print('UserWarning: u_k (=' + u_t[k].__str__() + ') is out of splitPos (=[' + ', '.join(self.splitPos.astype(str)) + '])')
                        ival_k = [(self.splitPos[i_k-1], self.splitPos[i_k]) for i_k in [np.min([np.max([np.searchsorted(self.splitPos,u_t[k]), 0]), len(self.splitPos) -1])]][0]
                        ival_diff = np.diff(ival_k)[0]
                        within_logic = (np.random.binomial(n=1,p=gamma) == 1) if (len(self.splitPos) > 2) else True
                        if within_logic:
                            z_star_k = np.random.normal(loc=math.log((u_t[k] -ival_k[0]) / (ival_k[1] - u_t[k])), scale=sigma_prop)
                            u_star_k = (math.exp(z_star_k) / (1 + math.exp(z_star_k))) * ival_diff +ival_k[0]
                        else:
                            z_star_k=np.random.uniform(0,1 -ival_diff)
                            u_star_k=z_star_k + (ival_diff if (ival_k[0] <= z_star_k) else 0)
                    else:
                        raise ValueError('proposal type \'' + str(proposal) + '\' is not implemented')
                    u_no_k=np.delete(u_t, k)
                    y_no_k=np.delete(self.sortG.A[k], k)
                    w_fct_star_k=np.minimum(np.maximum(self.w_fct(u_star_k,u_no_k), 1e-5), 1-1e-5)
                    w_fct_k=np.minimum(np.maximum(self.w_fct(u_t[k],u_no_k), 1e-5), 1-1e-5)
                    # for interpreter: 0**0 = 1 (by default)
                    prod_k=np.prod(np.squeeze(np.asarray(((w_fct_star_k/w_fct_k)**y_no_k))) * \
                        np.squeeze(np.asarray((((1-w_fct_star_k)/(1-w_fct_k))**(1-y_no_k)))))
                    if proposal == 'logit_norm':
                        alpha=min(1,prod_k*((u_star_k*(1-u_star_k))/(u_t[k]*(1-u_t[k]))))
                    if proposal == 'exclus_unif':
                        ## if uniform proposal over all other subintervals has been chosen
                        alpha=min(1,prod_k*((1-ival_diff)/(1-[(tau[i_k] - tau[i_k-1]) for i_k in [np.searchsorted(tau,u_star_k)]][0])))  # self.splitPos instead of tau (depends on how the SBM is designed)
                    if proposal == 'uniform':
                        alpha=min(1,prod_k)
                    if proposal == 'mixture':
                        if within_logic:
                            alpha=min(1,prod_k*(((u_star_k -ival_k[0])*(ival_k[1] -u_star_k))/((u_t[k] -ival_k[0])*(ival_k[1] -u_t[k]))))
                        else:
                            alpha = min(1, prod_k * ((1 - ival_diff) / (1 - [(self.splitPos[i_k] - self.splitPos[i_k - 1]) for i_k in [np.searchsorted(self.splitPos, u_star_k)]][0])))
                    Decision[step,k] = (np.random.binomial(n=1,p=alpha)==1)
                    if Decision[step,k]:
                        u_t[k] = np.min([np.max([u_star_k, 1e-5]), 1-1e-5])
                    if returnAllGibbs:
                        self.U_MCMC_all[rep_step * steps + step, k] = u_t[k]
            self.U_MCMC[rep_step] = u_t
            new_acceptRate = np.sum(Decision)/(self.sortG.N*steps)
            self.acceptRate[rep_step] = new_acceptRate
            print('Acceptance rate in repetition ' + (rep_step+1).__str__() + ' out of ' + rep.__str__() + ': ', new_acceptRate)
        if averageType == 'mean':
            self.Us_new = np.mean(self.U_MCMC, axis=0)
        elif averageType == 'median':
            self.Us_new = np.median(self.U_MCMC, axis=0)
        elif averageType == 'mode':
            self.Us_new = np.array([np.mean([self.splitPos[i - 1], self.splitPos[i]]) for ind_line in np.searchsorted(self.splitPos, self.U_MCMC).T for i in stats.mode(ind_line)[0]])
        else:
            raise ValueError('averaging method \'' + str(averageType) + '\' is not implemented')
        self.Us_new_std = np.zeros(self.sortG.N)
        memb1 = np.maximum(np.searchsorted(self.splitPos, self.Us_new), 1)
        for i in range(1,len(self.splitPos)):
            pos1 = memb1 == i
            self.Us_new_std[pos1] = np.linspace(self.splitPos[i-1],self.splitPos[i],pos1.sum()+2)[1:-1][np.argsort(np.argsort(self.Us_new[pos1]))]
        # self.Us_new_std = (np.linspace(0,1,self.sortG.N+2)[1:-1])[np.argsort(np.argsort(self.Us_new))]
        self.U_MCMC_std = np.zeros(self.U_MCMC.shape)
        memb2 = np.maximum(np.searchsorted(self.splitPos, self.U_MCMC), 1)
        for i in range(1,len(self.splitPos)):
            for j in range(self.U_MCMC.shape[0]):
                pos2 = memb2[j] == i
                self.U_MCMC_std[j][pos2] = np.linspace(self.splitPos[i-1],self.splitPos[i],pos2.sum()+2)[1:-1][np.argsort(np.argsort(self.U_MCMC[j][pos2]))]
        # self.U_MCMC_std = np.array([(np.linspace(0,1,self.sortG.N+2)[1:-1])[np.argsort(np.argsort(self.U_MCMC[i]))] for i in range(self.U_MCMC.shape[0])])
        if updateGraph:
            updateGraph(use_stdVals=use_stdVals)
            if printWarn:
                warnings.warn('U\'s from input graph have been updated')
                print('UserWarning: U\'s from input graph have been updated')
    def updateGraph(self, use_stdVals):
        try:
            self.sortG.update(Us_est=self.Us_new_std if use_stdVals else self.Us_new)  # only Us_est should be changed; if self.sortG.sorting=='real'_or_'emp' the result will anyway be saved as Us_est
        except AttributeError:
            warnings.warn('graph can only be updated after the Gibbs sampling has been executed, use [].gibbs()')
            print('UserWarning: graph can only be updated after the Gibbs sampling has been executed, use [].gibbs()')
    def showMove(self,Us_type=None,useAllGibbs=False,std=False,useColor=True,title=True,EMstep_sign=1,make_show=True,savefig=False,file_=None):
        if Us_type is None:
            Us_type = self.sortG.sorting
        Us_x = np.tile(self.sortG.Us_(Us_type), self.U_MCMC.shape[0] if useAllGibbs else 1)
        Us_y = np.hstack(self.U_MCMC_std if std else self.U_MCMC) if useAllGibbs else (self.Us_new_std if std else self.Us_new)
        col = plt.cm.binary(np.tile(self.sortG.Us_('real'), self.U_MCMC.shape[0] if useAllGibbs else 1)) if (useColor and (not self.sortG.Us_('real') is None)) else 'b'
        plot1 = plt.scatter(Us_x, Us_y, c= col)
        plot2 = plt.plot([0,1],[0,1])
        if title:
            plt.xlabel('$u_i$' if (Us_type == 'real') else ('$\hat{u}_i^{\;(' + (EMstep_sign-1).__str__() + ')}$'))
            plt.ylabel('$\hat{u}_i^{\;(' + EMstep_sign.__str__() + ')}$')
        plt.xlim((-1/20,21/20))
        plt.ylim((-1/20,21/20))
        plt.gca().set_aspect(np.abs(np.diff(plt.gca().get_xlim())/np.diff(plt.gca().get_ylim()))[0])
        plt.tight_layout()
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(plot1, plot2)
    def showPostDistr(self, ks_lab=None, ks_abs=None, Us_type_label=None, Us_type_add=None, distrN_fine=1000, useAllGibbs=True, EMstep_sign='(1)', figsize=None, mn_=None, useTightLayout=True, w_pad=2, h_pad = 1.5, make_show=True, savefig=False, file_=None):
        # ks_lab = labels of the nodes for which the posterior is calculated
        if not hasattr(self, 'Us_new'):
            raise TypeError('posterior distribution can only be calculated after the Gibbs sampling has been executed, use [].gibbs()')
        evalPoints = ((np.arange(distrN_fine) / distrN_fine) + (np.arange(distrN_fine + 1) / distrN_fine)[1:]) * (1 / 2)  # equidistant evaluation points
        if Us_type_label is None:
            Us_type_label = self.sortG.sorting
        if ks_lab is None:
            ks_lab = np.array([Ord_flip[ks_abs_i] for Ord_flip in [{i: j for j, i in list(eval('self.sortG.Ord_' + Us_type_label + '.items()'))}] for ks_abs_i in ks_abs])
        else:
            if not ks_abs is None:
                warnings.warn('k\'s have been specified by ks_lab, ks_abs has not been used')
                print('UserWarning: k\'s have been specified by ks_lab, ks_abs has not been used')
            ks_abs = np.array([eval('self.sortG.Ord_' + Us_type_label)[ks_lab_i] for ks_lab_i in ks_lab])
        n_k = len(ks_lab)
        distr_Uk = np.zeros((1, n_k, distrN_fine))
        for i_k in range(n_k):
            pos_ki = self.sortG.labels_() == ks_lab[i_k]
            y_no_k = np.squeeze(self.sortG.A[pos_ki])[np.invert(pos_ki)]
            Us_no_k = (self.U_MCMC if useAllGibbs else self.Us_new.reshape(1, self.sortG.N))[:, np.invert(pos_ki)]
            distrMat = np.array([[]]).reshape(0, distrN_fine)
            for i in (range(self.U_MCMC.shape[0]) if useAllGibbs else [0]):
                distr_Uk_uncorr = np.array([(probs**y_no_k * (1 - probs)**(1 - y_no_k)).prod(axis=1) for probs in [self.w_fct(evalPoints, Us_no_k[i])]])
                distrMat = np.row_stack((distrMat, distr_Uk_uncorr * len(distr_Uk_uncorr) / np.sum(distr_Uk_uncorr)))
            distr_Uk[0, i_k, :] = np.sum(distrMat, axis=0) * (1 / (self.U_MCMC.shape[0] if useAllGibbs else 1))
        distr_UkFinal = np.sum(distr_Uk, axis=0)
        distr_UkFinal_max = np.max(distr_UkFinal, axis=1)
        if Us_type_add is None:
            Us_type_add = self.sortG.sorting
        Us_consid = eval('self.sortG.Us_' + Us_type_add)
        u_ks = np.array([Us_consid[ks_lab_i] for ks_lab_i in ks_lab])  # [ordering is relevant]
        fig1 = plt.figure(1, figsize=figsize)
        ax_list = plot_list = line_list = []
        if mn_ is None:
            mn_ = int(np.ceil(n_k/np.ceil(np.sqrt(n_k)))).__str__() + int(np.ceil(np.sqrt(n_k))).__str__()
        for i_k in range(n_k):
            ax_list.append(plt.subplot(eval(mn_ + (i_k + 1).__str__())))
            plot_list.append(plt.plot(evalPoints, distr_UkFinal[i_k]))
            line_list.append(plt.axvline(x=u_ks[i_k], linestyle = '--'))
            try:  # ***
                plt.ylim((-distr_UkFinal_max[i_k] / 20, 27/20 *distr_UkFinal_max[i_k]))  # 25/20 *...
            except ValueError:
                warnings.warn('max. of posterior density could not be calculated')
                print('UserWarning: max. of posterior density could not be calculated')
                plt.gca().set_ylim(bottom=0)
            plt.text(((u_ks[i_k]) + 0.05) / 1.1, 0.9, transform=plt.gca().transAxes, s="{0:.4f}".format(u_ks[i_k]), horizontalalignment='center', fontsize = 10, bbox=dict(boxstyle='round', facecolor='white'))  # u_ks[i_k], 0.905
            if not i_k in range(n_k - int(np.ceil(np.sqrt(n_k))), n_k):
                plt.gca().get_xaxis().set_ticks([])
            if i_k in range(n_k - int(np.ceil(np.sqrt(n_k))), n_k):
                plt.xlabel('$u_{(k)}$')
            if ((i_k % int(np.ceil(np.sqrt(n_k)))) == 0):
                plt.ylabel('$\hat{f}_{(k)}^{\;' + EMstep_sign + '}(u_{(k)}\, |\, y)$')
            plt.title('$k = ' + (ks_abs[i_k]+1).__str__() + '$')
            # plt.title('$k = \hat{\psi}^{\;' + EMstep_sign + '}(' + (ks_abs[i_k]+1).__str__() + ')$')
        if useTightLayout:
            plt.tight_layout(w_pad=w_pad, h_pad=h_pad)
            # plt.subplots_adjust(top=top)
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(fig1, ax_list, plot_list, line_list)
#out: Sample Object
#     A = adjacency matrix, Us = U's used as start values, Us_real = real U's of the graph, N = order of the graph,
#     wEst_fct = estimated graphon function, wReal_fct = real graphon function,
#     U_MCMC = vector of Gibbs sampled U-vectors, U_MCMC_std = vector of standardized Gibbs sampled U-vectors ->[1/(N+1),...,N/(N+1)],
#     Us_new = mean over Gibbs sampling returns, Us_new_std = standardized version of Us_new ->[1/(N+1),...,N/(N+1)],
#     acceptRate = acceptance rate of new proposed/sampled values
#     gibbs = apply Gibbs sampling to start values for specified adj. matrix and graphon
#     showMove = plot comparison between (mean or vector of) sampled U's and start values or real U's

