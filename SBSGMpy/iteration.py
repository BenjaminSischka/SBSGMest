'''

Create the EM-type algorithm, meaning a routine that iterates between the sampling (E-) and the estimation (M-)step.
@author: Benjamin Sischka

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from copy import copy
import warnings
from graspologic.models import DCSBMEstimator
from SBSGMpy.fitting import estSplitPos, Estimator
from SBSGMpy.sampling import Sample

# Plot the development of the optimization with respect to the information criterion
def showOptForAIC(AIC_opt_lamb_list,AIC_opt_vals_list,m,make_show=True,savefig=False,file_=None):
    plots1 = []
    for i_ in range(len(AIC_opt_lamb_list)):
        plots1.append([])
        for j_ in range(len(AIC_opt_lamb_list[i_])):
            plots1[-1].append(plt.plot(AIC_opt_lamb_list[i_][j_], AIC_opt_vals_list[i_][j_],'-x')[0])
            plt.xlabel('$\lambda^{(' + m.__str__() + ')}_{' + (i_+1).__str__() + ',\,' + ((j_+i_)+1).__str__() + '}$')
            plt.ylabel('AIC')
            plt.tight_layout()
            if make_show:
                plt.show()
            if savefig:
                plt.savefig(file_ + '_' + m.__str__() + '_' + (i_+1).__str__() + '_' + ((j_+i_)+1).__str__() + '.png')
                plt.close(plt.gcf())
    if not savefig:
        return(plots1)


def iterateEM(sortG,
              k, nSubs, nKnots, useOneBasis, critType, canonical, est_initSplitPos, adjustSubs, adjustQuantiles,
              updateStrategy, n_steps, proposal, sigma_prop, averageType, max_steps, n_try, use_origFct, use_stdVals,
              n_maxIter, lambda_init, n_init, burnIn_init, rep_start, rep_grow, rep_forPost, n_eval,
              init_allRandom=False, stopVal=.0025, useDCSBM=False,
              splitPos=None, splitPos_real=None, lambdaLim_start=[5., 500.],
              returnLambList=True, returnGraphonList=False, returnSampList=False, returnAllGibbs=False,
              makePlots=False, make_show=None, savefig=False, simulate=None, log_scale=False, dir_=None):
    lambdaList, estGraphonList, sampleList = np.zeros((0, nSubs, nSubs)), [], []
    trajMat = np.zeros((0, n_eval, n_eval))
    if nSubs == 1:
        useOneBasis, est_initSplitPos, adjustSubs = True, False, False
        proposal = 'logit_norm' if (proposal == 'mixture') else ('uniform' if (proposal == 'exclus_unif') else proposal)
        averageType = 'mean' if (averageType == 'mode_mean') else averageType
    AIC_vec, doRep_logic = np.array([]), True
    rep = [0, 0]
    n_maxIter = n_maxIter + n_init
    index = 1
    ### EM based algorithm
    while (index <= n_maxIter):
        ### Update graph
        if index == 1:
            if useDCSBM:
                dcsbm_res = DCSBMEstimator(directed=False, loops=False, min_comm=nSubs, max_comm=nSubs)
                dcsbm_res.fit(sortG.A)
                counts_ = np.unique(dcsbm_res.vertex_assignments_, return_counts=True)
                if len(counts_[0]) != nSubs:
                    raise TypeError('degree-corrected stochastic blockmodel contains empty groups')
                if not np.all([(counts_[0][i_] in range(nSubs)) for i_ in range(nSubs)]):
                    raise TypeError('labeling of groups is not as expected, should be [0, ..., nSubs]')
                splitPos_ = np.cumsum(np.append([0], counts_[1])) / sortG.N
                Us_est_ = np.zeros(sortG.N)
                for gr_i in range(nSubs):
                    subIndices_logic = dcsbm_res.vertex_assignments_ == gr_i
                    Us_est_[subIndices_logic] = np.linspace(splitPos_[gr_i], splitPos_[gr_i + 1], counts_[1][gr_i] + 2)[1:-1][np.argsort(np.argsort(np.squeeze(dcsbm_res.degree_corrections_)[subIndices_logic]))]
                sortG.update(Us_est=Us_est_)
                if sortG.sorting != 'est':
                    sortG.sort('est')
                if not est_initSplitPos:
                    splitPos_ = np.linspace(0, 1, nSubs + 1)
                if splitPos is None:
                    splitPos = splitPos_
            else:
                if splitPos is None:
                    ## consider splitPos in Estimator.GraphonEstBySpline(..., est_splitPos=True, ...)
                    if est_initSplitPos:
                        splitPos = estSplitPos(sortG=sortG, nSubs=nSubs)
                    else:
                        splitPos = np.linspace(0, 1, nSubs + 1)
        else:
            print('Update graph')
            sample.updateGraph(use_stdVals=use_stdVals)
            if index in (([int(round(n_init / 2)) + 1] if (n_init >= 5) else []) + \
                         ([int(round(n_init * 1 / 3)) + 1, int(round(n_init * 3 / 4)) + 1] if (n_init >= 10) else []) + \
                         ([n_init + 1] if (n_init >= 1) else [])):
                splitPos = estSplitPos(sortG=sortG, nSubs=nSubs)
            else:
                splitPos = copy(estGraphon.splitPos) if hasattr(estGraphon, 'splitPos') else None
                ## force the algorithm to keep the given number of communities
                if not splitPos is None:
                    grInd = np.searchsorted(splitPos, sortG.Us_(sortG.sorting))
                    count_obj = np.unique(grInd, return_counts=True)
                    if not np.all([(gr_i in count_obj[0]) for gr_i in np.arange(1, nSubs + 1)]):
                        emptyGr = np.array([(not (gr_i in count_obj[0])) for gr_i in np.arange(1, nSubs + 1)])
                        splitPos_rmv = (splitPos[np.concatenate((np.logical_not(emptyGr), [True]))] + splitPos[np.concatenate(([True], np.logical_not(emptyGr)))]) / 2
                        splitPos_rmv[0], splitPos_rmv[-1] = 0., 1.
                        splitPos_new = estSplitPos(nSubs=nSubs, sortG=sortG, splitPos=splitPos_rmv)
                        newSplits = splitPos_new[[(not np.any(np.isclose(splitPos_new_i, splitPos_rmv))) for splitPos_new_i in splitPos_new]]
                        warnings.warn('[result nb ' + (index - 1).__str__() + '] empty communit' + (('y ' + (np.where(emptyGr)[0][0] + 1).__str__() + ' with interval ' + str(splitPos[np.where(emptyGr)[0][0] + np.array([0, 1])]) + ' has') if (emptyGr.sum() == 1) else \
                                                                                                  ('ies ' + ', '.join((np.where(emptyGr)[0] + 1).astype(str)) + ' with intervals ' + ', '.join([str(splitPos[epty_i + np.array([0, 1])]) for epty_i in np.where(emptyGr)[0]]) + ' have')) + \
                                      ' been removed;\n' + ('a new split has' if (emptyGr.sum() == 1) else 'new splits have') + ' been inserted at ' + ', '.join(map(lambda x: str(round(x, 4)), newSplits)))
                        print('UserWarning: [result nb ' + (index - 1).__str__() + '] empty communit' + (('y ' + (np.where(emptyGr)[0][0] + 1).__str__() + ' with interval ' + str(splitPos[np.where(emptyGr)[0][0] + np.array([0, 1])]) + ' has') if (emptyGr.sum() == 1) else \
                                                                                                             ('ies ' + ', '.join((np.where(emptyGr)[0] + 1).astype(str)) + ' with intervals ' + ', '.join([str(splitPos[epty_i + np.array([0, 1])]) for epty_i in np.where(emptyGr)[0]]) + ' have')) + \
                              ' been removed;\n' + ('a new split has' if (emptyGr.sum() == 1) else 'new splits have') + ' been inserted at ' + ', '.join(map(lambda x: str(round(x, 4)), newSplits)))
                        splitPos = splitPos_new
                        if not (estGraphon.tau_sep is None):
                            if not np.all([np.allclose(np.diff(tau_sep_i[k:-k]), np.repeat(tau_sep_i[k + 1] - tau_sep_i[k], len(tau_sep_i[k:-k]) - 1)) for tau_sep_i in estGraphon.tau_sep if (len(tau_sep_i[k:-k]) > 0)]):
                                warnings.warn('manually adapted knot positions have been discarded')
                                print('UserWarning: manually adapted knot positions have been discarded')
                        if not np.isscalar(estGraphon.lambda_):
                            estGraphon.lambda_ = [estGraphon.lambda_[np.logical_not(emptyGr)][:,np.logical_not(emptyGr)][lmbd_selction][:,lmbd_selction] \
                                                  for lmbd_selction in [np.sort(np.append(np.arange(np.logical_not(emptyGr).sum()), np.searchsorted(splitPos_rmv, newSplits) - 1))]][0]
        if makePlots:
            sortG.showAdjMat(make_show=make_show, savefig=savefig, file_=dir_ + 'adjMat_' + (index - 1).__str__() + '.png')
            if not (make_show or savefig):
                plt.clf()
            sortG.showNet(splitPos=splitPos, make_show=make_show, savefig=savefig, file_=dir_ + 'network_' + (index - 1).__str__() + '.png')
            if not (make_show or savefig):
                plt.clf()
            if simulate:
                sortG.showDiff(Us_type='est', splitPos_est=splitPos, splitPos_real=splitPos_real, EMstep_sign='(' + (index - 1).__str__() + ')', make_show=make_show, savefig=savefig, file_=dir_ + 'Us_diffReal_' + (index - 1).__str__() + '.png')
                if not (make_show or savefig):
                    plt.clf()
            if adjustSubs or adjustQuantiles:
                sortG.showUsCDF('est', make_show=make_show, savefig=savefig, file_=dir_ + 'Us_cdf_' + (index - 1).__str__() + '.png')
                if not (make_show or savefig):
                    plt.clf()
                sortG.showUsHist(bins=(sortG.N/20) if (adjustQuantiles or (splitPos is None)) else splitPos, make_show=make_show, savefig=savefig, file_=dir_ + 'Us_hist_' + (index - 1).__str__() + '.png')
                if not (make_show or savefig):
                    plt.clf()
        ### Estimate Graphon
        print('Estimation')
        if index == 1:
            tau = None
            tau_sep = None
        else:
            if adjustSubs:
                tau = None
                tau_sep = None
            elif adjustQuantiles:
                tau = copy(estGraphon.tau) if hasattr(estGraphon, 'tau') else None
                tau_sep = None
                splitPos = None
                nKnots = None
            else:
            ## note: 'else' means adjustSubs=adjustQuantiles=False since adjustSubs=adjustQuantiles=True is excluded through Estimator.GraphonEstBySpline
                if useOneBasis:
                    tau = copy(estGraphon.tau) if hasattr(estGraphon, 'tau') else None
                    tau_sep = None
                else:
                    tau = None
                    tau_sep = copy(estGraphon.tau_sep) if hasattr(estGraphon, 'tau_sep') else None
                splitPos = None
                nKnots = None
        if index <= n_init:
            if index in (int(round(n_init / 2)) + np.arange(-1,2)):
                lambda_ = lambda_init[1]
            else:
                lambda_ = lambda_init[0]
            averageType_ = 'mean'
            sigma_prop_ = sigma_prop
            gamma = 1 / nSubs
            if init_allRandom:
                burnIn = burnIn_init
            else:
                burnIn = int(round((1 - ((index - 1) / (n_init - 1))) * burnIn_init))
            rep[0], rep[1] = rep[1], index
            optForAIC = False
            lambda_adjustSubs = lambda_adjustQuant = (1/5) if (adjustSubs or adjustQuantiles) else None
            lambdaMin, lambdaMax = None, None
        else:
            averageType_ = averageType
            sigma_prop_ = sigma_prop
            gamma = 1 / nSubs
            burnIn = 0
            rep[0] = rep[1]
            rep[1] += (np.max([0, rep_start - rep[1]]) if (index == n_init + 1) else rep_grow) if (index < n_maxIter) else (rep_forPost - rep[1])
            optForAIC = True
            lambda_adjustSubs = lambda_adjustQuant = 1. if (adjustSubs or adjustQuantiles) else None
            [lambdaMin, lambdaMax] = lambdaLim_start if (index == n_init + 1) else ([np.maximum(5., estGraphon.lambda_ * fctr_i) for fctr_i in [.33, 3]] if hasattr(estGraphon, 'lambda_') else [None, None])
        estGraphonData=Estimator(sortG=sortG)
        estGraphon=estGraphonData.GraphonEstBySpline(k=k, nSubs=nSubs, nKnots=nKnots, splitPos=splitPos, est_splitPos=False, useOneBasis=useOneBasis, tau=tau, tau_sep=tau_sep,
                                                     optForAIC=optForAIC, lambdaMin=lambdaMin, lambdaMax=lambdaMax, calcAIC=True, lambda_=(None if optForAIC else lambda_), critType=critType,
                                                     adjustSubs=adjustSubs, lambda_adjustSubs=lambda_adjustSubs, adjustQuantiles=adjustQuantiles, lambda_adjustQuant=lambda_adjustQuant,
                                                     Us_mult=None, canonical=canonical, updateGraph=True, printWarn=False)
        if makePlots:
            estGraphon.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dir_ + 'graphon_est_' + index.__str__() + '.png')
            if not (make_show or savefig):
                plt.clf()
        ## force the algorithm to keep the given number of communities [should not occur when split positions are already updated, see above]
        if np.any(estGraphon.freqUsSub == 0):
            emptyGr = (estGraphon.freqUsSub == 0)
            splitPos_rmv = (estGraphon.splitPos[np.concatenate((np.logical_not(emptyGr), [True]))] + estGraphon.splitPos[np.concatenate(([True], np.logical_not(emptyGr)))]) / 2
            splitPos_rmv[0], splitPos_rmv[-1] = 0., 1.
            us_ = np.concatenate([np.linspace(estGraphon.splitPos[i],estGraphon.splitPos[i+1], int(np.round(estGraphonData.sortG.N*((estGraphon.splitPos[i+1]-estGraphon.splitPos[i])/(1-sizeOff)))) +2)[1:-1] for sizeOff in [np.sum([(estGraphon.splitPos[i+1]-estGraphon.splitPos[i]) for i in range(estGraphon.nSubs) if emptyGr[i]])] for i in range(estGraphon.nSubs) if (not emptyGr[i])])
            vs_list = [(tau_sep_i[k:-k] + np.concatenate(([1e-5],np.repeat(0,len(tau_sep_i[k:-k])-2),[-1e-5]))) for tau_sep_i in np.array(estGraphon.tau_sep)[np.logical_not(emptyGr)]]
            lenVs = np.array([len(vs_list_i) for vs_list_i in vs_list])
            vs_ = np.concatenate(vs_list)
            probMat = estGraphon.fct(us_, vs_)
            newSplits = [vs_list[group_i][val_i - (np.append([0], np.cumsum(lenVs - 1))[group_i]) + np.array([0, 1])].mean() for val_i in (np.delete(np.diff(probMat), np.cumsum(lenVs)[:-1] - 1, axis=1) ** 2).sum(axis=0).argsort()[-np.sum(emptyGr):] for group_i in [np.sum(np.cumsum(lenVs - 1) < (val_i + 1))]]
            warnings.warn('[result nb ' + index.__str__() + '] empty communit' + (('y ' + (np.where(emptyGr)[0][0]+1).__str__() + ' with interval ' + str(estGraphon.splitPos[np.where(emptyGr)[0][0] + np.array([0, 1])]) + ' has') if (emptyGr.sum() == 1) else \
                                                                                      ('ies ' + ', '.join((np.where(emptyGr)[0]+1).astype(str)) + ' with intervals ' + ', '.join([str(estGraphon.splitPos[epty_i + np.array([0, 1])]) for epty_i in np.where(emptyGr)[0]]) + ' have')) + \
                          ' been removed;\n' + ('a new split has' if (emptyGr.sum() == 1) else 'new splits have') + ' been inserted at ' + ', '.join(map(lambda x: str(round(x,4)), newSplits)))
            print('UserWarning: [result nb ' + index.__str__() + '] empty communit' + (('y ' + (np.where(emptyGr)[0][0]+1).__str__() + ' with interval ' + str(estGraphon.splitPos[np.where(emptyGr)[0][0] + np.array([0, 1])]) + ' has') if (emptyGr.sum() == 1) else \
                                                                                           ('ies ' + ', '.join((np.where(emptyGr)[0]+1).astype(str)) + ' with intervals ' + ', '.join([str(estGraphon.splitPos[epty_i + np.array([0, 1])]) for epty_i in np.where(emptyGr)[0]]) + ' have')) + \
                  ' been removed;\n' + ('a new split has' if (emptyGr.sum() == 1) else 'new splits have') + ' been inserted at ' + ', '.join(map(lambda x: str(round(x,4)), newSplits)))
            newSplitPos = np.sort(np.append(splitPos_rmv, newSplits))
            if not np.all([np.allclose(np.diff(tau_sep_i[k:-k]), np.repeat(tau_sep_i[k+1]-tau_sep_i[k], len(tau_sep_i[k:-k])-1)) for tau_sep_i in estGraphon.tau_sep if (len(tau_sep_i[k:-k]) > 0)]):
                warnings.warn('manually adapted knot positions have been discarded')
                print('UserWarning: manually adapted knot positions have been discarded')
            if (lambdaMin is not None) and (not np.isscalar(lambdaMin)):
                lambdaMin = [lambdaMin[np.logical_not(emptyGr)][:, np.logical_not(emptyGr)][lmbd_selction][:, lmbd_selction] \
                             for lmbd_selction in [np.sort(np.append(np.arange(np.logical_not(emptyGr).sum()), np.searchsorted(splitPos_rmv, newSplits) - 1))]][0]
            if (lambdaMax is not None) and (not np.isscalar(lambdaMax)):
                lambdaMax = [lambdaMax[np.logical_not(emptyGr)][:, np.logical_not(emptyGr)][lmbd_selction][:, lmbd_selction] \
                             for lmbd_selction in [np.sort(np.append(np.arange(np.logical_not(emptyGr).sum()), np.searchsorted(splitPos_rmv, newSplits) - 1))]][0]
            estGraphon=estGraphonData.GraphonEstBySpline(k=k, nSubs=nSubs, nKnots=nKnots, splitPos=newSplitPos, est_splitPos=False, useOneBasis=useOneBasis, tau=None, tau_sep=None,
                                                         optForAIC=optForAIC, lambdaMin=lambdaMin, lambdaMax=lambdaMax, calcAIC=True, lambda_=(None if optForAIC else lambda_), critType=critType,
                                                         adjustSubs=adjustSubs, lambda_adjustSubs=lambda_adjustSubs, adjustQuantiles=adjustQuantiles, lambda_adjustQuant=lambda_adjustQuant,
                                                         Us_mult=None, canonical=canonical, updateGraph=True, printWarn=False)
            if makePlots:
                estGraphon.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dir_ + 'graphon_est_' + index.__str__() + '_newSplit.png')
                if not (make_show or savefig):
                    plt.clf()
        ## remove marginalized segments [should not occur when a certain number of communities is forced, see above]
        if np.any([np.allclose(tau_sep_i, tau_sep_i[k]) for tau_sep_i in estGraphon.tau_sep]):
            emptySub = np.array([np.allclose(tau_sep_i, tau_sep_i[k]) for tau_sep_i in estGraphon.tau_sep])
            warnings.warn('[result nb ' + index.__str__() + '] empty segment' + ((' ' + (np.where(emptySub)[0][0]+1).__str__() + ' with interval ' + str(estGraphon.splitPos[np.where(emptySub)[0][0] + np.array([0, 1])]) + ' has') if (emptySub.sum() == 1) else \
                                                                                     ('s ' + ', '.join((np.where(emptySub)[0]+1).astype(str)) + ' with intervals ' + ', '.join([str(estGraphon.splitPos[epty_i + np.array([0, 1])]) for epty_i in np.where(emptySub)[0]]) + ' have')) + \
                          ' been removed')
            print('UserWarning: [result nb ' + index.__str__() + '] empty segment' + ((' ' + (np.where(emptySub)[0][0]+1).__str__() + ' with interval ' + str(estGraphon.splitPos[np.where(emptySub)[0][0] + np.array([0, 1])]) + ' has') if (emptySub.sum() == 1) else \
                                                                                     ('s ' + ', '.join((np.where(emptySub)[0]+1).astype(str)) + ' with intervals ' + ', '.join([str(estGraphon.splitPos[epty_i + np.array([0, 1])]) for epty_i in np.where(emptySub)[0]]) + ' have')) + \
                          ' been removed')
            tau_sep_new = list(np.array(estGraphon.tau_sep)[np.logical_not(emptySub)])
            tau_new = np.concatenate([tau_sep_new[i][(0 if (i == 0) else (k + 1)):] for i in range(len(tau_sep_new))])
            nKnots_new = estGraphon.nKnots[np.logical_not(emptySub)]
            warnings.warn('number of groups has been reduced from ' + nSubs.__str__() + ' to ' + (nSubs - emptySub.sum()).__str__())
            print('UserWarning: number of groups has been reduced from ' + nSubs.__str__() + ' to ' + (nSubs - emptySub.sum()).__str__())
            nSubs = nSubs - emptySub.sum()
            if (lambdaMin is not None) and (not np.isscalar(lambdaMin)):
                lambdaMin = lambdaMin[np.logical_not(emptySub)][:,np.logical_not(emptySub)]
            if (lambdaMax is not None) and (not np.isscalar(lambdaMax)):
                lambdaMax = lambdaMax[np.logical_not(emptySub)][:,np.logical_not(emptySub)]
            estGraphon = estGraphonData.GraphonEstBySpline(k=k, nSubs=nSubs, nKnots=nKnots_new, splitPos=None, est_splitPos=False, useOneBasis=useOneBasis, tau=tau_new, tau_sep=tau_sep_new,
                                                           optForAIC=optForAIC, lambdaMin=lambdaMin, lambdaMax=lambdaMax, calcAIC=True, lambda_=(None if optForAIC else lambda_), critType=critType,
                                                           adjustSubs=adjustSubs, lambda_adjustSubs=lambda_adjustSubs, adjustQuantiles=adjustQuantiles, lambda_adjustQuant=lambda_adjustQuant,
                                                           Us_mult=None, canonical=canonical, updateGraph=True, printWarn=False)
            if makePlots:
                estGraphon.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dir_ + 'graphon_est_' + index.__str__() + '_remSub.png')
                if not (make_show or savefig):
                    plt.clf()
            lambdaList = np.zeros((0, nSubs, nSubs)) if (lambdaList.shape[0] == 0) else np.array([lambda_i[np.logical_not(emptySub)][:, np.logical_not(emptySub)] for lambda_i in lambdaList])
        if returnLambList:
            lambdaList = np.concatenate((lambdaList, np.expand_dims((np.ones((nSubs, nSubs)) * estGraphon.lambda_) if np.isscalar(estGraphon.lambda_) else estGraphon.lambda_, axis=0)), axis=0)
        if returnGraphonList:
            estGraphonList.append(estGraphon)
        trajMat = np.append(trajMat, estGraphon.fct(np.arange(1, n_eval+1) / (n_eval+1), np.arange(1, n_eval+1) / (n_eval+1)).reshape(1, n_eval, n_eval), axis=0)
        AIC_vec = np.append(AIC_vec, [estGraphon.critValue])
        if optForAIC:
            if makePlots:
                showOptForAIC(AIC_opt_lamb_list=estGraphon.AIC_opt_lamb_list, AIC_opt_vals_list=estGraphon.AIC_opt_vals_list, m=index, make_show=make_show, savefig=savefig, file_=dir_ + 'AIC_OptProfile')
                if not (make_show or savefig):
                    plt.clf()
        if index >= 3:
            stopCrit = np.max(np.abs((AIC_vec[-2:] - AIC_vec[(-3):(-1)]) / AIC_vec[(-3):(-1)]))
            doRep_logic = stopCrit > stopVal
        eval("print('\\n' + ('initial ' if (index <= n_init) else '') + 'iteration completed:', index - (0 if (index <= n_init) else n_init), '/', n_init if (index <= n_init) else (n_maxIter - n_init), " + \
             "'\\n' + critType + ':', np.round(AIC_vec[-1], 1), " + ("'[ last iteration:', np.round(AIC_vec[-2], 1), ']', " if (index >= 2) else "") + \
             ("'\\nstop criterion:', np.round(stopCrit, 5), '>' if doRep_logic else '<=', np.round(stopVal, 5), " if (index >= 3) else "") + \
             "'\\npenalizing parameter lambda:', " + ("" if np.isscalar(estGraphon.lambda_) else "'\\n', ") + "estGraphon.lambda_, '\\nnumber of " + \
             ("Gibbs sampling stages:', rep[0]" if (updateStrategy == 'gibbs') else "greedy search trials:', n_try") + \
             (", '\\nparameter for step-wise adjustment of subareas:', np.round(lambda_adjustSubs, 3)" if adjustSubs else "") + \
             (", '\\nparameter for step-wise quantile adjustment:', np.round(lambda_adjustQuant, 3)" if adjustQuantiles else "") + \
             (", '\\n\\ncalculate posterior distribution:\\n'" if ((((not doRep_logic) and (index > n_init + 1)) or (index == n_maxIter)) and (rep_forPost > 0)) else "") + ", '\\n')")
        if (not doRep_logic) and (index > n_init + 1):
            if (rep_forPost > 0):
                rep[1] = rep_forPost
            else:
                break
        ### Sample U's
        if (index <= n_init) and init_allRandom:
            sortG.update(Us_est = np.random.uniform(0,1,sortG.N))
        if (index < n_maxIter) or (rep[1] > 0):
            print('Sampling')
            sample=Sample(sortG=sortG,graphon=estGraphon,use_origFct=use_origFct)
            if updateStrategy == 'gibbs':
                sample.gibbs(steps=n_steps,burnIn=burnIn,rep=rep[1],proposal=proposal,sigma_prop=sigma_prop_,gamma=gamma,splitPos=None,averageType=averageType_,returnAllGibbs=returnAllGibbs,updateGraph=False,use_stdVals=None,printWarn=False)
            elif updateStrategy == 'greedy':
                sample.greedy(max_steps=max_steps,n_try=n_try,useSubsmpl=True, alpha_subsmpl=None, splitPos=None, updateGraph=False, use_stdVals=None, printWarn=False)
            else:
                raise ValueError('update strategy \'' + str(updateStrategy) + '\' is not implemented')
            if makePlots:
                sample.showMove(useColor=True if simulate else False, showSplitPos=True, EMstep_sign=index, make_show=make_show, savefig=savefig, file_=dir_ + 'Us_move_' + index.__str__() + '.png')
                if not (make_show or savefig):
                    plt.clf()
                if simulate:
                    sample.showMove(Us_type='real', useAllGibbs=True, useColor=True, showSplitPos=True, splitPos_real=splitPos_real, EMstep_sign=index, make_show=make_show, savefig=savefig, file_=dir_ + 'UsMCMC_diffReal_' + index.__str__() + '.png')
                    if not (make_show or savefig):
                        plt.clf()
            if returnSampList:
                sampleList.append(sample)
        if (not doRep_logic) and (index > n_init + 1):
            break
        index += 1
    if index > n_maxIter:
        warnings.warn('EM algorithm did not converge; recent relative change in criterion: ' + \
                      np.round(stopCrit, 5).__str__() + ' > ' + np.round(stopVal, 5).__str__())
        print('UserWarning: EM algorithm did not converge; recent relative change in criterion: ' + \
              np.round(stopCrit, 5).__str__() + ' > ' + np.round(stopVal, 5).__str__())
    result = type('', (), {})()
    result.params = type('', (), {})()
    result.params.n_iter, result.params.rep = index - (1 - (not doRep_logic)) - n_init, rep[0]
    result.sortG = sortG
    result.estGraphon = estGraphon
    result.sample = sample
    result.trajMat = trajMat
    result.AIC_vec = AIC_vec
    if returnLambList:
        result.lambdaList = lambdaList
    if returnGraphonList:
        result.estGraphonList = estGraphonList
    if returnSampList:
        result.sampleList = sampleList
    return(result)


# Plot the trajectory of the sequence of w^{(m)}(u,v) for exemplary positions u and v
def showTraject(trajMat,us_=None,make_show=True,savefig=False,file_=None):
    n_eval = trajMat.shape[1]
    if us_ is None:
        us_ = np.arange(1, n_eval+1) / (n_eval+1)
    it_nbs = np.arange(1, trajMat.shape[0] +1)
    plots1 = [plt.plot(it_nbs, trajMat[:,r,s], label = '$\hat{w}^{\;(m)}(' + np.round(us_[r],3).__str__() + ',\, ' + np.round(us_[s],3).__str__() + ')$') for r in range(n_eval) for s in range(n_eval) if s >= r]
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_ticks = np.concatenate(([1], ax.get_xticks()[2:-1]))
    ax.set_xticks(x_ticks)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    legend1 = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('EM Iteration $m$')
    plt.tight_layout()
    if make_show:
        plt.show()
    if savefig:
        plt.savefig(file_, bbox_extra_artists=(legend1,), bbox_inches='tight')
        plt.close(plt.gcf())
    else:
        return(plots1, legend1)


# Plot the sequence of the penalization parameter lambda_
def showLambda(lambdaList,make_show=True,savefig=False,file_=None):
    logicalSmooth = (lambdaList.ndim == 1)
    if logicalSmooth:
        lambdaList = lambdaList[:,np.newaxis,np.newaxis]
    nSubs = lambdaList.shape[1]
    groups_ = np.arange(1, nSubs+1)
    it_nbs = np.arange(1, lambdaList.shape[0] +1)
    plots1 = [plt.plot(it_nbs, lambdaList[:,r,s], label = '$\lambda^{(m)}_{' + groups_[r].__str__() + ',\,' + groups_[s].__str__() + '}$') for r in range(nSubs) for s in range(nSubs) if s >= r]
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_ticks = np.concatenate(([1], ax.get_xticks()[2:-1]))
    ax.set_xticks(x_ticks)
    if not logicalSmooth:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        legend1 = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('EM Iteration $m$')
    plt.tight_layout()
    if make_show:
        plt.show()
    if savefig:
        if logicalSmooth:
            plt.savefig(file_)
        else:
            plt.savefig(file_, bbox_extra_artists=(legend1,), bbox_inches='tight')
        plt.close(plt.gcf())
    else:
        if logicalSmooth:
            return(plots1)
        else:
            return(plots1, legend1)


# Plot the sequence of the applied information criterion
def showAIC(AIC_vec,make_show=True,savefig=False,file_=None):
    it_nbs = np.arange(1, len(AIC_vec) +1)
    plot1 = plt.plot(it_nbs, AIC_vec)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_ticks = np.concatenate(([1], ax.get_xticks()[2:-1]))
    ax.set_xticks(x_ticks)
    plt.xlabel('EM Iteration $m$')
    plt.ylabel('AIC')
    plt.tight_layout()
    if make_show:
        plt.show()
    if savefig:
        plt.savefig(file_)
        plt.close(plt.gcf())
    else:
        return(plot1)

