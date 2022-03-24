'''

Run the EM based algorithm -> iteration between the sampling (E-) and the estimation (M-) step.
@author: sischkab

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from copy import copy
import warnings
from SBSGMpy.fitting import Estimator
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
              k, nSubs, nKnots, useOneBasis, canonical, est_initSplitPos, adjustSubs, adjustQuantiles,
              n_steps, proposal, sigma_prop, use_origFct, averageType, use_stdVals,
              n_iter, rep_start, rep_end, it_rep_grow, rep_forPost,
              lambda_start, lambda_skip1, lambda_lim1, lambda_skip2, lambda_lim2, lambda_last_m,
              n_eval, trajMat=None,
              startWithEst=True, estGraphon=None, endWithSamp=True, raiseLabNb=False,
              returnLambList=True, returnGraphonList=False, returnSampList=False, returnAllGibbs=False,
              lambdaList=None, estGraphonList=[], sampleList=[],
              makePlots=False, make_show=None, savefig=False, simulate=None, log_scale=False, dir_=None):
    lambdas_ = None
    if trajMat is None:
        trajMat = np.zeros((0, n_eval, n_eval))
    if trajMat.shape[1:] != (n_eval,n_eval):
        raise TypeError('dimension of trajMat does not match n_eval')
    AIC_vec = np.array([])
    if not startWithEst:
        n_iter = n_iter+1
    # EM based algorithm
    for index in range(1,n_iter+1):
        labNb = index + raiseLabNb
        ## global phases:
        glob_phase1 = (index == 1)                                                      # phase 1a: index == 1
        glob_NotPhase1 = (not glob_phase1)
        ## phases of lambda_:
        lmd_phase1 = (index <= lambda_skip1)                                            # phase 1b: index <= lambda_skip1
        lmd_phase2 = ((index > lambda_skip1) and (index <= lambda_lim1))                # phase 2: lambda_skip1 < index <= lambda_lim1
        #lmd_phase3 = ((index > lambda_lim1) and (index <= lambda_skip2))               # phase 3: lambda_lim1 < index <= lambda_skip2
        lmd_phase4a = ((index > lambda_skip2) and (index <= lambda_lim2))               # phase 4a: lambda_skip2 < index <= lambda_lim2
        lmd_phase4b = (index == lambda_lim2)                                            # phase 4b: index == lambda_lim2
        #lmd_phase5a = ((index > lambda_lim2) and (index <= (n_iter-lambda_last_m)))    # phase 5a: lambda_lim2 < index <= n_iter-lambda_last_m
        lmd_phase5b = (index == (n_iter-lambda_last_m))                                 # phase 5b: index == n_iter-lambda_last_m
        lmd_phase6 = (index > (n_iter-lambda_last_m))                                   # phase 6: n_iter-lambda_last_m < index
        ### Update the graph
        if glob_NotPhase1:  # index > 1
            print('Update graph')
            sample.updateGraph(use_stdVals =use_stdVals)
            if makePlots:
                sortG.showAdjMat(make_show=make_show, savefig=savefig, file_=dir_ + 'adjMat_' + (labNb-1).__str__() + '.png')
                if hasattr(estGraphon, 'splitPos'):
                    splitPos1 = copy(estGraphon.splitPos)
                else:
                    splitPos1 = None
                sortG.showNet(splitPos=splitPos1, make_show=make_show, savefig=savefig, file_=dir_ + 'network_' + (labNb-1).__str__() + '.png')
                if simulate:
                    sortG.showDiff(Us_type='est', EMstep_sign='(' + (labNb-1).__str__() + ')', make_show=make_show, savefig=savefig, file_=dir_ + 'Us_diffReal_' + (labNb-1).__str__() + '.png')
                if adjustSubs or adjustQuantiles:
                    sortG.showUsCDF('est', make_show=make_show, savefig=savefig, file_=dir_ + 'Us_cdf_' + labNb.__str__() + '.png')
                    sortG.showUsHist(bins=((sortG.N/20) if (adjustQuantiles or (splitPos1 is None)) else splitPos1), make_show=make_show, savefig=savefig, file_=dir_ + 'Us_hist_' + labNb.__str__() + '.png')
        ### Estimate Graphon
        if (glob_NotPhase1 or startWithEst):  # (index > 1)
            print('Estimation')
            if (estGraphon is None):
                tau = None
                tau_sep = None
                splitPos = None
            else:
                if (adjustSubs and (not adjustQuantiles)):
                    tau = None
                    tau_sep = None
                    splitPos = copy(estGraphon.splitPos) if hasattr(estGraphon, 'splitPos') else None
                elif ((not adjustSubs) and adjustQuantiles):
                    tau = copy(estGraphon.tau) if hasattr(estGraphon, 'tau') else None
                    tau_sep = None
                    splitPos = None
                    nKnots = copy(estGraphon.nKnots) if hasattr(estGraphon, 'nKnots') else None
                else:
                # note: 'else' means adjustSubs=adjustQuantiles=False since adjustSubs=adjustQuantiles=True is excluded through Estimator.GraphonEstBySpline
                    if useOneBasis:
                        tau = copy(estGraphon.tau) if hasattr(estGraphon, 'tau') else None
                        tau_sep = None
                    else:
                        tau = None
                        tau_sep = copy(estGraphon.tau_sep) if hasattr(estGraphon, 'tau_sep') else None
                    splitPos = copy(estGraphon.splitPos) if hasattr(estGraphon, 'splitPos') else None
                    nKnots = copy(estGraphon.nKnots) if hasattr(estGraphon, 'nKnots') else None
            if lmd_phase1:  # (index <= lambda_skip1)
                if ((index % 2) != 0):
                    lambda_ = (1 - (1 - (((index - 1) / (lambda_skip1 - 1)) if (lambda_skip1 > 1) else 1.))**2) * (lambda_start - 1) + 1
                    sigma_prop_ = sigma_prop * .5
                    gamma = .75
                else:
                    lambda_ = (1 - (((index - 2) / (lambda_skip1 - 2)) if (lambda_skip1 > 2) else 1.))**2 * (np.max([50000, lambda_start*5]) - lambda_start) + lambda_start
                    sigma_prop_ = sigma_prop * 2
                    gamma = .25
            else:
                sigma_prop_ = sigma_prop
                gamma = .25 + ((index-1)/(n_iter-1)) *.5 in range(1,n_iter+1)
            if lmd_phase5b:  # ***
                lambda_ = np.min([25, lambda_start])
                sigma_prop_ = sigma_prop * .5
                gamma = .75
            optForAIC = np.any([lmd_phase2, lmd_phase4a, lmd_phase6])
            if adjustSubs or adjustQuantiles:
                lambda_adjustSubs = lambda_adjustQuant = index / n_iter
            Us_mult = None
            estGraphonData=Estimator(sortG=sortG)
            estGraphon=estGraphonData.GraphonEstBySpline(k=k, nSubs=nSubs, nKnots=nKnots, splitPos=splitPos, est_splitPos=est_initSplitPos, useOneBasis=useOneBasis, tau=tau, tau_sep=tau_sep,
                                                         optForAIC=optForAIC, lambdaMin=None, lambdaMax=None, calcAIC=True, lambda_=(None if optForAIC else lambda_),
                                                         adjustSubs=adjustSubs, lambda_adjustSubs=lambda_adjustSubs, adjustQuantiles=adjustQuantiles, lambda_adjustQuant=lambda_adjustQuant,
                                                         Us_mult=Us_mult, canonical=canonical, updateGraph=True, printWarn=False)
            if makePlots:
                estGraphon.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dir_ + 'graphon_est_' + labNb.__str__() + '.png')
            ## force the algorithm to keep the given number of communities
            if np.any(estGraphon.freqUsSub == 0):
                emptyGr = (estGraphon.freqUsSub == 0)
                warnings.warn('[result nb ' + labNb.__str__() + '] empty communit' + (('y ' + (np.where(emptyGr)[0][0]+1).__str__() + ' has') if (emptyGr.sum() == 1) else ('ies ' + ', '.join((np.where(emptyGr)[0]+1).astype(str)) + ' have')) + ' been removed')
                print('UserWarning: [result nb ' + labNb.__str__() + '] empty communit' + (('y ' + (np.where(emptyGr)[0][0]+1).__str__() + ' has') if (emptyGr.sum() == 1) else ('ies ' + ', '.join((np.where(emptyGr)[0]+1).astype(str)) + ' have')) + ' been removed')
                us_ = np.concatenate([np.linspace(estGraphon.splitPos[i],estGraphon.splitPos[i+1], int(np.round(sortG.N*((estGraphon.splitPos[i+1]-estGraphon.splitPos[i])/(1-sizeOff)))) +2)[1:-1] for sizeOff in [np.sum([(estGraphon.splitPos[i+1]-estGraphon.splitPos[i]) for i in range(estGraphon.nSubs) if emptyGr[i]])] for i in range(estGraphon.nSubs) if (not emptyGr[i])])
                vs_list = [(tau_sep_i[k:-k] + np.concatenate(([1e-5],np.repeat(0,len(tau_sep_i[k:-k])-2),[-1e-5]))) for tau_sep_i in np.array(estGraphon.tau_sep)[np.logical_not(emptyGr)]]
                lenVs = np.array([len(vs_list_i) for vs_list_i in vs_list])
                vs_ = np.concatenate(vs_list)
                probMat = estGraphon.fct(us_, vs_)
                newSplits = [vs_list[group_i][val_i - (np.append([0], np.cumsum(lenVs - 1))[group_i]) + np.array([0, 1])].mean() for val_i in (np.delete(np.diff(probMat), np.cumsum(lenVs)[:-1] - 1, axis=1) ** 2).sum(axis=0).argsort()[-np.sum(emptyGr):] for group_i in [np.sum(np.cumsum(lenVs - 1) < (val_i + 1))]]
                newSplitPos = np.sort(np.concatenate(((estGraphon.splitPos[np.concatenate((np.logical_not(emptyGr), [True]))] + estGraphon.splitPos[np.concatenate(([True], np.logical_not(emptyGr)))])/2, newSplits)))
                newSplitPos[0], newSplitPos[-1] = 0., 1.
                warnings.warn((('a new split has') if (emptyGr.sum() == 1) else ('new splits have')) + ' been inserted at ' + ', '.join(map(lambda x: str(round(x,4)), newSplits)))
                print('UserWarning: ' + (('a new split has') if (emptyGr.sum() == 1) else ('new splits have')) + ' been inserted at ' + ', '.join(map(lambda x: str(round(x,4)), newSplits)))
                if not np.all([np.allclose(np.diff(tau_sep_i[k:-k]), np.repeat(tau_sep_i[k+1]-tau_sep_i[k], len(tau_sep_i[k:-k])-1)) for tau_sep_i in estGraphon.tau_sep if (len(tau_sep_i[k:-k]) > 0)]):
                    warnings.warn('manually adapted knot positions have been discarded')
                    print('UserWarning: manually adapted knot positions have been discarded')
                print(estGraphon.splitPos, newSplitPos)  # !!!
                estGraphon=estGraphonData.GraphonEstBySpline(k=k, nSubs=nSubs, nKnots=nKnots, splitPos=newSplitPos, est_splitPos=False, useOneBasis=useOneBasis, tau=None, tau_sep=None,
                                                             optForAIC=optForAIC, lambdaMin=None, lambdaMax=None, calcAIC=True, lambda_=(None if optForAIC else lambda_),
                                                             adjustSubs=adjustSubs, lambda_adjustSubs=lambda_adjustSubs, adjustQuantiles=adjustQuantiles, lambda_adjustQuant=lambda_adjustQuant,
                                                             Us_mult=Us_mult, canonical=canonical, updateGraph=True, printWarn=False)
                if makePlots:
                    estGraphon.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dir_ + 'graphon_est_' + labNb.__str__() + '_newSplit.png')
            ## remove marginalized segments [should not occur when a certain number of communities is forced, except in the last iteration]
            if np.any([np.allclose(tau_sep_i, tau_sep_i[k]) for tau_sep_i in estGraphon.tau_sep]):
                emptySub = np.array([np.allclose(tau_sep_i, tau_sep_i[k]) for tau_sep_i in estGraphon.tau_sep])
                warnings.warn('[result nb ' + labNb.__str__() + '] empty segment' + ((' ' + (np.where(emptySub)[0][0]+1).__str__() + ' has') if (emptySub.sum() == 1) else ('s ' + ', '.join((np.where(emptySub)[0]+1).astype(str)) + ' have')) + ' been removed')
                print('UserWarning: [result nb ' + labNb.__str__() + '] empty segment' + ((' ' + (np.where(emptySub)[0][0]+1).__str__() + ' has') if (emptySub.sum() == 1) else ('s ' + ', '.join((np.where(emptySub)[0]+1).astype(str)) + ' have')) + ' been removed')
                tau_sep_new = list(np.array(estGraphon.tau_sep)[np.logical_not(emptySub)])
                tau_new = np.concatenate([tau_sep_new[i][(0 if (i == 0) else (k + 1)):] for i in range(len(tau_sep_new))])
                nKnots_new = estGraphon.nKnots[np.logical_not(emptySub)]
                estGraphon = estGraphonData.GraphonEstBySpline(k=k, nSubs=nSubs -emptySub.sum(), nKnots=nKnots_new, splitPos=None, est_splitPos=False, useOneBasis=useOneBasis, tau=tau_new, tau_sep=tau_sep_new,
                                                               optForAIC=optForAIC, lambdaMin=None, lambdaMax=None, calcAIC=True, lambda_=(None if optForAIC else lambda_),
                                                               adjustSubs=adjustSubs, lambda_adjustSubs=lambda_adjustSubs, adjustQuantiles=adjustQuantiles, lambda_adjustQuant=lambda_adjustQuant,
                                                               Us_mult=Us_mult, canonical=canonical, updateGraph=True, printWarn=False)
                if makePlots:
                    estGraphon.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dir_ + 'graphon_est_' + labNb.__str__() + '_remSub.png')
                if not (lambdas_ is None):
                    lambdas_ = np.array([lambda_i[np.logical_not(emptySub)][:,np.logical_not(emptySub)] for lambda_i in lambdas_])
                if not (lambdaList is None):
                    lambdaList = np.array([lambda_i[np.logical_not(emptySub)][:, np.logical_not(emptySub)] for lambda_i in lambdaList])
            if optForAIC:
                lambda_ = estGraphon.lambda_
                if lmd_phase4a:  # (index in range(lambda_skip2+1,lambda_lim2+1))
                    if lambdas_ is None:
                        lambdas_ = np.expand_dims(lambda_, axis=0)
                    else:
                        lambdas_ = np.concatenate((lambdas_, np.expand_dims(lambda_, axis=0)), axis=0)
                if lmd_phase4b:  # (index == lambda_lim2)
                    lambda_ = lambdas_.mean(axis=0)
            if returnLambList:
                if (lambdaList is None):
                    lambdaList = np.array([estGraphon.lambda_])
                else:
                    lambdaList = np.append(lambdaList, np.array([estGraphon.lambda_]), axis=0)
        if returnGraphonList:
            estGraphonList.append(estGraphon)
        trajMat = np.append(trajMat, estGraphon.fct(np.arange(1, n_eval+1) / (n_eval+1), np.arange(1, n_eval+1) / (n_eval+1)).reshape(1, n_eval, n_eval), axis=0)
        AIC_vec = np.append(AIC_vec, [estGraphon.optValue])
        if optForAIC:
            if makePlots:
                showOptForAIC(AIC_opt_lamb_list=estGraphon.AIC_opt_lamb_list, AIC_opt_vals_list=estGraphon.AIC_opt_vals_list, m=index, make_show=make_show, savefig=savefig, file_=dir_ + 'AIC_OptProfile')
        ### Sample U's
        if (index < n_iter) or (endWithSamp and (rep_forPost > 0)):
            print('Sampling')
            rep = rep_start if (index < it_rep_grow) else (int(np.round((rep_start**(n_iter-it_rep_grow) / rep_end)**(1/(n_iter-it_rep_grow-1)) * np.exp(np.log(rep_start / (rep_start**(n_iter-it_rep_grow) / rep_end)**(1/(n_iter-it_rep_grow-1))) * (index-it_rep_grow+1)))) if (index < n_iter) else (rep_forPost))
            sample=Sample(sortG=sortG,graphon=estGraphon,use_origFct=use_origFct)
            sample.gibbs(steps=n_steps,proposal=proposal,rep=rep,sigma_prop=sigma_prop_,gamma=gamma,splitPos=None,returnAllGibbs=returnAllGibbs,averageType=averageType,updateGraph=False,use_stdVals =None,printWarn=False)
            if makePlots:
                sample.showMove(useColor=True if simulate else False, EMstep_sign=labNb, make_show=make_show, savefig=savefig, file_=dir_ + 'Us_move_' + labNb.__str__() + '.png')
                if simulate:
                    ## differences to real U's is no illustrated based on sortG.showDiff()
                    sample.showMove(Us_type='real', useAllGibbs=True, EMstep_sign=labNb, make_show=make_show, savefig=savefig, file_=dir_ + 'UsMCMC_diffReal_' + labNb.__str__() + '.png')
            if returnSampList:
                sampleList.append(sample)
            ## update graph is now included in Sample()
        eval("print('iteration completed:', index, ', penalizing parameter lambda:', lambda_, ',  number of Gibbs sampling stages:', rep" + \
             (", ',\\nparameter for step-wise adjustment of subareas:', np.round(lambda_adjustSubs, 3)" if (adjustSubs and (startWithEst or (index > 1))) else "") + \
             (", ',\\nparameter for step-wise quantile adjustment:', np.round(lambda_adjustQuant, 3)" if (adjustQuantiles and (startWithEst or (index > 1))) else "") + ")")
    result = type('', (), {})()
    result.sortG = sortG
    result.estGraphon = estGraphon
    result.sample = sample
    result.lambda_ = lambda_
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

