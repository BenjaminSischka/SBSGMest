'''

Application of the EM-based graphon estimation.
@author: Benjamin Sischka

'''
import sys,os
## when running the file as script
# dir1_ = os.path.dirname(__file__)
## when running the code in an interactive session
dir1_ = os.path.dirname(os.path.realpath(''))
## when running an interactive session on the server
# dir1_ = os.path.realpath('') + '/GraphonPy'
sys.path.append(dir1_)  # specify path to Module SBSGMpy
from SBSGMpy import *

import pickle

## specify graphic options
from matplotlib import use as use_backend
use_backend("Agg")  # "GTK3Agg"
make_show=False
plt.rcParams["axes.grid"] = False
plt.rcParams.update({'font.size': 16})
## specify figure sizes
figsize1 = (9, 4)
figsize2 = (9, 5)

## specify the considered object to analyze
simulate = False  # logical whether to used simulated or real data (-> knowledge about real U's)
byExID_nb = 2  # 2 ***  # specify the byExID function to use (only used if simulate == True)
idX = 205  # 202 ***  # specify graphon to consider (only used if simulate == True)
data_ = 'alliances'  # specify data to consider (only used if simulate == False)
start_ = 1  # start value for global repetition of the algorithm
stop_ = (2) + (start_ -1)  # stop value for global repetition of the algorithm, (.) = number of global repetitions

## graphic specifications
directory_ = os.path.join(dir1_, 'Graphics') + '/' + (('graphon_' + byExID_nb.__str__() + '_' + idX.__str__()) if simulate else data_) + '/'  # define directory to save graphics   # dir1_, '..', 'Graphics'
if not os.path.exists(directory_):  # create directory for graphics, if it does not already exist
    if not os.path.exists(os.path.join(dir1_, 'Graphics')):  # , '..'
        os.mkdir(os.path.join(dir1_, 'Graphics'))  # , '..'
    os.mkdir(directory_)
savefig = True  # logical whether to save figures
add_nTry = True  # logical whether the identification number should be added to graphic file names (only used if savefig == True)
plotAll = True  # logical whether to plot auxiliary graphics too
log_scale = False if simulate else True  # logical whether to use log_scale for graphon plot

## about initialization
estMethod = 'random'  # method for initial estimation of Us_est [options: 'degree', 'mds', 'random', None]
est_initSplitPos = estMethod != 'random'  # logical whether split positions should initially be estimated
useIndividGraphs = False  # logical whether to use the same or individual graphs for the global repetitions of the estimation routine (only used if simulate == True)
useIndividRandInit = True  # logical whether to use different random intializations (only used if estMethod == 'random' and useIndividGraphs == False)

initGraphonEst = False  # logical whether to make an initial estimate of the graphon
initCanonical = False  # logical whether to start with canoncial estimation (only used if initGraphonEst == True)
initPostDistr = False  # logical whether to calculate the initial posterior distribution (graphon specification necessary)
trueInit = False  # logical whether to start with true model (true ordering + true graphon, dominates 'estMethod', only used if simulate == True)
if initPostDistr and (not (initGraphonEst or trueInit)):
    raise TypeError('no initial graphon estimation available')

N = 80  # dimension of network (only used if simulate == True)  # *** 500
Us_real = None  # specify initial real U's (only used if simulate == True)
randomSample = True  # logical whether the real U's are a random or an equidistant sample (only used if simulate == True & Us_real == None)

## parameters for B-spline regression
k = 1  # order of B-splines (only 0 and 1 are implemented)
nSubs_low = 6  # lower bound of number of communities
nSubs_up = 7  # upper bound of number of communities
nKnots_all = None  # overall number of inner knots for the B-spline bases, if 'None' number of knots is based on N and nSubs
useOneBasis = False  # logical whether graphon should be estimated by using a single spline basis - with tau_k = tau_k+1
critType = 'ICL'  # criterion to be optimized with regard to smoothness [options: 'AIC', 'AICc', 'AICL1', 'AICL2', 'AICL2b', 'ICL']
canonical = False  # logical whether a canonical representation should be fitted
adjustSubs=True  # logical whether the subareas should be adjusted
adjustQuantiles=False  # logical whether it should be adjusted for the quantiles of the Us
if adjustSubs and adjustQuantiles:
    raise TypeError('only one type of adjustment should be used')

## parameters for the sampling
n_steps = 5  # steps of Gibbs iterations  # *** 200
proposal='mixture'  # type of proposal to use for the sampling (options: 'logit_norm', 'exclus_unif', 'uniform', 'mixture')
sigma_prop = 2  # variance of sampling step (-> proposal distribution, only used if proposal == 'logit_norm')
use_origFct=False  # logical whether to use the graphon function itself or a discrete approx
averageType='mean'  # specify the kind of posterior average
use_stdVals=True  # logical whether to use standardized Us (-> equidistant)

## parameters for calculating the illustrated posterior distribution
rep_forPost = 1  # number of repetitions/sequences of the Gibbs sampling for calculating the posterior distribution  # *** 50
useAllGibbs = True  # logical whether to use all Gibbs repetitions for calculating the posterior or simply the mean
distrN_fine = 1000  # fineness of the posterior distribution -> number of evaluation points
ks_rel = np.array([0,0.25,0.75,1])  # relative k's for which the posterior is calculated
ks_abs = None  # absolute k's for which the posterior is calculated, if None they will be calculated below depending on ks_rel (dominates 'ks_rel')

## parameters for the EM algorithm
n_iter = 35  # number of EM iterations
rep_start = 1  # start value for number of repetitions in the Gibbs sampling step
rep_end = 25  # end value for number of repetitions in the Gibbs sampling step  # *** 100
it_rep_grow = 20  # iteration from which rep starts to grow

# N/5 ,  when Us are not standardized: set lambda >0 to make the model identifiable [e.g. if #{u_i in (t_k, t_{k+2}]} = 0]
lambda_start = 35  # start value for the penalization parameter
lambda_skip1 = 23  # iterations to skip before optimizing lambda
lambda_lim1 = (2) + lambda_skip1  # (.) = optimized lambdas not to use for the mean penalization parameter
lambda_skip2 = (2) + lambda_lim1  # (.) = iterations to skip before optimizing lambda
lambda_lim2 = (2) + lambda_skip2  # (.) = optimized lambdas to use for the mean penalization parameter
lambda_last_m = 3  # last m iterations at which lambda is optimized again
if (lambda_lim2 >= (n_iter - lambda_last_m)):
    warnings.warn('lambda_lim2 should be smaller than n_iter-lambda_last_m -> specification should be reconsidered')
    print('UserWarning: lambda_lim2 should be smaller than n_iter-lambda_last_m -> specification should be reconsidered')

## parameter for observing convergence
n_eval = 3  # number of evaluation points for the trajectory for observing convergence -> equidistant positions

## marker/tag for special modifications
# !!! = start with true ordering and with true graphon
# *** = changes required

## marker/tag for temporary modifications
# ´´´ = special initialization for karate data set


print('simulate: ' + simulate.__str__() + ((' ,  byExID_nb: ' + byExID_nb.__str__() + ' ,  idX: ' + idX.__str__()) if simulate else (' ,  data_: ' + data_.__str__())))


### Define graphon (if simulation is considered -> simulate = True)

if simulate:
    exec('graphon0=byExID' + byExID_nb.__str__() + '(idX=idX, size=1001)')
    graphonMin0, graphonMax0 = np.max([np.floor(np.min(graphon0.mat) / 0.05) *0.05, 0]), np.min([np.ceil(np.max(graphon0.mat) / 0.05) *0.05, 1])

useSameGraph = np.any([simulate and (not useIndividGraphs), not simulate])
useIndividRandInit = (estMethod=='random') and useIndividRandInit

result_list = []
for nSubs_ in range(nSubs_low, nSubs_up+1):
    #nSubs_ = nSubs_low

    for glob_ind in range(start_,stop_+1):
        #glob_ind = start_

        seed_ = nSubs_ + 1000 * glob_ind
        np.random.seed(seed_)

        nTry = glob_ind.__str__() + '_'  # specify an identification for the run
        dirExt = directory_ + (nSubs_.__str__() + '_') + (nTry if add_nTry else '')


        if simulate:
            # plot true graphon
            graphon0.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dirExt + 'graphon_true.png')
            # *** make second plot of true graphon when sameGraphonScale=True -> down below



        ### Define graph

        if simulate:
            if (useIndividGraphs or (glob_ind == start_)):
                graph0=GraphByGraphon(graphon=graphon0,Us_real=Us_real,N=N,randomSample=randomSample,estMethod=estMethod)
                graph0.sort(Us_type = 'est')
        else:
            if glob_ind == start_:
                graph0 = GraphFromData(data_=data_, dir_=dir1_, estMethod=estMethod)
                graph0.sort(Us_type = 'est')
                N=graph0.N
        if (simulate and (useIndividGraphs or (glob_ind == start_))) or ((not simulate) and (glob_ind == start_)):
            print('Average degree in the network:', graph0.averDeg)  # average number of links per node
            print('Overall density:', graph0.density)  # graph density
        if useSameGraph:
            if useIndividRandInit:
                if glob_ind != start_:
                    graph0.update(Us_est=np.random.permutation(np.linspace(0, 1, N + 2)[1:-1]))
            else:
                if glob_ind == start_:
                    Us_est_unique = copy(graph0.Us_est)
                else:
                    graph0.update(Us_est=Us_est_unique)


        # plot adjacency matrix based on initial ordering
        graph0.showAdjMat(make_show=make_show, savefig=savefig, file_=dirExt + 'adjMat_0.png')

        # plot network with initial ordering
        ## consider splitPos in Estimator.GraphonEstBySpline(..., est_splitPos=True, ...)
        if (est_initSplitPos and (nSubs_ >1)):
            splitPos0 = estSplitPos(A=graph0.A, nSubs=nSubs_)
        else:
            splitPos0 = np.linspace(0,1,nSubs_+1)
        graph0.showNet(splitPos=splitPos0, make_show=make_show, savefig=savefig, file_=dirExt + 'network_0.png')


        if simulate:
            # plot network with true ordering
            graph0.showNet(Us_type='real', splitPos=graphon0.splitPos if hasattr(graphon0, 'splitPos') else None, make_show=make_show, savefig=savefig, file_=dirExt + 'network_true.png')

            # plot difference between initial estimates and real U's
            graph0.showDiff(Us_type='est', EMstep_sign='(0)', make_show=make_show, savefig=savefig, file_=dirExt + 'Us_diffReal_0.png')

            if plotAll:
                # define graph ordered by real U's
                graph0_trueSort=graph0.makeCopy()
                graph0_trueSort.sort(Us_type='real')

                # plot adjacency matrix based on true ordering
                graph0_trueSort.showAdjMat(make_show=make_show, savefig=savefig, file_=dirExt + 'adjMat_true.png')

                # plot observed vs expected degree profile
                graph0_trueSort.showObsDegree(absValues=False, norm=False, fmt = 'C1o', title = False, make_show=False, savefig=False)
                graphon0.showExpDegree(norm=False, fmt = 'C0--', title = False, make_show=False, savefig=False)
                plt.xlabel('(i) $u$   /   (ii) $u_i$')
                plt.ylabel('(i) $g(u)$   /   (ii) $degree(i) \;/\; (N-1)$')
                plt.tight_layout()
                if make_show:
                    plt.show()
                if savefig:
                    plt.savefig(dirExt + 'obsVSreal_expDegree.png')
                    plt.close(plt.gcf())

                # plot histogram of real U's
                graph0_trueSort.showUsHist(bins=graphon0.splitPos if hasattr(graphon0, 'splitPos') else np.linspace(0,1,nSubs_ +1), make_show=make_show, savefig=savefig, file_=dirExt + 'Us_hist_true.png')



        ### Initial fit of graphon + initial posterior distribution of U_k

        nKnots = np.max([int(np.max([np.log(N / np.sqrt(nSubs_)) * 8 -25, 5]) + np.sqrt(nSubs_ - 1) * 5), 2*nSubs_]) if (nKnots_all is None) else nKnots_all
        # nKnots = np.max([int(np.max([(N / np.sqrt(nSubs_))**(1/3) * 4, 5]) + np.sqrt(nSubs_ - 1) * 5), 2*nSubs_]) if (nKnots_all is None) else nKnots_all

        if trueInit:
            if simulate:
                graph0.update(Us_est = graph0.Us_('real'))
            else:
                warnings.warn('real data example is considered, ground truth is unknown')
                print('UserWarning: real data example is considered, ground truth is unknown')

        if initGraphonEst:
            estGraphonData0 = Estimator(sortG=graph0)
            estGraphon0 = estGraphonData0.GraphonEstBySpline(k=1, nSubs=nSubs_, nKnots=nKnots, splitPos=splitPos0, est_splitPos=False, useOneBasis=useOneBasis, tau=None, tau_sep=None, \
                                                             optForAIC=True, lambdaMin=None, lambdaMax=None, calcAIC=False, lambda_=None, critType=critType, \
                                                             adjustSubs=adjustSubs, lambda_adjustSubs=2/3, adjustQuantiles=adjustQuantiles, lambda_adjustQuant=2/3, \
                                                             Us_mult=None, canonical=initCanonical, updateGraph=True, printWarn=False)
            trajMat = estGraphon0.fct(np.arange(1, n_eval + 1) / (n_eval + 1), np.arange(1, n_eval + 1) / (n_eval + 1)).reshape(1, n_eval, n_eval)
            lambdaList = np.array([estGraphon0.lambda_])
            estGraphonList = [estGraphon0]
            # plot initial graphon estimate
            estGraphon0.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dirExt + 'graphon_est_1.png')
            if simulate:
                # plot true vs initial estimated graphon
                graphonMin1, graphonMax1 = np.max([np.floor(np.min([graphonMin0, np.min(estGraphon0.mat)]) / 0.05) * 0.05, 0]), np.min([np.ceil(np.max([graphonMax0, np.max(estGraphon0.mat)]) / 0.05) * 0.05, 1])
                plt.figure(1, figsize=figsize1)
                plt.subplot(121)
                graphon0.showColored(log_scale=log_scale, fig_ax=(plt.gcf(), plt.gca()), vmin=graphonMin1, vmax=graphonMax1, make_show=False, savefig=False)
                plt.subplot(122)
                estGraphon0.showColored(log_scale=log_scale, fig_ax=(plt.gcf(), plt.gca()), vmin=graphonMin1, vmax=graphonMax1, make_show=False, savefig=False)
                plt.tight_layout(w_pad=2, h_pad = 1.5)
                if make_show:
                    plt.show()
                if savefig:
                    plt.savefig(dirExt + 'graphon_compare_1.png')
                    plt.close(plt.gcf())
        else:
            estGraphon0 = None
            trajMat = None
            lambdaList = None
            estGraphonList = []


        if trueInit:
            if simulate:
                if initGraphonEst:
                    graph0.update(Us_est = graph0.Us_('real'))
                estGraphon0 = graphon0
            else:
                warnings.warn('real data example is considered and ground truth is unknown, so no true initialization can be used')
                print('UserWarning: real data example is considered and ground truth is unknown, so no true initialization can be used')

        seed2_ = seed_ * 50 + 3
        np.random.seed(seed2_)

        if ks_abs is None:
            ks_abs = np.unique(np.minimum(np.maximum(np.round(ks_rel * N).astype('int') - 1, 0), N - 1))  # absolute k's for which the posterior is calculated
        if initPostDistr:
            if rep_forPost == 0:
                warnings.warn('number of repetitions for calculating the posterior distribution is 0, no calculation is carried out')
                print('UserWarning: number of repetitions for calculating the posterior distribution is 0, no calculation is carried out')
                sampleList = []
            else:
                # apply Gibbs sampling to the initial U's given the graphon estimate based on the initial ordering
                print('Sampling')
                sample0=Sample(sortG=graph0,graphon=estGraphon0,use_origFct=use_origFct)
                sample0.gibbs(steps=n_steps,rep=rep_forPost,proposal=proposal,sigma_prop=sigma_prop, gamma=.5, splitPos=None, averageType=averageType, returnAllGibbs=False, updateGraph=False, use_stdVals=None, printWarn=True)
                # calculate and plot the posterior distribution of U_k based on the initial graphon estimate, with k corresponding to the initial ordering
                sample0.showPostDistr(ks_lab=None, ks_abs=ks_abs, Us_type_label='est', Us_type_add='est', distrN_fine=distrN_fine, useAllGibbs=True, EMstep_sign='(1)', figsize=figsize2, mn_=None, useTightLayout=True, w_pad=2, h_pad = 1.5, make_show=make_show, savefig=savefig, file_=dirExt + 'postDistr_0.png')
                sample0.updateGraph(use_stdVals=use_stdVals)
                # plot adjacency matrix based on initial ordering
                graph0.showAdjMat(make_show=make_show, savefig=savefig, file_=dirExt + 'adjMat_1.png')
                # plot network with initial ordering
                graph0.showNet(splitPos=estGraphon0.splitPos if hasattr(estGraphon0, 'splitPos') else None, make_show=make_show, savefig=savefig, file_=dirExt + 'network_1.png')
                if simulate:
                    # plot differences to real U's
                    graph0.showDiff(Us_type='est', EMstep_sign='(1)', make_show=make_show, savefig=savefig, file_=dirExt + 'Us_diffReal_1.png')
                sampleList = [sample0]
        else:
            sampleList = []



        ### Sample U's and fit graphon again and again

        EM_obj = iterateEM(sortG=graph0,
                           k=k, nSubs=nSubs_, nKnots=nKnots, useOneBasis=useOneBasis, critType=critType, canonical=canonical, est_initSplitPos=est_initSplitPos, adjustSubs=adjustSubs, adjustQuantiles=adjustQuantiles,
                           n_steps=n_steps, proposal=proposal, sigma_prop=sigma_prop, use_origFct=use_origFct, averageType=averageType, use_stdVals=use_stdVals,
                           n_iter=n_iter, rep_start=rep_start, rep_end=rep_end, it_rep_grow=it_rep_grow, rep_forPost=rep_forPost,
                           lambda_start=lambda_start, lambda_skip1=lambda_skip1, lambda_lim1=lambda_lim1, lambda_skip2=lambda_skip2, lambda_lim2=lambda_lim2, lambda_last_m=lambda_last_m,
                           n_eval=n_eval, trajMat=trajMat,
                           startWithEst=(not initGraphonEst) or (initGraphonEst and initPostDistr), estGraphon=estGraphon0,
                           endWithSamp=True, raiseLabNb=initGraphonEst and initPostDistr,
                           returnLambList=True, returnGraphonList=False, returnSampList=False, returnAllGibbs=False,
                           lambdaList=lambdaList, estGraphonList=estGraphonList, sampleList=sampleList,
                           makePlots=plotAll, make_show=False, savefig=savefig, simulate=simulate, log_scale=log_scale, dir_=dirExt)



        # plot adjacency matrix based on final ordering
        EM_obj.sortG.showAdjMat(make_show=make_show, savefig=savefig, file_=dirExt + 'adjMat_EM.png')

        # plot network with final ordering
        EM_obj.sortG.showNet(splitPos=EM_obj.estGraphon.splitPos, make_show=make_show, savefig=savefig, file_=dirExt + 'network_EM.png')
        EM_obj.sortG.showNet(splitPos=EM_obj.estGraphon.splitPos, byGroup=True, make_show=make_show, savefig=savefig, file_=dirExt + 'network_EM2.png')

        # plot trajectory of graphon estimation sequence for specific positions u and v
        showTraject(trajMat=EM_obj.trajMat, make_show=make_show, savefig=savefig, file_=dirExt + 'trajectory_graphonSeq.png')

        # plot sequence of penalization parameter lambda_
        showLambda(lambdaList=EM_obj.lambdaList, make_show=make_show, savefig=savefig, file_=dirExt + 'trajectory_lambdaSeq.png')

        # plot sequence of information criterion
        showAIC(AIC_vec=EM_obj.AIC_vec, make_show=make_show, savefig=savefig, file_=dirExt + 'trajectory_AIC.png')

        # plot final graphon estimate
        EM_obj.estGraphon.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dirExt + 'graphon_est_EM.png')
        EM_obj.estGraphon.showSlices(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dirExt + 'graphon_slices_EM.png')

        if simulate:
            # plot true vs estimated graphon
            graphonMin1, graphonMax1 = np.max([np.floor(np.min([graphonMin0, np.min(EM_obj.estGraphon.mat)]) / 0.05) * 0.05, 0]), np.min([np.ceil(np.max([graphonMax0, np.max(EM_obj.estGraphon.mat)]) / 0.05) * 0.05, 1])
            graphon0.showColored(log_scale=log_scale, vmin=graphonMin1, vmax=graphonMax1, make_show=make_show, savefig=savefig, file_=dirExt + 'graphon_true2.png')
            EM_obj.estGraphon.showColored(log_scale=log_scale, vmin=graphonMin1, vmax=graphonMax1, make_show=make_show, savefig=savefig, file_=dirExt + 'graphon_est_EM2.png')
            plt.figure(1, figsize=figsize1)
            plt.subplot(121)
            graphon0.showColored(log_scale=log_scale, fig_ax=(plt.gcf(), plt.gca()), vmin=graphonMin1, vmax=graphonMax1, make_show=make_show, savefig=False)
            plt.subplot(122)
            EM_obj.estGraphon.showColored(log_scale=log_scale, fig_ax=(plt.gcf(), plt.gca()), vmin=graphonMin1, vmax=graphonMax1, make_show=make_show, savefig=False)
            plt.tight_layout(w_pad=2, h_pad=1.5)
            if make_show:
                plt.show()
            if savefig:
                plt.savefig(dirExt + 'graphon_compare_EM.png')
                plt.close(plt.gcf())

            # plot difference between EM estimates and real U's
            EM_obj.sortG.showDiff(Us_type='est', EMstep_sign='EM', make_show=make_show, savefig=savefig, file_=dirExt + 'Us_diffReal_EM.png')

            if canonical:
                # plot true vs estimated marginalization
                graphon0.showExpDegree(size=1000,norm=False,fmt='-',title=False,make_show=make_show,savefig=False)
                EM_obj.estGraphon.showExpDegree(size=1000,norm=False,fmt='-',title=True,make_show=make_show,savefig=savefig,file_=dirExt + 'margin_compare_EM.png')

                # plot true vs estimated marginalization in direct comparison
                g1_ = fctToMat(fct=graphon0.fct, size=(10 * 100, 100)).mean(axis=0)
                g2_ = fctToMat(fct=EM_obj.estGraphon.fct, size=(10 * 100, 100)).mean(axis=0)
                minLim, maxLim = np.min(np.append(g1_, g2_)), np.max(np.append(g1_, g2_))
                lim_ = np.array([minLim, maxLim]) + (maxLim - minLim) * 1 / 20 * np.array([-1, 1])
                plt.xlim(lim_)
                plt.ylim(lim_)
                plt.plot(g1_, g2_, 'C1')
                plt.plot([minLim, maxLim], [minLim, maxLim], 'C0--')
                plt.xlabel('$g(u)$')
                plt.ylabel('$\hat g^{\;EM}(u)$')
                if make_show:
                    plt.show()
                if savefig:
                    plt.savefig(dirExt + 'margin_compare2_EM.png')
                    plt.close(plt.gcf())

        if plotAll:
            # plot observed vs expected degree profile based on EM ordering
            EM_obj.sortG.showObsDegree(absValues=False, norm=False, fmt = 'C1o', title=False, make_show=make_show, savefig=False)
            EM_obj.estGraphon.showExpDegree(norm=False, fmt = 'C0--', title=None, make_show=make_show, savefig=False)
            plt.xlabel('(i) $u$   /   (ii) $\hat{u}_i^{\;EM}$')
            plt.ylabel('(i) $\hat{g}^{\;EM}(u)$   /   (ii) $degree(i) \;/\; (N-1)$')
            plt.tight_layout()
            if make_show:
                plt.show()
            if savefig:
                plt.savefig(dirExt + 'obsVsEM_expDegree.png')
                plt.close(plt.gcf())

        if rep_forPost != 0:
            # calculate and plot the posterior distribution of U_k based on the final graphon estimate, with k corresponding to the final ordering
            EM_obj.sample.showPostDistr(ks_lab=None, ks_abs=ks_abs, Us_type_label='est', Us_type_add='est', distrN_fine=distrN_fine, useAllGibbs=True, EMstep_sign='EM', figsize=figsize2, mn_=None, useTightLayout=True, w_pad=2, h_pad = 1.5, make_show=make_show, savefig=savefig, file_=dirExt + 'postDistr_EM.png')


        result_ = np.array([nSubs_, nTry, EM_obj.estGraphon.AIC, EM_obj.estGraphon.AICc, EM_obj.estGraphon.AICL1, EM_obj.estGraphon.AICL2, EM_obj.estGraphon.AICL2b, EM_obj.estGraphon.ICL, EM_obj.estGraphon.critType, EM_obj.estGraphon.critValue, EM_obj.sortG.logLik(graphon = EM_obj.estGraphon), EM_obj.estGraphon.logLik, EM_obj.estGraphon.df_lambda, (np.round(EM_obj.lambda_, 4).__str__() if np.isscalar(EM_obj.lambda_) else ' '.join([np.round(EM_obj.lambda_, 1)[i].__str__() for i in range(EM_obj.lambda_.shape[0])]).replace('\n', ''))])
        print(result_)
        result_list.append(result_)


        graph_simple = {'A': EM_obj.sortG.A, 'labels': EM_obj.sortG.labels_(),
                        'Us_real': EM_obj.sortG.Us_('real'), 'Us_est': EM_obj.sortG.Us_('est')}
        graphon_simple = {'mat': EM_obj.estGraphon.mat, 'order': EM_obj.estGraphon.order, 'nSubs': EM_obj.estGraphon.nSubs,
                          'nKnots': EM_obj.estGraphon.nKnots, 'splitPos': EM_obj.estGraphon.splitPos,
                          'tau': EM_obj.estGraphon.tau, 'tau_sep': EM_obj.estGraphon.tau_sep, 'tau_sep_ext': EM_obj.estGraphon.tau_sep_ext,
                          'P_mat': EM_obj.estGraphon.P_mat, 'P_mat_sep_ext': EM_obj.estGraphon.P_mat_sep_ext,
                          'theta': EM_obj.estGraphon.theta, 'theta_sep_ext': EM_obj.estGraphon.theta_sep_ext}
        if rep_forPost != 0:
            sample_simple = {'U_MCMC': EM_obj.sample.U_MCMC, 'U_MCMC_std': EM_obj.sample.U_MCMC_std,
                             'U_MCMC_all': EM_obj.sample.U_MCMC_all, 'acceptRate': EM_obj.sample.acceptRate,
                             'Us_new': EM_obj.sample.Us_new, 'Us_new_std': EM_obj.sample.Us_new_std}
        else:
            sample_simple = None


        with open(dirExt + 'final_result.pkl', 'wb') as output:
            pickle.dump(result_, output, protocol=3)  # pickle.HIGHEST_PROTOCOL
            pickle.dump(graph_simple, output, protocol=3)
            pickle.dump(graphon_simple, output, protocol=3)
            pickle.dump(sample_simple, output, protocol=3)
            pickle.dump(EM_obj.lambdaList, output, protocol=3)
            pickle.dump(EM_obj.AIC_vec, output, protocol=3)


        # add parameter settings to a csv file
        fname = directory_ + '_register.csv'
        if not os.path.isfile(fname):
            with open(fname, 'a') as fd:
                fd.write('nSubs; nTry; AIC; AICc; AICL1; AICL2; AICL2b; ICL; critType; critValue; logLik1; logLik2; df_lambda; seed; seed2; estMethod; initGraphonEst; initCanonical; initPostDistr; trueInit; N; k; nKnots_all; nKnots; splits; canonical; adjustSubs; adjustQuantiles; n_steps; sigma_prop; averageType; use_stdVals; rep_forPost; n_iter; rep_start; rep_end; it_rep_grow; lambda_start; lambda_skip1; lambda_lim1; lambda_skip2; lambda_lim2; lambda_last_m; lambda_; \n')
        with open(fname, 'a') as fd:
            fd.write(';'.join(result_[:13].astype(str)) + ';' + seed_.__str__() + ';' + seed2_.__str__() + '; ' + estMethod.__str__() + '; ' + initGraphonEst.__str__() + '; ' + initCanonical.__str__() + '; ' + initPostDistr.__str__() + '; ' + trueInit.__str__() + '; ' + N.__str__() + '; ' + k.__str__() + '; ' + EM_obj.estGraphon.nKnots.sum().__str__() + '; ' + EM_obj.estGraphon.nKnots.__str__() + ';' + np.array2string(EM_obj.estGraphon.splitPos, precision = 3, max_line_width = 100000, separator = ',') + '; ' + canonical.__str__() + '; ' + adjustSubs.__str__() + '; ' + adjustQuantiles.__str__() + '; ' + n_steps.__str__() + '; ' + sigma_prop.__str__() + '; ' + averageType + '; ' + use_stdVals.__str__() + '; ' +
                     rep_forPost.__str__() + '; ' + n_iter.__str__() + '; ' + rep_start.__str__() + '; ' + rep_end.__str__() + '; ' + it_rep_grow.__str__() + '; ' + lambda_start.__str__() + '; ' + lambda_skip1.__str__() + '; ' + lambda_lim1.__str__() + '; ' + lambda_skip2.__str__() + '; ' + lambda_lim2.__str__() + '; ' + lambda_last_m.__str__() + '; ' + (np.round(EM_obj.lambda_, 4).__str__() if np.isscalar(EM_obj.lambda_) else ' '.join([np.round(EM_obj.lambda_, 1)[i].__str__() for i in range(EM_obj.lambda_.shape[0])]).replace('\n', '')) + '; \n')


        print('\n\n\nGlobal repetition complete:    ' + glob_ind.__str__() + '\n\n\n\n\n\n\n')
    print('\n\n\nModel with ' + nSubs_.__str__() + ' communities completed\n\n\n\n\n\n\n')


[print(entry_i) for entry_i in result_list]

# return best final result
#entry_i[2] = AIC, entry_i[3] = AICc, entry_i[4] = AICL1, entry_i[5] = AICL2, entry_i[6] = AICL2b, entry_i[7] = ICL, entry_i[9] = critValue, entry_i[11] = entry_i[12] = logLik
bestRes_id = np.argmin([float(entry_i[9]) for entry_i in result_list])
print(', '.join(result_list[bestRes_id][0:2]))
fname = directory_ + '_register2.csv'
if not os.path.isfile(fname):
    with open(fname, 'a') as fd:
        fd.write('nSubs; nTry; AIC; AICc; AICL1; AICL2; AICL2b; ICL; critType; critValue; logLik1; logLik2; df_lambda; lambda_; \n')
with open(fname, 'a') as fd:
    fd.write("; ".join(list(map(str, result_list[bestRes_id]))) + '; \n')

