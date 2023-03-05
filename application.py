'''

Application of the EM-based graphon estimation.
@author: Benjamin Sischka

'''
import sys,os
import pickle
import time



########################################################################################################################
############################################## individual settings #####################################################
########################################################################################################################

dir1_ = os.path.dirname(os.path.realpath(''))  # define path to package
directory_ = os.path.join(dir1_, 'Graphics/ex1') + '/'  # specify path where graphics should be saved

### Network specification
simulate = True  # boolean whether to used simulated or real data
idX = 1  # choose prespecified graphon (only used if simulate == True) [options: 1, 2, 3]
N = 150  # size of network (only used if simulate == True)
data_ = 'alliances'  # choose data (only used if simulate == False) [options: 'polblogs_new', 'alliances', 'brain']
## note: individual data set can be used by setting 'simulate = False' and specifying
##       adjacency matrix individually (see below ***)


### Settings for the algorithm
## model initialization ( -> starting positions for U's)
estMethod = 'random'  # method for initial estimation of Us_est [options: 'degree', 'mds', 'dcsbm', 'random']
## bounds of considered group numbers
nSubs_low = 2  # lower bound
nSubs_up = 4  # upper bound
## number of global repetitions
glob_reps = 1  # (only used if estMethod == 'random', otherwise it is set to 1 by default)
## plot option
show_allFigs = True  # boolean whether to plot interim results as well
## ATTENTION: plotting all figures (including illustrations of the network) is very time-consuming!!
##            to achieve comparable running times, setting 'show_allFigs = False' is required!

########################################################################################################################
############################################ individual settings end ###################################################
########################################################################################################################



### Load package
sys.path.append(dir1_)  # add package path to available directories
from SBSGMpy import *  # import package for fitting the Stochastic Block Smooth Graphon Model


### Graphic options
from matplotlib import use as use_backend
use_backend("Agg")  # specify usable backend
plt.rcParams["axes.grid"] = False  # disable grid
plt.rcParams.update({'font.size': 16})  # adjust font size
## create directory for graphics, if it does not already exist
if not os.path.exists(directory_):
    if not os.path.exists(os.path.join(dir1_, 'Graphics')):
        os.mkdir(os.path.join(dir1_, 'Graphics'))
    os.mkdir(directory_)







########################################################################################################################
##################################################### Algorithm ########################################################
################################################ (no need to modify) ###################################################
########################################################################################################################

## show choice of data
print('simulate: ' + simulate.__str__() + ((' ,  idX: ' + idX.__str__()) if simulate else \
                                               (' ,  data_: ' + data_.__str__())))





### Define graphon (only used if simulate == True)

if simulate:
    ## choose specification from prespecified graphons
    graphon0=byExID2(idX=(205 if (idX == 1) else (207 if (idX == 2) else (208 if (idX == 3) else None))), size=1001)

    ## plot true graphon
    graphon0.showColored(make_show=False, savefig=True, file_=directory_ + 'graphon_true.png')





### Define graph

np.random.seed(1)  # set seed (also required for replicability of the algorithm)

if simulate:
    ## generate simulated graph from prespecified graphon
    graph0 = GraphByGraphon(graphon=graphon0, Us_real=None, N=N, randomSample=True, estMethod=None if \
        (estMethod in ['random', 'dcsbm']) else estMethod)
else:
    ## generate graph from data
    graph0 = GraphFromData(data_=data_, dir_=os.path.join(dir1_, 'SBSGMest'), 
                           estMethod=None if (estMethod in ['random', 'dcsbm']) else estMethod)
## alternatively: specify network data through adjacency matrix 'A'  [symmetric 2-dimensional numpy array] ***:
# graph0 = ExtGraph(A, estMethod=None if (estMethod in ['random', 'dcsbm']) else estMethod)

N = graph0.N  # allocate size of graph
## save initialization for global repetitions
if (estMethod != 'random') and (estMethod != 'dcsbm'):
    Us_est_unique = copy(graph0.Us_est)

print('Average degree in the network:', graph0.averDeg)  # average number of links per node
print('Overall density:', graph0.density)  # graph density

## plot network
graph0.showNet(make_show=False, savefig=True, file_=directory_ + 'network_0.png')

if simulate:
    ## plot network with true ordering
    graph0.showNet(Us_type='real', splitPos=graphon0.splitPos if hasattr(graphon0, 'splitPos') else None,
                   make_show=False, savefig=True, file_=directory_ + 'network_true.png')

    ## calculate log-likelihood for true model
    logLik_true = graph0.logLik(graphon=graphon0, Us_type='real')
    print('Log-likelihood for true model:', logLik_true)





critValue_best = np.log(N ** 2) * N ** 2 * 2  # intialization of best result


start_glob_time = time.time()  # global starting time of algorithm

## loop for different numbers of groups
for nSubs_ in range(nSubs_low, nSubs_up+1):

    ## loop for repeated runs
    for glob_ind in range(1, (glob_reps + 1) if (estMethod == 'random') else 2):

        graph1 = graph0.makeCopy()  # define individual graph object for each repetition

        if estMethod == 'random':
            ## initialize graph with random node positions
            graph1.update(Us_est=np.random.permutation(np.linspace(0, 1, N + 2)[1:-1]))
            graph1.sort(Us_type='est')
        else:
            if estMethod != 'dcsbm':
                ## reset node positions to previous initialization
                graph1.update(Us_est=Us_est_unique)
                graph1.sort(Us_type='est')


        start_time = time.time()  # individual starting time

        ### EM algorithm
        EM_obj_i = iterateEM(sortG=graph1,
                             k=1, nSubs=nSubs_,
                             nKnots=np.max([int(np.round(np.log(N * np.sqrt(nSubs_)) / np.log(1.35))), 2*nSubs_]),
                             useOneBasis=False, critType='AICL2b', canonical=False,
                             est_initSplitPos=estMethod!='random', adjustSubs=True, adjustQuantiles=False,
                             updateStrategy='gibbs', n_steps=20, proposal='mixture', sigma_prop=2,
                             averageType='mode_mean', max_steps=10, n_try=1, use_origFct=False, use_stdVals=True,
                             n_maxIter=25, lambda_init=np.array([35., 500.]), n_init=(estMethod=='random') * 10,
                             burnIn_init=15, rep_start=5, rep_grow=3, rep_forPost=0, n_eval=3,
                             stopVal=.005, useDCSBM=estMethod=='dcsbm', splitPos=None,
                             splitPos_real=graphon0.splitPos if (simulate and hasattr(graphon0, 'splitPos')) else None,
                             lambdaLim_start=[5., 500.],
                             returnLambList=True, returnGraphonList=False, returnSampList=False, returnAllGibbs=False,
                             makePlots=show_allFigs, make_show=False, savefig=show_allFigs, simulate=simulate,
                             log_scale=False if simulate else True,
                             dir_=directory_ + nSubs_.__str__() + '_' + glob_ind.__str__() + '_')

        duration_ = time.time() - start_time  # individual duration

        ## save result as final result if better performance
        if EM_obj_i.estGraphon.critValue < critValue_best:
            EM_obj = EM_obj_i
            critValue_best = EM_obj_i.estGraphon.critValue

        ## inform about completion of repeated runs
        print('\n\n\nRepetition complete:    ' + glob_ind.__str__() + '\n\n\n')

    ## inform about completion of runs with specific group number
    print('\n\n\nModel with ' + nSubs_.__str__() + ' communit' + ('ies' if (nSubs_ > 1.5) else 'y') +' completed\n\n\n')

## show run time of complete algorithm
glob_duration = time.time() - start_glob_time
m_, s_ = divmod(round(glob_duration), 60)
h_, m_ = divmod(m_, 60)
print('global run time [hh:mm:ss]:', f'{h_:02d}:{m_:02d}:{s_:02d}',
      ('\n\n####################\nNote that running time is not comparable when all interim ' + \
       'results are plotted!!\n[see \'show_allFigs = True\']\n####################') if show_allFigs else '')
## note: setting of 'show_allFigs = True' yields not comparable results



## plot adjacency matrix based on final ordering
EM_obj.sortG.showAdjMat(make_show=False, savefig=True, file_=directory_ + 'adjMat_EM.png')

## plot network with final ordering
EM_obj.sortG.showNet(splitPos=EM_obj.estGraphon.splitPos, make_show=False, savefig=True,
                     file_=directory_ + 'network_EM.png')
EM_obj.sortG.showNet(splitPos=EM_obj.estGraphon.splitPos, byGroup=True, make_show=False, savefig=True,
                     file_=directory_ + 'network_EM2.png')

## plot sequence of information criterion
showAIC(AIC_vec=EM_obj.AIC_vec, make_show=False, savefig=True, file_=directory_ + 'trajectory_AIC.png')

## plot final graphon estimate
if simulate:
    EM_obj.estGraphon.showColored(vmin=np.max([np.floor(np.min(graphon0.mat) / 0.05) *0.05, 0]),
                              vmax=np.min([np.ceil(np.max(graphon0.mat) / 0.05) *0.05, 1]),
                              log_scale=False if simulate else True, make_show=False, savefig=True,
                              file_=directory_ + 'graphon_est_EM.png')
else:
    EM_obj.estGraphon.showColored(log_scale=False if simulate else True, make_show=False, savefig=True,
                              file_=directory_ + 'graphon_est_EM.png')
EM_obj.estGraphon.showSlices(log_scale=False if simulate else True, make_show=False, savefig=True,
                             file_=directory_ + 'graphon_slices_EM.png')

## plot moves of latent positions in the final step
EM_obj.sample.showMove(useColor=simulate, showSplitPos=True, EMstep_sign=EM_obj.params.n_iter, make_show=False,
                       savefig=True, file_=directory_ + 'Us_move_last.png')

if simulate:
    ## plot difference between EM estimates and real U's
    EM_obj.sortG.showDiff(Us_type='est', splitPos_est=EM_obj.estGraphon.splitPos,
                          splitPos_real=graphon0.splitPos if hasattr(graphon0, 'splitPos') else None,
                          EMstep_sign='EM', make_show=False, savefig=True, file_=directory_ + 'Us_diffReal_EM.png')



### Save results
graph_simple = {'A': EM_obj.sortG.A, 'labels': EM_obj.sortG.labels_(),
                'Us_real': EM_obj.sortG.Us_('real'), 'Us_est': EM_obj.sortG.Us_('est')}
graphon_simple = {'mat': EM_obj.estGraphon.mat, 'order': EM_obj.estGraphon.order, 'nSubs': EM_obj.estGraphon.nSubs,
                  'nKnots': EM_obj.estGraphon.nKnots, 'splitPos': EM_obj.estGraphon.splitPos,
                  'tau': EM_obj.estGraphon.tau, 'tau_sep': EM_obj.estGraphon.tau_sep,
                  'tau_sep_ext': EM_obj.estGraphon.tau_sep_ext,
                  'P_mat': EM_obj.estGraphon.P_mat, 'P_mat_sep_ext': EM_obj.estGraphon.P_mat_sep_ext,
                  'theta': EM_obj.estGraphon.theta, 'theta_sep_ext': EM_obj.estGraphon.theta_sep_ext}

with open(directory_ + 'final_result.pkl', 'wb') as output:
    pickle.dump(graph_simple, output, protocol=3)
    pickle.dump(graphon_simple, output, protocol=3)

########################################################################################################################
################################################### Algorithm End ######################################################
########################################################################################################################

