'''

Fit the graphon for a given graph and conditional on the U_i.
@author: Benjamin Sischka

'''
import numpy as np
import cvxopt
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import scipy.ndimage.filters as flt
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import sys
import io
import warnings
from copy import copy
from SBSGMpy.graphon import Graphon, fctToFct
from SBSGMpy.graphon import colorBar

# split population into N_fine groups
def split_Ind(N, N_fine = None, h = None):
    # N = number of total, N_fine = number of groups, h = size of groups
    if N_fine is None:
        if h is None:
            raise TypeError('no information about the segmentation')
        N_fine = int(np.floor(N / h))
    else:
        if N_fine > N:
            raise TypeError('amount of segments must be less than index vector size')
        if not h is None:
            warnings.warn('\'h\' has not been used')
            print('UserWarning: \'h\' has not been used')
    h = N / N_fine
    result = type('', (), {})()
    result.seqInd = [range(N)[int(round(i*h)):int(round((i+1)*h))] for i in range(N_fine)]
    result.h = h
    result.N_fine = N_fine
    return(result)
#out: groups of similar size

# divide overall number of knots proportionally over intervals with irregular lengths
def divKnots(nKnots, splitPos):
    if (nKnots < ((len(splitPos) -1) *2)):
        raise TypeError('too few inner knots compared to number of intervals (per interval not less than 2 inner knots)')
    nKnots_dec = np.diff(splitPos) * nKnots
    nKnots_ = nKnots_dec.copy()
    rel_diff = np.array([(1. if (nKnots_dec_i < 1) else ((nKnots_dec_i - np.floor(nKnots_dec_i)) / nKnots_dec_i)) for nKnots_dec_i in nKnots_dec])
    n_ceil = int(round(nKnots - np.floor(nKnots_dec).sum()))
    pos_ceil = rel_diff.argsort()[-n_ceil:]
    pos_floor = rel_diff.argsort()[:(len(splitPos) -1 - n_ceil)]
    nKnots_[pos_ceil] = np.ceil(nKnots_[pos_ceil])
    nKnots_[pos_floor] = np.floor(nKnots_[pos_floor])
    nKnots_ = nKnots_.astype(int)
    loop_ind = 0
    while np.any(nKnots_ < 2):
        nKnots_[np.argmin(nKnots_)] += 1
        nKnots_[np.argmin(np.array([(((nKnots_dec[i] - (nKnots_[i] -1)) / nKnots_dec[i]) if (nKnots_[i] >= 3) else np.inf) for i in range(len(nKnots_dec))]))] -= 1
        loop_ind += 1
        if (loop_ind >= ((len(splitPos) -1) *4)):
            warnings.warn('inner knots could not be distributed appropriately over all intervals, some of them have still less than 2 inner knots; see\nnKnots: ' + str(nKnots_) +\
                          '\ninputs were: \nnKnots = ' + nKnots.__str__() + '\nsplitPos = ' + 'np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')')
            print('UserWarning: inner knots could not be distributed appropriately over all intervals, some of them have still less than 2 inner knots; see\nnKnots: ' + str(nKnots_) +\
                  '\ninputs were: \nnKnots = ' + nKnots.__str__() + '\nsplitPos = ' + 'np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')')
            break
    nKnots_[nKnots_ < 2] = 2
    if (nKnots_.sum() != nKnots):
        warnings.warn('given number of inner knots [=' + str(nKnots) + '] and derived number of inner knots [=' + str(nKnots_.sum()) + '] differ')
        print('UserWarning: given number of inner knots [=' + str(nKnots) + '] and derived number of inner knots [=' + str(nKnots_.sum()) + '] differ')
    return(nKnots_)
#out: number of knots per interval

# estimate split points based on ordered adjacency matrix
def estSplitPos(A, nSubs):
    size_part = np.max([1, int(np.log(1.5 * A.shape[0] / nSubs) ** 2)])
    diff_list = np.zeros((0, A.shape[1] // size_part - 1))
    for i in range(size_part):
        n_part = (A.shape[0] - i) // size_part
        if (n_part > 1):
            N_trim = n_part * size_part
            A_new = A[i:(N_trim + i)][:, i:(N_trim + i)]
            diff_list = np.append(diff_list, np.array([np.append(np.nansum(np.diff(A_new.reshape(N_trim, n_part, size_part).mean(axis=2).reshape((n_part, size_part, n_part)).mean(axis=1), 1) ** 2, axis=0), [] if (n_part == (A.shape[1] // size_part)) else [0])]), axis=0)
    diff_locWhichMax = diff_list.argmax(axis=0)
    diff_sort = diff_list.max(axis=0).argsort()[::-1]
    diff_whichMax = np.array([]).astype(int)
    while ((len(diff_whichMax) < (nSubs - 1)) and (len(diff_sort) > 0)):
        diff_whichMax = np.append(diff_whichMax, np.array([]).astype(int) if np.any(np.abs(diff_whichMax - diff_sort[0]) == 1) else diff_sort[0])
        diff_sort = diff_sort[1:]
    diff_whichMax.sort()
    splitPos = np.concatenate(([0.], (size_part * (diff_whichMax + 1) + diff_locWhichMax[diff_whichMax]) / A.shape[1], [1.]))
    if (len(splitPos) < (nSubs + 1)):
        warnings.warn('split positions based on blockwise distances in the adjacency matrix are too few, artificial splits need to be added; see\nsplitPos = ' + \
                      'np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')' + '\nlen_splitPos = ' + str(len(splitPos)) + '\nnSubs = -1 + ' + str((nSubs + 1)))
        print('UserWarning: split positions based on blockwise distances in the adjacency matrix are too few, artificial splits need to be added; see\nsplitPos = ' + \
              'np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')' + '\nlen_splitPos = ' + str(len(splitPos)) + '\nnSubs = -1 + ' + str((nSubs + 1)))
    while (len(splitPos) < (nSubs + 1)):
        ## if [len(splitPos) -1 < (nSubs +1) - len(splitPos)], then the splits should be distributed more equally; however, should not happen very often
        splitPos = np.sort(np.append(splitPos, (np.diff(splitPos) / 2 + splitPos[:-1])[np.argsort(np.diff(splitPos))[::-1][:np.min([len(splitPos) - 1, (nSubs + 1) - len(splitPos)])]]))
        print('add split position to get', len(splitPos)-1, 'communities; see\nsplitPos =', 'np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')', '\nlen_splitPos =', len(splitPos), '\nnSubs = -1 +', (nSubs + 1))
    return(splitPos)
#out: estimate of split positions (including '0.0' and '1.0')

# smooth adjacency matrix
def smoothAdjMat(mat, h, N_fine, cuts):
    # mat = adjacency matrix, h = bandwidth of histogram summarization, N_fine = fineness of the graphon/number of communities, cuts = proportions at which the adj. mat. should be cut
    N = mat.shape[0]
    if cuts is None:
        if N_fine is None:
            if h is None:
                warnings.warn('bandwidth \'h\' has automatically been set to 1')
                print('UserWarning: bandwidth \'h\' has automatically been set to 1')
                h = 1
            seqIndObj = split_Ind(N = N, h = h)
            kl = seqIndObj.seqInd
            h = seqIndObj.h
            N_fine = seqIndObj.N_fine
            mat_cuts = [max(rnge)+1 for rnge in kl][:-1]
            adjust_ = False
        else:
            seqIndObj = split_Ind(N = N, N_fine = N_fine)
            kl = seqIndObj.seqInd
            h = seqIndObj.h
            N_fine = seqIndObj.N_fine
            mat_cuts = [max(rnge)+1 for rnge in kl][:-1]
            adjust_ = False
            if not h is None:
                warnings.warn('\'h\' has not been used')
                print('UserWarning: \'h\' has not been used')
    else:
        mat_cuts = np.sort(np.floor(cuts * N).astype(int))
        kl = [range(cutS, cutE) for cutS, cutE in zip(np.append([0], mat_cuts), np.append(mat_cuts, [N]))]
        N_fine = len(kl)
        h = N / N_fine
        adjust_ = True
        if not ((h is None) and (N_fine is None)):
            warnings.warn('only \'cuts\' has been used to sequence the matrix')
            print('UserWarning: only \'cuts\' has been used to sequence the matrix')
    H = np.zeros(shape=np.repeat(N_fine,2))  # hist matrix basis
    for k0 in range(N_fine):
        for k1 in range(N_fine):
            H[k0,k1]=np.mean(mat[kl[k0]][:, kl[k1]]) * ((1 / (1 - (1/len(kl[k0])))) if k0 == k1 else 1)
    if adjust_:
        freq = np.diff(np.concatenate(([0], mat_cuts, [N])))
        H = np.repeat(np.repeat(H, freq, axis=0), freq, axis=1)
    result = type('', (), {})()
    result.H = H
    result.kl = kl
    result.h = h
    result.N_fine = N_fine
    result.mat_cuts = mat_cuts
    return(result)
#out: H = smoothed adjacency matrix, kl = limits of hist. matrix, h = bandwidth of histogram summarization,
#     N_fine = fineness of the graphon/number of communities, mat_cuts = positions at which the adj. mat. has been cut

# extend adjacency matrix based on Us
def ExtAdjMatByUs(mat, Us, extFac):
    # mat = adjacency matrix, Us = vector of Us, extFac = factor of extension
    if not mat.shape[0] == len(Us):
        raise ValueError('dimension of \'mat\' and length of \'Us\' don\'t match')
    N = mat.shape[0]
    extSize=N*extFac
    pos = np.floor(Us * extSize).astype(int)
    extA = np.zeros(np.repeat(extSize, 2))
    extACompare = np.zeros(np.repeat(extSize, 2))
    for i in range(N):
        for j in range(N):
            extA[pos[i],pos[j]] += mat[i,j]
            extACompare[pos[i],pos[j]] += 1
    np.fill_diagonal(extACompare, np.array([(x - np.sqrt(x)) for x in np.diag(extACompare)]))  # for correcting the diagonal elements estimation
    result = type('', (), {})()
    result.pos = pos
    result.extA = extA
    result.extACompare = extACompare
    return(result)
#out: pos = new positions of U_i's, extA = extended adjacency matrix, extACompare = extended comparing matrix


# Define an Estimator Class
class Estimator:

    def __init__(self, sortG):
        # sortG = sorted extended graph
        if sortG.sorting is None:
            warnings.warn('no specification about Us_type (see sortG.sorting), empirical degree ordering is used')
            print('UserWarning: no specification about Us_type (see sortG.sorting), empirical degree ordering is used')
            sortG.sort(Us_type='emp')
            warnings.warn('input graph is now sorted by empirical degree')
            print('UserWarning: input graph is now sorted by empirical degree')
        self.sortG = sortG
        self.graph_updated = False
    def GraphonEstByHist(self, h=None, N_fine=None, cuts=None):
         # h = bandwidth of histogram summarization, N_fine = fineness of the graphon, cuts = proportions at which the adj. mat. should be cut
        return (Graphon(mat=smoothAdjMat(mat=self.sortG.A, h=h, N_fine=N_fine, cuts=cuts).H))
    def GraphonEstBySmoothHist(self, extFac=1, h=None, N_fine=None, cuts=None, sigma=1):
         # extFac = factor of extension ( -> manipulate smoothing), h = bandwidth of histogram summarization, N_fine = fineness of the graphon, sigma = standard deviation of kernel smoother
        H = np.kron(smoothAdjMat(mat=self.sortG.A, h=h, N_fine=N_fine, cuts=cuts).H, np.ones((extFac, extFac)))
        return (Graphon(mat=flt.gaussian_filter(H, sigma * extFac, mode='reflect')))
    def GraphonEstByExtHist(self, extFac=3, h=None, N_fine=None, cuts=None):
         # extFac = factor of extension ( -> manipulate smoothing), h = bandwidth of histogram summarization, N_fine = fineness of the graphon
        extAdjObj = ExtAdjMatByUs(mat=self.sortG.A, Us=self.sortG.Us_(self.sortG.sorting), extFac=extFac)
        H_uncorObj = smoothAdjMat(mat=extAdjObj.extA, h=h * extFac, N_fine=N_fine, cuts=cuts)
        HWeights = np.maximum(smoothAdjMat(mat=extAdjObj.extACompare, h=h * extFac, N_fine=N_fine, cuts=cuts).H, 1 / (H_uncorObj.h ** 2 * 2))
        freq = np.histogram(a=extAdjObj.pos, bins=[kl_[0] for kl_ in H_uncorObj.kl] + [H_uncorObj.kl[-1][-1] + 1])[0]
        return (Graphon(mat=np.repeat(np.repeat(H_uncorObj.H * (1 / HWeights), freq, axis=0), freq, axis=1)))
    def GraphonEstByMovAve(self, h=3, Us_mult=None):
        if Us_mult is None:
            Us_mult = self.sortG.Us_(self.sortG.sorting).reshape(1, self.sortG.N)
        A = np.zeros((np.repeat(self.sortG.N, 2)))
        for i in range(Us_mult.shape[0]):
            A += self.sortG.A[np.argsort(Us_mult[i])][:, np.argsort(Us_mult[i])]
        A *= 1 / Us_mult.shape[0]
        return (Graphon(mat=flt.uniform_filter(A, size=h, mode='constant', cval=0) * (1 / flt.uniform_filter(np.ones(A.shape) - np.identity(A.shape[0]), size=h, mode='constant', cval=0))))
    def GraphonEstByFilter(self, extFac=3, sigma=1, N_fine=None, Us_mult=None):
         # extFac = factor of extension, sigma = standard deviation of kernel smoother, N_fine = fineness of graphon
        if N_fine is None:
            N_fine = self.sortG.N
        if Us_mult is None:
            Us_mult = self.sortG.Us_(self.sortG.sorting).reshape(1, self.sortG.N)
        extA_sum, extACompare_sum = np.zeros((np.repeat(extFac * self.sortG.N, 2))), np.zeros((np.repeat(extFac * self.sortG.N, 2)))
        for i in range(Us_mult.shape[0]):
            extAdjObj = ExtAdjMatByUs(mat=self.sortG.A, Us=Us_mult[i], extFac=extFac)
        extA_sum += extAdjObj.extA
        extACompare_sum += extAdjObj.extACompare
        extASmooth_distort = flt.gaussian_filter(extA_sum, sigma=sigma * extFac, mode='constant', cval=0) * (1 / flt.gaussian_filter(extACompare_sum, sigma=sigma * extFac, mode='constant', cval=0))
        choPos = np.percentile(np.append([0, 1], Us_mult.reshape(Us_mult.shape[0] * self.sortG.N, )), q=np.linspace(0, 100, N_fine + 2))[1:-1] * (extASmooth_distort.shape[0] - 1)
        extASmooth_rap1 = np.multiply(extASmooth_distort[np.floor(choPos).astype(int)].T, (np.floor(choPos) + 1 - choPos)) + np.multiply(extASmooth_distort[np.ceil(choPos).astype(int)].T, choPos - np.floor(choPos))
        return (Graphon(mat=np.multiply(extASmooth_rap1[np.floor(choPos).astype(int)].T, (1 - (choPos - np.floor(choPos)))) + np.multiply(extASmooth_rap1[np.ceil(choPos).astype(int)].T, choPos - np.floor(choPos))))
    def GraphonEstBySimpleFilter(self, sigma=1, Us_mult=None):
        if Us_mult is None:
            Us_mult = self.sortG.Us_(self.sortG.sorting).reshape(1, self.sortG.N)
        A = np.zeros((np.repeat(self.sortG.N, 2)))
        for i in range(Us_mult.shape[0]):
            A += self.sortG.A[np.argsort(Us_mult[i])][:, np.argsort(Us_mult[i])]
        A *= 1 / Us_mult.shape[0]
        return (Graphon(mat=flt.gaussian_filter(A, sigma=sigma, mode='constant', cval=0) * (1 / flt.gaussian_filter(np.ones(A.shape) - np.identity(A.shape[0]), sigma=sigma, mode='constant', cval=0))))
    def GraphonEstBySpline(self, k=1, nSubs=None, nKnots=None, splitPos=None, est_splitPos=True, useOneBasis=False, tau=None, tau_sep=None, \
                           optForAIC=False, lambdaMin=None, lambdaMax=None, calcAIC=False, lambda_=None, critType='ICL', \
                           adjustSubs=False, lambda_adjustSubs=None, adjustQuantiles=False, lambda_adjustQuant=None, \
                           Us_mult=None, canonical=False, updateGraph=True, printWarn=True):
        # k = degree of splines (only 0 and 1 are implemented), nKnots = number of inner knots, lambda_ = parameter of penalty, Us_mult = multiple U-vectors in form of a matrix,
        # returnAIC = logical whether graphon estimate or AIC should be returned
        if Us_mult is None:
            Us_mult = self.sortG.Us_(self.sortG.sorting).reshape(1, self.sortG.N)
        m = Us_mult.shape[0]
        if np.all([optForAIC is False, calcAIC is True, True if (lambda_ is None) else np.all(np.isnan(lambda_))]):
            raise TypeError('information about penalization parameter lambda_ is required')
        if (optForAIC or calcAIC) and (critType not in ['AIC', 'AICc', 'AICL1', 'AICL2', 'AICL2b', 'ICL']):
            raise ValueError('criterion must be one of \'AIC\', \'AICc\', \'AICL1\', \'AICL2\', \'AICL2b\', \'ICL\'')
        if adjustSubs and adjustQuantiles:
            raise TypeError('only one type of adjustment should be used')
        if ((tau is None) and (tau_sep is None)):
            if nKnots is None:
                nKnots = 20
                warnings.warn('number of inner knots has automatically been set to ' + nKnots.__str__())
                print('UserWarning: number of inner knots has automatically been set to ' + nKnots.__str__())
            if nSubs is None:
                if (np.isscalar(nKnots) and (splitPos is None)):
                    warnings.warn('number of segments has automatically been set to 1')
                    print('UserWarning: number of segments has automatically been set to 1')
                nSubs = (1 if np.isscalar(nKnots) else len(nKnots)) if (splitPos is None) else (len(splitPos) - 1)
            if splitPos is None:
                # consider splitPos in 'network_0' of application.py
                if (est_splitPos and (nSubs >1)):
                    splitPos = estSplitPos(A=self.sortG.A, nSubs=nSubs)
                else:
                    splitPos = np.linspace(0,1,nSubs+1)
            else:
                if (est_splitPos and (nSubs >1)):
                    warnings.warn('split positions have been used as specified (see \'splitPos=np.array(' + np.array2string(splitPos, separator=',').replace('\n ', ' \\\n\t') + ')\')\n\tand have not been estimated as requested (see \'est_splitPos=True\')')
                    print('UserWarning: split positions have been used as specified (see \'splitPos=np.array(' + np.array2string(splitPos, separator=',').replace('\n ', ' \\\n\t') + ')\')\n\tand have not been estimated as requested (see \'est_splitPos=True\')')
            if (len(splitPos) != (nSubs +1)):
                raise TypeError('number of split positions and specified number of segments do not match')
            freqVecSub = np.sum([np.diff(np.sum(Us <= splitPos.reshape(nSubs + 1, 1), axis=1)) for Us in Us_mult], axis=0)
            if adjustSubs:
                if (nSubs < 2):
                    warnings.warn('adjustment of segments can only be applied when there is more than one block')
                    print('UserWarning: adjustment of segments can only be applied when there is more than one block')
                else:
                    splitPos_old = splitPos.copy()
                    if lambda_adjustSubs is None:
                        lambda_adjustSubs = 1
                    propVec = lambda_adjustSubs * freqVecSub / (self.sortG.N * m) + (1 - lambda_adjustSubs) * np.diff(splitPos)
                    splitPos = np.cumsum(np.append([0], propVec))
                    Us_mult = [(((Us_mult - splitPos_old[subCom]) / np.diff(splitPos_old)[subCom]) * np.diff(splitPos)[subCom] + splitPos[subCom]) for subCom in [np.searchsorted(splitPos_old, Us_mult) - 1]][0]
                    if updateGraph:
                        self.sortG.update(Us_est=[(((self.sortG.Us_(self.sortG.sorting) - splitPos_old[subCom]) / np.diff(splitPos_old)[subCom]) * np.diff(splitPos)[subCom] + splitPos[subCom]) for subCom in [np.searchsorted(splitPos_old, self.sortG.Us_(self.sortG.sorting)) - 1]][0])
                        # only Us_est should be changed; if self.sortG.sorting=='real'_or_'emp' the result will anyway be saved as Us_est
                        self.graph_updated = True
                        if printWarn:
                            warnings.warn('U\'s from input graph have been updated')
                            print('UserWarning: U\'s from input graph have been updated')
                    else:
                        self.Us = [(((self.sortG.Us_(self.sortG.sorting) - splitPos_old[subCom]) / np.diff(splitPos_old)[subCom]) * np.diff(splitPos)[subCom] + splitPos[subCom]) for subCom in [np.searchsorted(splitPos_old, self.sortG.Us_(self.sortG.sorting)) - 1]][0]
                        if printWarn:
                            warnings.warn('U\'s from input graph should be adjusted, use [].Us from this estimation object')
                            print('UserWarning: U\'s from input graph should be adjusted, use [].Us from this estimation object')
            if np.isscalar(nKnots):
                nKnots = divKnots(nKnots=nKnots, splitPos=splitPos)
            if not np.allclose([np.min(splitPos), np.max(splitPos), len(splitPos)], [0, 1, nSubs+1]):
                raise TypeError('split positions have wrong specification, it must be min=0, max=1, and #=nSubs+1; see\nnSubs = ' + str(nSubs) + \
                                '\nsplitPos = np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')')
            if (len(nKnots) != nSubs):
                raise TypeError('array of numbers of inner B-spline knots has wrong specification, length must be equal to nSubs; see\nnSubs = ' + str(nSubs) + \
                                '\nnKnots = np.array(' + np.array2string(nKnots, separator=', ').replace('\n ', ' \\\n\t') + ')')
            tau = np.concatenate(([np.concatenate((np.repeat(splitPos[i], k) if (i == 0) else [], np.linspace(splitPos[i], splitPos[i + 1], nKnots[i]), np.repeat(splitPos[i + 1], k)))[:-1] for i in range(nSubs)] + [[splitPos[-1]]]))
            tau_sep = [np.concatenate((np.repeat(splitPos[i], k), np.linspace(splitPos[i], splitPos[i + 1], nKnots[i]), np.repeat(splitPos[i + 1], k))) for i in range(nSubs)]
            if not np.all(np.concatenate([tau_sep[i][(0 if (i == 0) else (k+1)):] for i in range(len(tau_sep))]) == tau):
                raise TypeError('tau and tau_sep do not match, something went wrong in the calculations; see\ntau = np.array(' + np.array2string(tau, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                '\ntau_sep = ' + '[' + ', \\\n\t'.join([('np.array(' + np.array2string(tau_sep_i, separator=', ').replace('\n ', ' \\\n\t') + ')') for tau_sep_i in tau_sep]) + ']')
        else:
            if (tau is None):
                tau = np.concatenate([tau_sep[i][(0 if (i == 0) else (k+1)):] for i in range(len(tau_sep))])
            if (tau_sep is None):
                sVecP = np.where(np.isclose(tau[k:] - tau[:-k], 0))[0]
                lmts = [np.array([sVecP + shft_ for shft_ in [0, k+1]])[(0,1),(i,i+1)] for i in range(len(sVecP)-1)]
                tau_sep = [tau[lmts[i][0]:lmts[i][1]] for i in range(len(lmts))]
            if not np.allclose(np.concatenate([tau_sep[i][(0 if (i == 0) else (k+1)):] for i in range(len(tau_sep))]), tau):
                raise TypeError('tau and tau_sep do not match; see\ntau = np.array(' + np.array2string(tau, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                '\ntau_sep = ' + '[' + ', \\\n\t'.join([('np.array(' + np.array2string(tau_sep_i, separator=', ').replace('\n ', ' \\\n\t') + ')') for tau_sep_i in tau_sep]) + ']')
            if (np.any(np.diff(np.where(np.isclose(tau[k:] - tau[:-k], 0))[0]) == 1) or np.any([np.any(np.isclose(np.diff(tau_sep_i[k:-k]), 0)) for tau_sep_i in tau_sep])):
                raise TypeError('tau and tau_sep are not specified suitably, consecutive knot positions should not be equal; see\ntau = np.array(' + np.array2string(tau, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                '\ntau_sep = ' + '[' + ', \\\n\t'.join([('np.array(' + np.array2string(tau_sep_i, separator=', ').replace('\n ', ' \\\n\t') + ')') for tau_sep_i in tau_sep]) + ']')
            if not np.allclose([tau[k], tau[-(k+1)], tau_sep[0][k], tau_sep[-1][-(k+1)]], [0., 1., 0., 1.]):
                raise TypeError('tau and tau_sep are not specified suitably, min and max of inner knot positions should be 0 and 1, respectively; see' + \
                                '\nlimits of tau: ' + tau[k].__str__() + ', ' + tau[-(k+1)].__str__() + \
                                '\nlimits of tau_sep: ' + tau_sep[0][k].__str__() + ', ' + tau_sep[-1][-(k+1)].__str__() + \
                                '\ntau = np.array(' + np.array2string(tau, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                '\ntau_sep = ' + '[' + ', \\\n\t'.join([('np.array(' + np.array2string(tau_sep_i, separator=', ').replace('\n ', ' \\\n\t') + ')') for tau_sep_i in tau_sep]) + ']')
            if not nKnots is None:
                if (np.sum(nKnots) != (len(tau) - 2*k - (len(np.atleast_1d(nKnots))-1)*(k-1))):
                    warnings.warn('nKnots and tau do not match, nKnots has not been used; see\nnKnots = ' + (str(nKnots) if np.isscalar(nKnots) else ('np.array(' + np.array2string(nKnots, separator=', ').replace('\n', ' \\\n\t') + ')')) + \
                                  '\ntau = np.array(' + np.array2string(tau, separator=', ').replace('\n ', ' \\\n\t') + ')')
                    print('UserWarning: nKnots and tau do not match, nKnots has not been used; see\nnKnots = ' + (str(nKnots) if np.isscalar(nKnots) else ('np.array(' + np.array2string(nKnots, separator=', ').replace('\n', ' \\\n\t') + ')')) + \
                          '\ntau = np.array(' + np.array2string(tau, separator=', ').replace('\n ', ' \\\n\t') + ')')
                if not np.isscalar(nKnots):
                    if not np.all(nKnots == np.array([(len(tau_sep_i) - 2 * k) for tau_sep_i in tau_sep])):
                        warnings.warn('nKnots and tau_sep do not match, nKnots has not been used; see\nnKnots = ' + (str(nKnots) if np.isscalar(nKnots) else ('np.array(' + np.array2string(nKnots, separator=', ').replace('\n', ' \\\n\t') + ')')) + \
                                      '\ntau_sep = ' + '[' + ', \\\n\t'.join([('np.array(' + np.array2string(tau_sep_i, separator=', ').replace('\n ', ' \\\n\t') + ')') for tau_sep_i in tau_sep]) + ']')
                        print('UserWarning: nKnots and tau_sep do not match, nKnots has not been used; see\nnKnots = ' + (str(nKnots) if np.isscalar(nKnots) else ('np.array(' + np.array2string(nKnots, separator=', ').replace('\n', ' \\\n\t') + ')')) + \
                              '\ntau_sep = ' + '[' + ', \\\n\t'.join([('np.array(' + np.array2string(tau_sep_i, separator=', ').replace('\n ', ' \\\n\t') + ')') for tau_sep_i in tau_sep]) + ']')
            if not splitPos is None:
                if not np.allclose(splitPos, tau[np.where(np.isclose(tau[k:] - tau[:-k], 0))[0] +k]):
                    warnings.warn('splitPos and tau do not match, splitPos has not been used; see\nsplitPos = np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                  '\ntau = np.array(' + np.array2string(tau, separator=', ').replace('\n ', ' \\\n\t') + ')')
                    print('UserWarning: splitPos and tau do not match, splitPos has not been used; see\nsplitPos = np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                          '\ntau = np.array(' + np.array2string(tau, separator=', ').replace('\n ', ' \\\n\t') + ')')
                if not np.allclose(splitPos, np.append([tau_sep_i[k] for tau_sep_i in tau_sep], [1])):
                    warnings.warn('splitPos and tau_sep do not match, splitPos has not been used; see\nsplitPos = np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                  '\ntau_sep = ' + '[' + ', \\\n\t'.join([('np.array(' + np.array2string(tau_sep_i, separator=', ').replace('\n ', ' \\\n\t') + ')') for tau_sep_i in tau_sep]) + ']')
                    print('UserWarning: splitPos and tau_sep do not match, splitPos has not been used; see\nsplitPos = np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                          '\ntau_sep = ' + '[' + ', \\\n\t'.join([('np.array(' + np.array2string(tau_sep_i, separator=', ').replace('\n ', ' \\\n\t') + ')') for tau_sep_i in tau_sep]) + ']')
            if np.any([np.any(np.diff(tau) < 0), round(tau[k], 5) != 0, round(tau[-k -1], 5) != 1]):
                raise TypeError('position of knots are wrong specified, tau must divide the interval [0, 1]; see\nsplitPos = np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                '\nnSubs = ' + str(nSubs) + '\nnKnots = ' + (str(nKnots) if np.isscalar(nKnots) else ('np.array(' + np.array2string(nKnots, separator=', ').replace('\n', ' \\\n\t') + ')')) + \
                                '\ntau = np.array(' + np.array2string(tau, separator=', ').replace('\n ', ' \\\n\t') + ')')
            if np.any([np.any(np.diff(np.concatenate(tau_sep)) < 0), round(np.concatenate(tau_sep)[k], 5) != 0, round(np.concatenate(tau_sep)[-k -1], 5) != 1, \
                       not np.allclose(np.array([tau_sep_i[k] for tau_sep_i in tau_sep])[1:], np.array([tau_sep_i[-k -1] for tau_sep_i in tau_sep])[:-1])]):
                raise TypeError('position of knots are wrong specified, tau_sep must divide the interval [0, 1]; see' + \
                                '\nsplitPos = np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                '\nnSubs = ' + str(nSubs) + '\nnKnots = ' + (str(nKnots) if np.isscalar(nKnots) else ('np.array(' + np.array2string(nKnots, separator=', ').replace('\n', ' \\\n\t') + ')')) + \
                                '\ntau_sep = ' + '[' + ', \\\n\t'.join([('np.array(' + np.array2string(tau_sep_i, separator=', ').replace('\n ', ' \\\n\t') + ')') for tau_sep_i in tau_sep]) + ']')
            splitPos = np.append([tau_sep_i[k] for tau_sep_i in tau_sep], [1])
            nSubs = len(splitPos) -1
            freqVecSub = np.sum([np.diff(np.sum(Us <= splitPos.reshape(nSubs + 1, 1), axis=1)) for Us in Us_mult], axis=0)
            if adjustSubs:
                if (nSubs < 2):
                    warnings.warn('adjustment of segments can only be applied when there is more than one block')
                    print('UserWarning: adjustment of segments can only be applied when there is more than one block')
                else:
                    splitPos_old = splitPos.copy()
                    if lambda_adjustSubs is None:
                        lambda_adjustSubs = 1
                    propVec = lambda_adjustSubs * freqVecSub / (self.sortG.N * m) + (1 - lambda_adjustSubs) * np.diff(splitPos)
                    splitPos = np.cumsum(np.append([0], propVec))
                    tau_subNb = (np.ones((len(tau))) * (-1)).astype(int)
                    for i in range(nSubs):
                        tau_subNb[np.logical_and(tau > splitPos_old[i], tau <= splitPos_old[i + 1])] = i
                    for i in range(nSubs):
                        tau[tau_subNb == i] = (splitPos[i + 1] - splitPos[i]) * ((tau[tau_subNb == i] - splitPos_old[i]) / (splitPos_old[i + 1] - splitPos_old[i])) + splitPos[i]
                    tau_sep = [(splitPos[i + 1] - splitPos[i]) * ((tau_sep[i] - splitPos_old[i]) / (splitPos_old[i + 1] - splitPos_old[i])) + splitPos[i] for i in range(nSubs)]
                    Us_mult = [(((Us_mult - splitPos_old[subCom]) / np.diff(splitPos_old)[subCom]) * np.diff(splitPos)[subCom] + splitPos[subCom]) for subCom in [np.searchsorted(splitPos_old, Us_mult) - 1]][0]
                    if updateGraph:
                        self.sortG.update(Us_est=[(((self.sortG.Us_(self.sortG.sorting) - splitPos_old[subCom]) / np.diff(splitPos_old)[subCom]) * np.diff(splitPos)[subCom] + splitPos[subCom]) for subCom in [np.searchsorted(splitPos_old, self.sortG.Us_(self.sortG.sorting)) - 1]][0])
                        self.graph_updated = True
                        if printWarn:
                            warnings.warn('U\'s from input graph have been updated')
                            print('UserWarning: U\'s from input graph have been updated')
                    else:
                        self.Us = [(((self.sortG.Us_(self.sortG.sorting) - splitPos_old[subCom]) / np.diff(splitPos_old)[subCom]) * np.diff(splitPos)[subCom] + splitPos[subCom]) for subCom in [np.searchsorted(splitPos_old, self.sortG.Us_(self.sortG.sorting)) - 1]][0]
                        if printWarn:
                            warnings.warn('U\'s from input graph should be adjusted, use [].Us from this estimation object')
                            print('UserWarning: U\'s from input graph should be adjusted, use [].Us from this estimation object')
            nKnots = np.array([len(tau_sep_i[k:-k]) for tau_sep_i in tau_sep])
            if not np.all([nSubs == len(tau_sep), np.all(nKnots == np.array([(len(tau_sep_i) - 2*k) for tau_sep_i in tau_sep])), np.allclose(splitPos, np.append([tau_sep_i[k] for tau_sep_i in tau_sep], [1]))]):
                raise TypeError('splitPos, nSubs, or nKnots are not specified in accordance with tau_sep; see\nsplitPos = np.array(' + np.array2string(splitPos, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                '\nnSubs = ' + str(nSubs) + '\nnKnots = np.array(' + np.array2string(nKnots, separator=', ').replace('\n', ' \\\n\t') + ')' + \
                                '\ntau = np.array(' + np.array2string(tau, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                '\ntau_sep = ' + '[' + ', \\\n\t'.join([('np.array(' + np.array2string(tau_sep_i, separator=', ').replace('\n ', ' \\\n\t') + ')') for tau_sep_i in tau_sep]) + ']')
        if (adjustQuantiles and (nSubs > 1)):
            raise TypeError('for hierarchical estimation, use adjustSubs; adjustQuantiles is only suitable for an overall smooth estimation')
        freqVecTau = np.sum([np.diff(np.sum(Us <= np.concatenate([tau_sep_i[k:-(k + 1)] for tau_sep_i in tau_sep] + [np.array([1])])[:, np.newaxis], axis=1)) for Us in Us_mult], axis=0)
        if adjustQuantiles:
            tau_old = tau.copy()
            if lambda_adjustQuant is None:
                lambda_adjustQuant = 1
            propVec = lambda_adjustQuant * freqVecTau / self.sortG.N + (1 - lambda_adjustQuant) * np.diff(tau[k:-k])
            tau = np.concatenate((tau[:k + 1], np.cumsum(propVec), tau[-k:]))
            sVecP = np.where(np.isclose(tau[k:] - tau[:-k], 0))[0]
            lmts = [np.array([sVecP + shft_ for shft_ in [0, k+1]])[(0,1),(i,i+1)] for i in range(len(sVecP)-1)]
            tau_sep = [tau[lmts[i][0]:lmts[i][1]] for i in range(len(lmts))]
            Us_mult = [(((Us_mult - tau_old[subCom]) / np.diff(tau_old)[subCom]) * np.diff(tau)[subCom] + tau[subCom]) for subCom in [np.searchsorted(tau_old, Us_mult) - 1]][0]
            if updateGraph:
                self.sortG.update(Us_est=[(((self.sortG.Us_(self.sortG.sorting) - tau_old[subCom]) / np.diff(tau_old)[subCom]) * np.diff(tau)[subCom] + tau[subCom]) for subCom in [np.searchsorted(tau_old, self.sortG.Us_(self.sortG.sorting)) - 1]][0])
                # only Us_est should be changed; if self.sortG.sorting=='real'_or_'emp' the result will anyway be saved as Us_est
                self.graph_updated = True
                if printWarn:
                    warnings.warn('U\'s from input graph have been updated')
                    print('UserWarning: U\'s from input graph have been updated')
            else:
                self.Us = [(((self.sortG.Us_(self.sortG.sorting) - tau_old[subCom]) / np.diff(tau_old)[subCom]) * np.diff(tau)[subCom] + tau[subCom]) for subCom in [np.searchsorted(tau_old, self.sortG.Us_(self.sortG.sorting)) - 1]][0]
                if printWarn:
                    warnings.warn('U\'s from input graph should be adjusted, use [].Us from this estimation object')
                    print('UserWarning: U\'s from input graph should be adjusted, use [].Us from this estimation object')
        if useOneBasis:
            if np.any(np.array([np.logical_and(Us_mult >= splitPos[i_], Us_mult < splitPos[i_ + 1]).sum() for i_ in range(nSubs)]) == 0):
                warnings.warn('the following communities are empty: ' + ', '.join((np.where(np.array([np.logical_and(Us_mult >= splitPos[i_], Us_mult < splitPos[i_ + 1]).sum() for i_ in range(nSubs)]) == 0)[0] +1).astype(str)) + '; estimation results in .5')
                print('UserWarning: the following communities are empty: ' + ', '.join((np.where(np.array([np.logical_and(Us_mult >= splitPos[i_], Us_mult < splitPos[i_ + 1]).sum() for i_ in range(nSubs)]) == 0)[0] +1).astype(str)) + '; estimation results in .5')
            nSpline1d = len(tau) - k - 1
            nSpline = nSpline1d**2
            if (optForAIC or calcAIC):
                N_squ_eff = np.logical_not(np.isnan(self.sortG.A)).sum() - np.logical_not(np.isnan(np.diagonal(self.sortG.A))).sum()
                N_squ_eff_log = np.log(N_squ_eff)
            if k == 1:
                B = np.array([np.array([interpolate.bisplev(x=np.sort(Us), y=np.sort(Us), tck=(tau, tau, np.lib.pad([1], (i, nSpline - i - 1), 'constant', constant_values=(0)), k, k), dx=0, dy=0) for i in np.arange(nSpline)])[:, np.argsort(np.argsort(Us)), :][:, :, np.argsort(np.argsort(Us))] for Us in Us_mult])
            else:
                raise TypeError('B-splines of degree k = ' + k.__str__() + ' have not been implemented yet')
            B_cbind = np.array([np.delete(B[l].reshape(nSpline, self.sortG.N ** 2), np.arange(self.sortG.N) * (self.sortG.N + 1), axis=1) for l in range(m)])
            if canonical:
                raise TypeError('canonical B-splines representation has not been implemented yet')
                # A1 = np.vstack((np.array([np.pad(np.append(-A_part, A_part), (nSpline1d * i, nSpline1d * (nSpline1d - i - 2)), 'constant', constant_values=(0, 0)) for i in range(nSpline1d - 1)]), np.identity(nSpline), -np.identity(nSpline)))
            else:
                A1 = np.vstack((np.identity(nSpline), -np.identity(nSpline)))
            A2 = np.array([]).reshape((nSpline, 0))
            for i in range(nSpline1d):
                for j in range(i + 1, nSpline1d):
                    NullMat = np.zeros((nSpline1d, nSpline1d))
                    NullMat[i, j], NullMat[j, i] = 1, -1
                    A2 = np.hstack((A2, NullMat.reshape(nSpline, 1)))
            A2 = A2.T
            L_part = block_diag(*[np.vstack((np.identity(dim_i)[:-1] - np.hstack((np.zeros((dim_i - 1, 1)), np.identity(dim_i - 1))))) for dim_i in (nKnots +k -1)])
            I_part = np.identity(nSpline1d)
            penalize = np.dot(np.kron(I_part, L_part).T, np.kron(I_part, L_part)) + np.dot(np.kron(L_part, I_part).T, np.kron(L_part, I_part))
            G_ = cvxopt.matrix(-A1)
            A_ = cvxopt.matrix(A2)
            cvxopt.solvers.options['show_progress'] = False
            def estTheta(lambda_, onlyAIC=False):
                theta_t = np.repeat(self.sortG.density, nSpline)
                differ = 5
                index_marker = 1
                while (differ > 0.01**2):
                    Pi = np.minimum(np.maximum(np.sum(B.swapaxes(1, 3) * theta_t, axis=3), 1e-5), 1 - 1e-5)
                    mat1 = (B.swapaxes(0, 1) * ((self.sortG.A * (1 / Pi)) - ((1 - self.sortG.A) * (1 / (1 - Pi))))).swapaxes(0, 1)
                    score = np.sum(np.sum(np.nansum(mat1, axis=0), axis=1), axis=1) - np.sum(np.nansum([np.diagonal(mat1[l], axis1=1, axis2=2).T for l in range(m)], axis=0), axis=0)
                    mat2 = 1 / (Pi * (1 - Pi))
                    fisher = np.sum(np.array([np.dot(B_cbind[l] * np.delete(mat2[l].reshape(self.sortG.N ** 2, ), np.arange(self.sortG.N) * (self.sortG.N + 1)), B_cbind[l].T) for l in range(m)]), 0)
                    P_ = cvxopt.matrix(fisher + lambda_ * penalize)
                    q_ = cvxopt.matrix(-score + lambda_ * np.dot(theta_t, penalize))
                    if canonical:
                        h_ = cvxopt.matrix(np.dot(A1, theta_t) + np.append(np.zeros(nSpline1d - 1 + nSpline, ), np.ones(nSpline, )))
                    else:
                        h_ = cvxopt.matrix(np.dot(A1, theta_t) + np.append(np.zeros(nSpline, ), np.ones(nSpline, )))
                    b_ = cvxopt.matrix(np.dot(-A2, theta_t))
                    delta_t = np.squeeze(np.array(cvxopt.solvers.qp(P=P_, q=q_, G=G_, h=h_, A=A_, b=b_)['x']))
                    theta_tOld = copy(theta_t)
                    theta_t = delta_t + theta_t
                    differ = (1 / nSpline) * np.sum((theta_t - theta_tOld)**2)
                    print('Iteration of estimating theta:', index_marker)
                    index_marker = index_marker + 1
                    if index_marker > 10:
                        warnings.warn('Fisher scoring did not converge; see\ntheta_tOld = np.array(' + np.array2string(theta_tOld, separator=',').replace('\n ', ' \\\n\t') + ')' + \
                                      '\ntheta_t = np.array(' + np.array2string(theta_t, separator=',').replace('\n ', ' \\\n\t') + ')' + \
                                      '\ndelta_t = np.array(' + np.array2string(np.round(theta_t - theta_tOld, 4), separator=',').replace('\n ', ' \\\n\t') + ')')
                        print('UserWarning: Fisher scoring did not converge; see\ntheta_tOld = np.array(' + np.array2string(theta_tOld, separator=',').replace('\n ', ' \\\n\t') + ')' + \
                              '\ntheta_t = np.array(' + np.array2string(theta_t, separator=',').replace('\n ', ' \\\n\t') + ')' + \
                              '\ndelta_t = np.array(' + np.array2string(np.round(theta_t - theta_tOld, 4), separator=',').replace('\n ', ' \\\n\t') + ')')
                        break
                if (optForAIC or calcAIC):
                    Pi = np.minimum(np.maximum(np.sum(B.swapaxes(1, 3) * theta_t, axis=3), 1e-5), 1 - 1e-5)
                    mat2 = 1 / (Pi * (1 - Pi))
                    fisher = np.sum(np.array([np.dot(B_cbind[l] * np.delete(mat2[l].reshape(self.sortG.N ** 2, ), np.arange(self.sortG.N) * (self.sortG.N + 1)), B_cbind[l].T) for l in range(m)]), 0)
                    try:
                        df_lambda = np.trace(np.dot(np.linalg.inv(fisher + lambda_ * penalize), fisher))
                    except np.linalg.LinAlgError as err:
                        warnings.warn('penalized fisher matrix is singular, shape: ' + fisher.shape.__str__() + ' & rank: ' + np.linalg.matrix_rank(fisher + lambda_ * penalize).__str__() + '; use generalized (Moore-Penrose) pseudo-inverse')
                        print('UserWarning: penalized fisher matrix is singular, shape: ' + fisher.shape.__str__() + ' & rank: ' + np.linalg.matrix_rank(fisher + lambda_ * penalize).__str__() + '; use generalized (Moore-Penrose) pseudo-inverse')
                        df_lambda = np.trace(np.dot(np.linalg.pinv(fisher + lambda_ * penalize), fisher))
                    df_lambda_correct = df_lambda - nSubs ** 2
                    logProbMat = (self.sortG.A * np.log(Pi)) + ((1 - self.sortG.A) * np.log(1 - Pi))
                    [np.fill_diagonal(logProbMat_i, 0) for logProbMat_i in logProbMat]
                    logLik = np.sum([np.nansum(logProbMat_i) for logProbMat_i in logProbMat])
                    AIC = -2 * logLik + 2 * (df_lambda + (nSubs -1))
                    AICc = -2 * logLik + 2 * (df_lambda + (nSubs -1)) + ((2 * (df_lambda + (nSubs -1)) * ((df_lambda + (nSubs -1)) + 1)) / ((N_squ_eff * m) - (df_lambda + (nSubs -1)) - 1))
                    AICL1 = -2 * logLik + N_squ_eff_log * df_lambda + 2 * (nSubs -1)
                    AICL2 = -2 * logLik + 2 * df_lambda + np.log(self.sortG.N) * (nSubs -1)
                    AICL2b = -2 * logLik + 2 * df_lambda_correct + N_squ_eff_log * nSubs ** 2 + np.log(self.sortG.N) * (nSubs -1)
                    ICL = -logLik + (1/2) * (N_squ_eff_log * df_lambda + np.log(self.sortG.N) * (nSubs -1))
                else:
                    df_lambda = None  # !!!
                    df_lambda_correct = None  # !!!
                    logProbMat = None  # !!!
                    logLik = None  # !!!
                    AIC, AICc, AICL1, AICL2, AICL2b, ICL = None, None, None, None, None, None
                return(eval('(' + ('' if onlyAIC else 'Pi, mat2, fisher, logProbMat, logLik, df_lambda, df_lambda_correct, theta_t, AIC, AICc, AICL1, AICL2, AICL2b, ICL, ') + critType + ')'))
            if optForAIC:
                lambdaMin = 0 if (lambdaMin is None) else lambdaMin
                lambdaMax = 10000 if (lambdaMax is None) else lambdaMax
                if (not lambda_ is None):
                    if (not np.isnan(lambda_)):
                        warnings.warn('lambda_ is only used when the estimation is not optimized w.r.t. AIC')
                        print('UserWarning: lambda_ is only used when the estimation is not optimized w.r.t. AIC')
                maxfun = 30
                old_stdout = sys.stdout
                new_stdout = io.StringIO()
                sys.stdout = new_stdout
                lambda_ = optimize.fminbound(func=lambda lambda_: estTheta(lambda_=lambda_, onlyAIC=True), x1=lambdaMin, x2=lambdaMax, xtol=5e-01, maxfun=maxfun, disp=3)
                output_ = new_stdout.getvalue()
                sys.stdout = old_stdout
                AIC_opt_lamb = np.array([str_ii[str_ii != ''][1].astype(float) for str_i in np.array(output_.split('\n'))[np.in1d(np.array([words_i[np.argmax((np.array(words_i) != ''))] for words_i in [row_i.split(' ') for row_i in output_.split('\n')]]), np.arange(1, maxfun + 1).astype(str))] for str_ii in [np.array(str_i.split(' '))]])
                AIC_opt_vals = np.array([str_ii[str_ii != ''][2].astype(float) for str_i in np.array(output_.split('\n'))[np.in1d(np.array([words_i[np.argmax((np.array(words_i) != ''))] for words_i in [row_i.split(' ') for row_i in output_.split('\n')]]), np.arange(1, maxfun + 1).astype(str))] for str_ii in [np.array(str_i.split(' '))]])
                AIC_opt_lamb_list = [[AIC_opt_lamb[np.argsort(AIC_opt_lamb)]]]
                AIC_opt_vals_list = [[AIC_opt_vals[np.argsort(AIC_opt_lamb)]]]
            else:
                AIC_opt_lamb_list = [[]]
                AIC_opt_vals_list = [[]]
            Pi_, mat2_, fisher_, logProbMat_, logLik_, df_lambda_, df_lambda_correct_, theta_t, AIC, AICc, AICL1, AICL2, AICL2b, ICL, critValue = estTheta(lambda_)  # !!!
            P_mat = theta_t.reshape(nSpline1d, nSpline1d)
            theta_sep_ext = None
            P_mat_sep_ext = None
            nodeInd_i_ = None  # !!!
            nodeInd_j_ = None  # !!!
            if not (optForAIC or calcAIC):
                critType = None
        else:
            nSpline1d = nKnots + k - 1
            nSpline = np.dot(nSpline1d[np.newaxis].T, nSpline1d[np.newaxis])
            nSpl1_cum = np.cumsum(np.append([0], nSpline1d))
            N_squ_eff_log = np.log(np.logical_not(np.isnan(self.sortG.A)).sum() - np.logical_not(np.isnan(np.diagonal(self.sortG.A))).sum())
            if (np.isscalar(lambda_) or (lambda_ is None)):
                lambda_ = np.full((nSubs, nSubs), np.nan if (lambda_ is None) else lambda_)
            if lambda_.shape != (nSubs, nSubs):
                raise TypeError('number of lambdas and number of Subs need to correspond')
            P_mat = np.full([np.sum(nSpline1d)]*2, -1.)
            P_mat_sep_ext = [[[] for j in range(nSubs)] for i in range(nSubs)]
            theta_sep_ext = [[[] for j in range(nSubs)] for i in range(nSubs)]
            AIC, AICc, AICL1, AICL2, AICL2b, ICL = 0, 0, 0, 0, 0, 0
            AIC_opt_lamb_list, AIC_opt_vals_list = [], []
            logLik_ = 0  # !!!
            df_lambda_ = 0  # !!!
            df_lambda_correct_ = 0  # !!!
            Pi_ = []  # !!!
            mat2_ = []  # !!!
            fisher_ = []  # !!!
            logProbMat_ = []  # !!!
            nodeInd_i_ = []  # !!!
            nodeInd_j_ = []  # !!!
            warnEmptyGr = np.repeat(False, nSubs)
            warnUselessLamb = False
            for i_ in range(nSubs):
                AIC_opt_lamb_list.append([])
                AIC_opt_vals_list.append([])
                for j_ in range(i_, nSubs):
                    nodeInd_i = np.logical_and(Us_mult >= splitPos[i_], Us_mult < splitPos[i_ + 1])
                    nodeInd_j = np.logical_and(Us_mult >= splitPos[j_], Us_mult < splitPos[j_ + 1])
                    if ((nodeInd_i.sum() * nodeInd_j.sum()) == 0):  # ***
                        theta_t_part = np.repeat(self.sortG.density, nSpline[i_, j_])
                        AIC_part, AICc_part, AICL1_part, AICL2_part, AICL2b_part, ICL_part = 0, 0, 0, 0, 0, 0
                        logLik_part = 0  # !!!
                        df_lambda_part = 0  # !!!
                        df_lambda_correct_part = 0  # !!!
                        if ((nodeInd_i.sum() == 0) and (not warnEmptyGr[i_])):
                            warnings.warn('community ' + (i_+1).__str__() + ' is empty, estimation is based on overall network density')
                            print('UserWarning: community ' + (i_+1).__str__() + ' is empty, estimation is based on overall network density')
                            warnEmptyGr[i_] = True
                        if ((nodeInd_j.sum() == 0) and (not warnEmptyGr[j_])):
                            warnings.warn('community ' + (j_+1).__str__() + ' is empty, estimation is based on overall network density')
                            print('UserWarning: community ' + (j_+1).__str__() + ' is empty, estimation is based on overall network density')
                            warnEmptyGr[j_] = True
                    else:
                        Us_mult_reduc_i = [Us_mult[i][nodeInd_i[i]] for i in range(m)]
                        Us_mult_reduc_j = [Us_mult[j][nodeInd_j[j]] for j in range(m)]
                        A_reduc = [self.sortG.A[nodeInd_i[i]][:, nodeInd_j[i]] for i in range(m)]
                        if k == 1:
                            B = [np.array([
                                (interpolate.bisplev(x=np.sort(Us_mult_reduc_i[j]), y=np.sort(Us_mult_reduc_j[j]), tck=(tau_sep[i_], tau_sep[j_], np.lib.pad([1], (i, nSpline[i_, j_] - i - 1), 'constant', constant_values=(0)), k, k), dx=0, dy=0)
                                 if (len(Us_mult_reduc_i[j]) * len(Us_mult_reduc_j[j]) > 0) else np.array([])).reshape(len(Us_mult_reduc_i[j]), len(Us_mult_reduc_j[j])) for i in np.arange(nSpline[i_, j_])
                            ])[:, np.argsort(np.argsort(Us_mult_reduc_i[j])), :][:, :, np.argsort(np.argsort(Us_mult_reduc_j[j]))] for j in range(m)]
                        else:
                            raise TypeError('B-splines of degree k = ' + k.__str__() + ' have not been implemented yet')
                        N_part_i = nodeInd_i.sum(axis=1)
                        N_part_j = nodeInd_j.sum(axis=1)
                        B_cbind = [np.delete(B[l].reshape(nSpline[i_, j_], N_part_i[l] * N_part_j[l]), (np.arange(N_part_i[l]) * (N_part_i[l] + 1)) if (i_ == j_) else [], axis=1) for l in range(m)]
                        if canonical:
                            raise TypeError('canonical B-splines representation has not been implemented yet')
                        else:
                            A1 = np.vstack((np.identity(nSpline[i_, j_]), -np.identity(nSpline[i_, j_])))
                        if i_ == j_:
                            A2 = np.array([]).reshape((nSpline[i_, j_], 0))
                            for i in range(nSpline1d[i_]):
                                for j in range(i + 1, nSpline1d[i_]):
                                    NullMat = np.zeros((nSpline1d[i_], nSpline1d[i_]))
                                    NullMat[i, j], NullMat[j, i] = 1, -1
                                    A2 = np.hstack((A2, NullMat.reshape(nSpline[i_, j_], 1)))
                            A2 = A2.T
                        L_part_1 = np.identity(nSpline1d[i_])[:-1] - np.hstack((np.zeros((nSpline1d[i_] - 1, 1)), np.identity(nSpline1d[i_] - 1)))
                        L_part_2 = np.identity(nSpline1d[j_])[:-1] - np.hstack((np.zeros((nSpline1d[j_] - 1, 1)), np.identity(nSpline1d[j_] - 1)))
                        I_part_1 = np.identity(nSpline1d[j_])
                        I_part_2 = np.identity(nSpline1d[i_])
                        penalize = np.dot(np.kron(L_part_1, I_part_1).T, np.kron(L_part_1, I_part_1)) + np.dot(np.kron(I_part_2, L_part_2).T, np.kron(I_part_2, L_part_2))
                        G_ = cvxopt.matrix(-A1)
                        A_ = cvxopt.matrix(A2) if (i_ == j_) else None
                        cvxopt.solvers.options['show_progress'] = False
                        def estTheta(lambda_, onlyAIC=False):
                            theta_t_part = np.repeat(np.nansum([np.nanmean(A_reduc[i]) * A_reduc[i].size for i in range(m)]) / np.sum([A_reduc[i].size for i in range(m)]), nSpline[i_, j_])
                            differ = 5
                            index_marker = 1
                            while (differ > 0.01**2):
                                Pi = [np.minimum(np.maximum(np.sum(B[i].swapaxes(0, 2) * theta_t_part, axis=2).swapaxes(0, 1), 1e-5), 1 - 1e-5) for i in range(m)]
                                mat1 = [B[i] * ((A_reduc[i] * (1 / Pi[i])) - ((1 - A_reduc[i]) * (1 / (1 - Pi[i])))) for i in range(m)]
                                score = np.sum([np.sum(np.nansum(mat1[i], axis=1), axis=1) for i in range(m)], axis=0) - (np.sum([np.nansum(np.diagonal(mat1[l], axis1=1, axis2=2).T, axis=0) for l in range(m)], axis=0) if (i_ == j_) else 0)
                                mat2 = [1 / (Pi[i] * (1 - Pi[i])) for i in range(m)]
                                fisher = np.sum(np.array([np.dot(B_cbind[l] * np.delete(mat2[l].reshape(N_part_i[l] * N_part_j[l], ), (np.arange(N_part_i[l]) * (N_part_i[l] + 1)) if (i_ == j_) else []), B_cbind[l].T) for l in range(m)]), 0)
                                P_ = cvxopt.matrix(fisher + lambda_ * penalize)
                                q_ = cvxopt.matrix(-score + lambda_ * np.dot(theta_t_part, penalize))
                                if canonical:
                                    raise TypeError('canonical B-splines representation has not been implemented yet')
                                else:
                                    h_ = cvxopt.matrix(np.dot(A1, theta_t_part) + np.append(np.zeros(nSpline[i_, j_], ), np.ones(nSpline[i_, j_], )))
                                b_ = cvxopt.matrix(np.dot(-A2, theta_t_part)) if (i_ == j_) else None
                                try:  # ***
                                    delta_t = np.squeeze(np.array(cvxopt.solvers.qp(P=P_, q=q_, G=G_, h=h_, A=A_, b=b_)['x']))
                                except ValueError:
                                    warnings.warn('optimizer could not be applied; see\ntheta_t_part = np.array(' + np.array2string(theta_t_part, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                                  '\nP_ = cvxopt.matrix(np.array(' + np.array2string(np.array(P_), separator=',').replace('\n ', ' \\\n\t') + '))' + \
                                                  '\nq_ = cvxopt.matrix(np.array(' + np.array2string(np.array(q_), separator=',').replace('\n ', ' \\\n\t') + '))' + \
                                                  '\nG_ = cvxopt.matrix(np.array(' + np.array2string(np.array(G_), separator=',').replace('\n ', ' \\\n\t') + '))' + \
                                                  '\nh_ = cvxopt.matrix(np.array(' + np.array2string(np.array(h_), separator=',').replace('\n ', ' \\\n\t') + '))' + \
                                                  '\nA_ = cvxopt.matrix(np.array(' + np.array2string(np.array(A_), separator=',').replace('\n ', ' \\\n\t') + '))' + \
                                                  '\nb_ = cvxopt.matrix(np.array(' + np.array2string(np.array(b_), separator=',').replace('\n ', ' \\\n\t') + '))')
                                    print('UserWarning: optimizer could not be applied; see\ntheta_t_part = np.array(' + np.array2string(theta_t_part, separator=', ').replace('\n ', ' \\\n\t') + ')' + \
                                          '\nP_ = cvxopt.matrix(np.array(' + np.array2string(np.array(P_), separator=',').replace('\n ', ' \\\n\t') + '))' + \
                                          '\nq_ = cvxopt.matrix(np.array(' + np.array2string(np.array(q_), separator=',').replace('\n ', ' \\\n\t') + '))' + \
                                          '\nG_ = cvxopt.matrix(np.array(' + np.array2string(np.array(G_), separator=',').replace('\n ', ' \\\n\t') + '))' + \
                                          '\nh_ = cvxopt.matrix(np.array(' + np.array2string(np.array(h_), separator=',').replace('\n ', ' \\\n\t') + '))' + \
                                          '\nA_ = cvxopt.matrix(np.array(' + np.array2string(np.array(A_), separator=',').replace('\n ', ' \\\n\t') + '))' + \
                                          '\nb_ = cvxopt.matrix(np.array(' + np.array2string(np.array(b_), separator=',').replace('\n ', ' \\\n\t') + '))')
                                    delta_t = np.zeros(theta_t_part.shape)
                                theta_t_partOld = copy(theta_t_part)
                                theta_t_part = delta_t + theta_t_part
                                differ = (1 / nSpline[i_, j_]) * np.sum((theta_t_part - theta_t_partOld)**2)
                                print('Iteration of estimating theta:', index_marker)
                                index_marker = index_marker + 1
                                if index_marker > 10:
                                    warnings.warn('Fisher scoring did not converge; see\ntheta_t_partOld = np.array(' + np.array2string(theta_t_partOld, separator=',').replace('\n ', ' \\\n\t') + ')' + \
                                                  '\ntheta_t_part = np.array(' + np.array2string(theta_t_part, separator=',').replace('\n ', ' \\\n\t') + ')' + \
                                                  '\ndelta_t_part = np.array(' + np.array2string(np.round(theta_t_part - theta_t_partOld, 4), separator=',').replace('\n ', ' \\\n\t') + ')')
                                    print('UserWarning: Fisher scoring did not converge; see\ntheta_t_partOld = np.array(' + np.array2string(theta_t_partOld, separator=',').replace('\n ', ' \\\n\t') + ')' + \
                                          '\ntheta_t_part = np.array(' + np.array2string(theta_t_part, separator=',').replace('\n ', ' \\\n\t') + ')' + \
                                          '\ndelta_t_part = np.array(' + np.array2string(np.round(theta_t_part - theta_t_partOld, 4), separator=',').replace('\n ', ' \\\n\t') + ')')
                                    break
                            if (optForAIC or calcAIC):
                                Pi = [np.minimum(np.maximum(np.sum(B[i].swapaxes(0, 2) * theta_t_part, axis=2).swapaxes(0, 1), 1e-5), 1 - 1e-5) for i in range(m)]
                                mat2 = [1 / (Pi[i] * (1 - Pi[i])) for i in range(m)]
                                fisher = np.sum(np.array([np.dot(B_cbind[l] * np.delete(mat2[l].reshape(N_part_i[l] * N_part_j[l], ), (np.arange(N_part_i[l]) * (N_part_i[l] + 1)) if (i_ == j_) else []), B_cbind[l].T) for l in range(m)]), 0)
                                try:
                                    df_lambda = np.trace(np.dot(np.linalg.inv(fisher + lambda_ * penalize), fisher))
                                except np.linalg.LinAlgError as err:
                                    warnings.warn('penalized fisher matrix is singular, shape: ' + fisher.shape.__str__() + ' & rank: ' + np.linalg.matrix_rank(fisher + lambda_ * penalize).__str__() + '; use generalized (Moore-Penrose) pseudo-inverse')
                                    print('UserWarning: penalized fisher matrix is singular, shape: ' + fisher.shape.__str__() + ' & rank: ' + np.linalg.matrix_rank(fisher + lambda_ * penalize).__str__() + '; use generalized (Moore-Penrose) pseudo-inverse')
                                    df_lambda = np.trace(np.dot(np.linalg.pinv(fisher + lambda_ * penalize), fisher))
                                df_lambda_correct = df_lambda -1
                                logProbMat = [(A_reduc[i] * np.log(Pi[i])) + ((1 - A_reduc[i]) * np.log(1 - Pi[i])) for i in range(m)]
                                if i_ == j_:
                                    [np.fill_diagonal(logProbMat_i, 0) for logProbMat_i in logProbMat]
                                logLik = np.sum([np.nansum(logProbMat_i) for logProbMat_i in logProbMat])
                                N_squ_eff_reduc = np.sum([(np.logical_not(np.isnan(A_reduc[l])).sum() - np.logical_not(np.isnan(np.diagonal(A_reduc[l]))).sum()) for l in range(m)])
                                AIC_part = -2 * logLik + 2 * df_lambda
                                AICc_part = -2 * logLik + 2 * df_lambda + ((2 * df_lambda * (df_lambda + 1)) / (N_squ_eff_reduc - df_lambda - 1))  # wrong !!!
                                AICL1_part = -2 * logLik + N_squ_eff_log * df_lambda
                                AICL2_part = -2 * logLik + 2 * df_lambda
                                AICL2b_part = -2 * logLik + 2 * df_lambda_correct
                                ICL_part = -logLik + (1/2) * N_squ_eff_log * df_lambda
                            else:
                                df_lambda = 0  # !!!
                                df_lambda_correct = 0  # !!!
                                logProbMat = None  # !!!
                                logLik = 0  # !!!
                                AIC_part, AICc_part, AICL1_part, AICL2_part, AICL2b_part, ICL_part = 0, 0, 0, 0, 0, 0
                            return(eval('(' + ('' if onlyAIC else 'Pi, mat2, fisher, logProbMat, logLik, df_lambda, df_lambda_correct, theta_t_part, AIC_part, AICc_part, AICL1_part, AICL2_part, AICL2b_part, ICL_part, ') + critType + '_part)'))
                        if optForAIC:
                            lambdaMin = 0 if (lambdaMin is None) else lambdaMin
                            lambdaMax = 10000 if (lambdaMax is None) else lambdaMax
                            if ((not np.isnan(lambda_[i_,j_])) and (not warnUselessLamb)):
                                warnings.warn('lambda_ is only used when the estimation is not optimized w.r.t. AIC')
                                print('UserWarning: lambda_ is only used when the estimation is not optimized w.r.t. AIC')
                                warnUselessLamb = True
                            maxfun=30
                            old_stdout = sys.stdout
                            new_stdout = io.StringIO()
                            sys.stdout = new_stdout
                            lambda_[i_, j_] = optimize.fminbound(func=lambda lambda_: estTheta(lambda_=lambda_, onlyAIC=True), x1=lambdaMin, x2=lambdaMax, xtol=5e-01, maxfun=maxfun, disp=3)
                            output_ = new_stdout.getvalue()
                            sys.stdout = old_stdout
                            AIC_opt_lamb = np.array([str_ii[str_ii != ''][1].astype(float) for str_i in np.array(output_.split('\n'))[np.in1d(np.array([words_i[np.argmax((np.array(words_i) != ''))] for words_i in [row_i.split(' ') for row_i in output_.split('\n')]]), np.arange(1, maxfun + 1).astype(str))] for str_ii in [np.array(str_i.split(' '))]])
                            AIC_opt_vals = np.array([str_ii[str_ii != ''][2].astype(float) for str_i in np.array(output_.split('\n'))[np.in1d(np.array([words_i[np.argmax((np.array(words_i) != ''))] for words_i in [row_i.split(' ') for row_i in output_.split('\n')]]), np.arange(1, maxfun + 1).astype(str))] for str_ii in [np.array(str_i.split(' '))]])
                            AIC_opt_lamb_list[-1].append(AIC_opt_lamb[np.argsort(AIC_opt_lamb)])
                            AIC_opt_vals_list[-1].append(AIC_opt_vals[np.argsort(AIC_opt_lamb)])
                        Pi_part, mat2_part, fisher_part, logProbMat_part, logLik_part, df_lambda_part, df_lambda_correct_part, theta_t_part, AIC_part, AICc_part, AICL1_part, AICL2_part, AICL2b_part, ICL_part, critValue_part = estTheta(lambda_[i_,j_])  # !!!
                        Pi_.append(Pi_part)  # !!!
                        mat2_.append(mat2_part)  # !!!
                        fisher_.append(fisher_part)  # !!!
                        logProbMat_.append(logProbMat_part)  # !!!
                    theta_ext = np.vstack((
                        np.zeros((k + 1, nSpline1d[j_] + 2 * (k + 1))),
                        np.hstack((np.zeros((nSpline1d[i_], k + 1)), theta_t_part.reshape(nSpline1d[i_], nSpline1d[j_]), np.zeros((nSpline1d[i_], k + 1)))),
                        np.zeros((k + 1, nSpline1d[j_] + 2 * (k + 1)))
                    ))
                    P_mat[nSpl1_cum[i_]:nSpl1_cum[i_+1]][:,nSpl1_cum[j_]:nSpl1_cum[j_+1]] = theta_t_part.reshape((nSpline1d[i_], nSpline1d[j_]))
                    P_mat_sep_ext[i_][j_] = theta_ext
                    theta_sep_ext[i_][j_] = theta_ext.reshape((nSpline1d[i_] + 2 * (k + 1)) * (nSpline1d[j_] + 2 * (k + 1)), )
                    if i_ != j_:
                        theta_sep_ext[j_][i_] = theta_ext.T.reshape((nSpline1d[i_] + 2 * (k + 1)) * (nSpline1d[j_] + 2 * (k + 1)), )
                        P_mat[nSpl1_cum[j_]:nSpl1_cum[j_+1]][:,nSpl1_cum[i_]:nSpl1_cum[i_+1]] =  theta_t_part.reshape((nSpline1d[i_], nSpline1d[j_])).T
                    AIC += AIC_part * (1 if (i_ == j_) else 2)
                    AICc += AICc_part * (1 if (i_ == j_) else 2)  # wrong !!!
                    AICL1 += AICL1_part * (1 if (i_ == j_) else 2)
                    AICL2 += AICL2_part * (1 if (i_ == j_) else 2)
                    AICL2b += AICL2b_part * (1 if (i_ == j_) else 2)
                    ICL += ICL_part * (1 if (i_ == j_) else 2)
                    logLik_ += logLik_part * (1 if (i_ == j_) else 2)  # !!!
                    df_lambda_ += df_lambda_part * (1 if (i_ == j_) else 2)  # !!!
                    df_lambda_correct_ += df_lambda_correct_part * (1 if (i_ == j_) else 2)  # !!!
                    nodeInd_i_.append(nodeInd_i)  # !!!
                    nodeInd_j_.append(nodeInd_j)  # !!!
            if (optForAIC or calcAIC):
                AIC += 2 * (nSubs - 1)
                AICc += 2 * (nSubs - 1) + ((2 * (nSubs - 1) * ((nSubs - 1) + 1)) / ((self.sortG.N * m) - (nSubs - 1) - 1))  # wrong !!!
                AICL1 += 2 * (nSubs - 1)
                AICL2 += np.log(self.sortG.N) * (nSubs - 1)
                AICL2b += N_squ_eff_log * nSubs ** 2 + np.log(self.sortG.N) * (nSubs - 1)
                ICL += (1/2) * ((nSubs -1) * np.log(self.sortG.N))
                critValue = eval(critType)
            else:
                AIC, AICc, AICL1, AICL2, AICL2b, ICL, critType, critValue = None, None, None, None, None, None, None, None
            theta_t = P_mat.reshape(nSpl1_cum[-1]**2)
        tau_sep_ext = [np.concatenate((np.repeat(-0.1, k + 1), tau_sep[i], np.repeat(1.1, k + 1))) for i in range(nSubs)]
        tau_new, tau_sep_ext_new = copy(tau), copy(tau_sep_ext)
        tau_new[np.isclose(tau_new, 1.)] = 1 +1e-5
        for i in range(nSubs):
            tau_sep_ext_new[i][np.isclose(tau_sep_ext_new[i], 1.)] = 1 +1e-5
        if k == 1:
            if useOneBasis:
                def fct(x_eval, y_eval):
                    x_eval_order = np.argsort(x_eval)
                    y_eval_order = np.argsort(y_eval)
                    fct_eval_order = interpolate.bisplev(x=np.array(x_eval, ndmin=1, copy=False)[x_eval_order], y=np.array(y_eval, ndmin=1, copy=False)[y_eval_order], tck=(tau_new, tau_new, theta_t, k, k), dx=0, dy=0)
                    return (eval('fct_eval_order' + (('[np.argsort(x_eval_order)]' + ('[:,' if len(y_eval_order) > 1 else '')) if len(x_eval_order) > 1 else ('[' if len(y_eval_order) > 1 else '')) + ('np.argsort(y_eval_order)]' if len(y_eval_order) > 1 else '')))
            else:
                def fct(x_eval, y_eval):
                    x_eval_order = np.argsort(x_eval)
                    y_eval_order = np.argsort(y_eval)
                    fct_eval_order = np.sum([interpolate.bisplev(x=np.array(x_eval, ndmin=1, copy=False)[x_eval_order],
                                                                 y=np.array(y_eval, ndmin=1, copy=False)[y_eval_order],
                                                                 tck=(tau_sep_ext_new[i_], tau_sep_ext_new[j_], theta_sep_ext[i_][j_], k, k), dx=0, dy=0)
                                             for j_ in range(nSubs) for i_ in range(nSubs)], axis=0)
                    return (eval('fct_eval_order' +
                                 (('[np.argsort(x_eval_order)]' + ('[:,' if len(y_eval_order) > 1 else '')) if
                                  (len(x_eval_order) > 1) else ('[' if (len(y_eval_order) > 1) else '')) +
                                 ('np.argsort(y_eval_order)]' if (len(y_eval_order) > 1) else '')))
        self.graphonEst = Graphon(fct=fct)
        self.graphonEst.order = k
        self.graphonEst.nSubs = nSubs
        self.graphonEst.nKnots = nKnots
        self.graphonEst.splitPos = splitPos
        self.graphonEst.tau = tau
        self.graphonEst.tau_sep = tau_sep  # None if useOneBasis else tau_sep
        self.graphonEst.tau_sep_ext = tau_sep_ext  # None if useOneBasis else tau_sep_ext
        self.graphonEst.P_mat = P_mat
        self.graphonEst.P_mat_sep_ext = P_mat_sep_ext
        self.graphonEst.theta = theta_t
        self.graphonEst.theta_sep_ext = theta_sep_ext
        self.graphonEst.lambda_ = lambda_
        self.graphonEst.AIC = AIC
        self.graphonEst.AICc = AICc
        self.graphonEst.AICL1 = AICL1
        self.graphonEst.AICL2 = AICL2
        self.graphonEst.AICL2b = AICL2b
        self.graphonEst.ICL = ICL
        self.graphonEst.critType = critType
        self.graphonEst.critValue = critValue
        self.graphonEst.AIC_opt_lamb_list = AIC_opt_lamb_list
        self.graphonEst.AIC_opt_vals_list = AIC_opt_vals_list
        self.graphonEst.freqUsSub = freqVecSub
        self.graphonEst.freqUsTau = freqVecTau
        self.graphonEst.logLik = logLik_  # !!!
        self.graphonEst.df_lambda = df_lambda_  # !!!
        self.graphonEst.df_lambda_correct = df_lambda_correct_  # !!!
        self.graphonEst.list = [Pi_,mat2_,fisher_,logProbMat_,nodeInd_i_,nodeInd_j_]  # !!!
        return (self.graphonEst)
# out: Estimator Object
#     A = adjacency matrix, Us = U's used for the graphon estimation, N = order of the adj. matrix, degree = degrees (marginal sum of the adj. matrix)
#     GraphonEstBySpline = function for graphon estimation by B-splines

