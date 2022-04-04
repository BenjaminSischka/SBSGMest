'''

Define a graphon class.
@author: Benjamin Sischka

'''
import numpy as np
from scipy import stats
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
from copy import copy
from copy import deepcopy
from matplotlib.colors import LogNorm
import scipy.interpolate as interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable

# auxiliary function to create the color bar for the graphon
def colorBar(mappable, ticks=None):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    if ticks is None:
        return(fig.colorbar(mappable, cax=cax))
    else:
        return(fig.colorbar(mappable, cax=cax, ticks=ticks))
#out: a color bar for heat plots

# auxiliary function to get a matrix as result of a discrete evaluation of the graphon function
# -> piecewise constant approximation of the graphon
def fctToMat(fct,size):
    # fct = specific graphon function, size = fineness of the return matrix
    if np.isscalar(size):
        us_i=us_j=np.linspace(0,1,size)
        size0 = size1 = size
    else:
        us_i=np.linspace(0,1,size[0])
        us_j=np.linspace(0,1,size[1])
        size0 = size[0]
        size1 = size[1]
    try:
        if fct(np.array([0.3,0.7]), np.array([0.3,0.7])).ndim == 1:
            if len(us_i) < len(us_j):
                mat=np.array([fct(us_i[i],us_j) for i in range(size0)])
            else:
                mat=np.array([fct(us_i,us_j[j]) for j in range(size1)])      
        else:
            mat=fct(us_i,us_j)
    except ValueError:
        warnings.warn('not appropriate graphon definition, slow from function to matrix derivation')
        print('UserWarning: not appropriate graphon definition, slow from function to matrix derivation')
        mat=np.zeros((size0, size1))
        for i in range(size0):
            for j in range(size1):
                mat[i,j]=fct(us_i[i],us_j[j])
    return(mat)
#out: squared matrix of the dimension (size,size) 

# auxiliary function to get a piecewise constant graphon function out of a matrix
def matToFct(mat):
    # mat = approx. graphon function on regular grid -> graphon matrix
    def auxFct(u,v):
        if np.isscalar(u):
            return(mat[np.minimum(np.floor(u*mat.shape[0]).astype(int), mat.shape[0]-1)][np.minimum(np.floor(v*mat.shape[1]).astype(int), mat.shape[1]-1)])
        else:
            return(mat[np.minimum(np.floor(u*mat.shape[0]).astype(int), mat.shape[0]-1)][:, np.minimum(np.floor(v*mat.shape[1]).astype(int), mat.shape[1]-1)])
    return(auxFct)
#out: piecewise constant bivariate function

# auxiliary function to get a vectorized version of a specified graphon function
def fctToFct(fct):
    # fct = specific graphon function
    try:
        if fct(np.array([0.3,0.7]), np.array([0.3,0.7])).shape != (2, 2):
            def auxFct(u,v):
                if np.isscalar(u) or np.isscalar(v):
                    return(fct(u,v))
                else:
                    if len(u) < len(v):
                        return(np.array([fct(u_i,v) for u_i in u]))
                    else:
                        return(np.array([fct(u,v_i) for v_i in v]).T)
            return(deepcopy(auxFct)) 
        else:
            return(deepcopy(fct)) 
    except ValueError:
        warnings.warn('function only accepts scalars')
        print('UserWarning: function only accepts scalars')
        def auxFct(u,v):
            if np.isscalar(u) and np.isscalar(v):
                return(fct(u,v))
            elif (not np.isscalar(u)) and np.isscalar(v):
                return(np.array([fct(u_i,v) for u_i in u]))
            elif np.isscalar(u) and (not np.isscalar(v)):
                return(np.array([fct(u,v_i) for v_i in v]))
            else:
                return(np.array([[fct(u_i,v_i) for v_i in v] for u_i in u]))
        return(deepcopy(auxFct)) 
#out: vectorized bivariate function


# Define Graphon Class
class Graphon:
    
    def __init__(self,fct=None,mat=None,size=501):
        # fct = specific graphon function, mat = approx. graphon function on regular grid, size = fineness of the graphon matrix
        if fct is None:
            if mat is None:
                raise TypeError('no informations about the graphon')
            self.mat = copy(np.asarray(mat))
            self.fct = matToFct(self.mat)
            self.byMat = True
        else:
            self.fct = fctToFct(fct)
            self.mat = fctToMat(fct,size)
            self.byMat = False
            if not mat is None:
                if not np.array_equal(np.round(fctToMat(fct,mat.shape), 5),np.round(mat, 5)):
                    warnings.warn('the partitioning of the graphon in a grid \'mat\' is not exactly according to the graphon function \'fct\' or might be rotated')
                    print('UserWarning: the partitioning of the graphon in a grid \'mat\' is not exactly according to the graphon function \'fct\' or might be rotated')
    def showColored(self, vmin=None, vmax=None, vmin_=0.01, log_scale=False, ticks = [0, 0.25, 0.5, 0.75, 1], showColorBar=True, showSplitPos=False, linestyle='--', colorMap = 'plasma_r', fig_ax=None, make_show=True, savefig=False, file_=None):
        if (self.mat.min() < -1e-3) or (self.mat.max() > 1 + 1e-3):
            warnings.warn('graphon has bad values, correction has been applied -> codomain: [0,1]')
            print('UserWarning: graphon has bad values, correction has been applied -> codomain: [0,1]')
        self_mat = np.minimum(np.maximum(self.mat,0),1)
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        if vmin is None:
            vmin = np.nanmin(self_mat)
        vmin_diff = np.max([vmin_ - vmin, 0])
        if vmax is None:
            vmax = np.nanmax(self_mat)
        ## define color for nan values -> light grey
        cmap_ = plt.get_cmap(colorMap)
        cmap_.set_bad(color=plt.get_cmap('binary')(0.2))
        plotGraphon = ax.matshow(self_mat + vmin_diff, cmap=cmap_, interpolation='none', norm=LogNorm(vmin=vmin + vmin_diff, vmax=vmax + vmin_diff)) if log_scale else \
        ax.matshow(self_mat, cmap=cmap_, interpolation='none', vmin=vmin, vmax=vmax)
        plt.xticks(self_mat.shape[1] * np.array(ticks) - 0.5, [(round(round(i,4)) if np.isclose(round(i,4), round(round(i,4))) else round(i,4)).__str__() for i in ticks])
        plt.yticks(self_mat.shape[0] * np.array(ticks) - 0.5, [(round(round(i,4)) if np.isclose(round(i,4), round(round(i,4))) else round(i,4)).__str__() for i in ticks])
        plt.tick_params(bottom=False)
        if showColorBar:
            ticks_CBar = [((10**(np.log10(vmin + vmin_diff) - i * (np.log10(vmin + vmin_diff) - np.log10(vmax + vmin_diff)) / 5)) if log_scale else ((i/5) * (vmax - vmin) + vmin)) for i in range(6)]
            try:  # ***
                cbar = colorBar(plotGraphon, ticks = ticks_CBar)
                cbar.ax.minorticks_off()
                cbar.ax.set_yticklabels(np.round(np.array(ticks_CBar) - (vmin_diff if log_scale else 0), 4))
            except ValueError:
                warnings.warn('plotting color bar not possible')
                print('UserWarning: plotting color bar not possible')
                cbar = None
        if showSplitPos:  # only possible if graphon has been specified as B-spline function - e.g. see byBSpline
            try:
                [ax.axvline(x=x_, color='k', linestyle=linestyle) for x_ in (self.splitPos[1:-1] * self_mat.shape[1] - 0.5)]
                [ax.axhline(y=y_, color='k', linestyle=linestyle) for y_ in (self.splitPos[1:-1] * self_mat.shape[0] - 0.5)]
            except AttributeError:
                warnings.warn('no information about the split positions - graphon has to be specified as hierarchical B-spline function')
                print('UserWarning: no information about the split positions - graphon has to be specified as hierarchical B-spline function')
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(eval('plotGraphon' + (', cbar' if showColorBar else '')))
    def showSlices(self, vmin=None, vmax=None, vmin_=0.01, log_scale=False, showColorBar=True, lineAttr = {'linewidth': 3, 'alpha': 0.7}, colorMap = 'jet', fig_ax=None, make_show=True, savefig=False, file_=None, figsize_=None):
        if (self.mat.min() < 0) or (self.mat.max() > 1):
            warnings.warn('graphon has bad values, correction has been applied -> codomain: [0,1]')
            print('UserWarning: graphon has bad values, correction has been applied -> codomain: [0,1]')
        self_mat = np.minimum(np.maximum(self.mat,0),1)
        if fig_ax is None:
            fig, ax = plt.subplots(figsize = figsize_)
        else:
            fig, ax = fig_ax
        if vmin is None:
            vmin = np.nanmin(self_mat)
        vmin_diff = np.max([vmin_ - vmin, 0])
        if vmax is None:
            vmax = np.nanmax(self_mat)
        us_ = np.maximum(1e-3, np.minimum(1-1e-3, np.linspace(0,1,self_mat.shape[0])))
        vs_ = np.linspace(0,1,self_mat.shape[1])
        if ((len(self.splitPos) > 2) if (hasattr(self, 'splitPos')) else False):
            # consider adjustment in ExtGraph.showNet
            nSubs = len(self.splitPos) -1
            deltaTotal = (.45 - 1/nSubs)**2 - (.45**2 -.4)
            splitPos_ext = np.cumsum(np.append([0], np.concatenate([[prop_i, deltaTotal/(nSubs-1)] for prop_i in np.diff(self.splitPos) * (1-deltaTotal)])[:-1]))
            vals_onColScal = [np.linspace(splitPos_ext[i*2],splitPos_ext[i*2+1],int(np.round((splitPos_ext[i*2+1]-splitPos_ext[i*2])*self_mat.shape[0] *10))) for i in range(nSubs)]
            splitPos_new = np.append([0], np.cumsum([(len(vals_i)/len_total) for len_total in [len(np.concatenate(vals_onColScal))] for vals_i in vals_onColScal]))
            removeInd = np.maximum(np.minimum((np.floor(self.splitPos[1:-1] * self_mat.shape[0]).astype(int) - np.array([[1],[0]])).flatten(), self_mat.shape[0] -1), 0)
            self_mat = np.delete(self_mat, removeInd, axis=0)
            us_ = np.delete(us_, removeInd)
            us_copy = copy(us_)
            for i in range(nSubs):
                us_[np.logical_and(us_copy >= self.splitPos[i], us_copy < self.splitPos[i + 1])] = (us_[np.logical_and(us_copy >= self.splitPos[i], us_copy < self.splitPos[i + 1])] - self.splitPos[i]) * (splitPos_new[i + 1] - splitPos_new[i]) / (self.splitPos[i + 1] - self.splitPos[i]) + splitPos_new[i]
            cmap_vals = plt.get_cmap(colorMap)((np.concatenate(vals_onColScal)))
            cmap = LinearSegmentedColormap.from_list('my_colormap', cmap_vals)
        else:
            cmap = plt.get_cmap(colorMap)
        ax = plt.subplot(111)
        if log_scale:
            self_mat = np.log10(self_mat + vmin_diff)
        plotSlices = [ax.plot(vs_, self_mat[i], color=cmap(us_[i]), linewidth=lineAttr['linewidth'], alpha=lineAttr['alpha']) for i in range(self_mat.shape[0])]
        plt.xlim(np.array([0, 1]))
        if log_scale:
            loc_yticks = np.linspace(np.log10(vmin + vmin_diff), np.log10(vmax + vmin_diff), 6)
            plt.yticks(loc_yticks, np.round(10**loc_yticks - vmin_diff, 4))
            plt.ylim(np.log10(vmin + vmin_diff), np.log10(vmax + vmin_diff))
        else:
            loc_yticks = np.linspace(vmin, vmax, 6)
            plt.yticks(loc_yticks, np.round(loc_yticks, 4))
            plt.ylim(vmin, vmax)
        if showColorBar:
            ax.get_xaxis().set_visible(False)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="4%", pad=0)
            cbar = fig.colorbar(sm, orientation="horizontal", cax=cax)
        if make_show:
            plt.show()
        if savefig:
            plt.tight_layout()
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(eval('plotSlices' + (', cbar' if showColorBar else '')))
    def showExpDegree(self,size=101,norm=False,fmt='-',title=True,showSplitPos=False,make_show=True,savefig=False,file_=None):
        if self.byMat:
            g_ = self.mat.mean(axis=0)
            us = np.linspace(0,1,self.mat.shape[1])
        else:
            g_ = fctToMat(fct=self.fct,size=(10*size,size)).mean(axis=0)
            us = np.linspace(0,1,size)
        if norm:
            plt.ylim((-1/20,21/20))
        plt.xlim((-1/20,21/20))
        plotDegree = plt.plot(us, g_, fmt)
        if showSplitPos:  # only possible if graphon has been specified as B-spline function - e.g. see byBSpline
            try:
                plotSplits = [plt.plot(np.repeat(x_,2), [0,np.max(g_)], color='k', linestyle='--') for x_ in (self.splitPos[1:-1])]
            except AttributeError:
                warnings.warn('no information about the split positions - graphon has to be specified as hierarchical B-spline function')
                print('UserWarning: no information about the split positions - graphon has to be specified as hierarchical B-spline function')
        if title:
            plt.xlabel('u')
            plt.ylabel('g(u)')
        plt.gca().set_aspect(np.abs(np.diff(plt.gca().get_xlim())/np.diff(plt.gca().get_ylim()))[0])
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(eval('plotDegree' + (', plotSplits' if showSplitPos else '')))
#out: Graphon Object
#     fct = graphon function, mat = graphon matrix, byMat = logical whether graphon was specified by function or matrix
#     showColored = plot of the graphon function, showExpDegree = plot of the expected degree profile


# Define graphon generating function by predefined functions
def byExID1(idX,size=101):
    # idX = id of function (see below), size = fineness of the graphon matrix
    examples = {
                1: lambda u,v: 1/2*(u+v),
                2: lambda u,v: ((1-u)*(1-v))**(1/1) * 0.8 + (u*v)**(1/1) * 0.85,
                3: lambda u,v: [eval('interpolate.bisplev(x= np.array(u_, ndmin=1, copy=False)[u_order], y=np.array(v_, ndmin=1, copy=False)[v_order], tck=(tau, tau, np.concatenate((np.tile([0.5,0.5,0.2,0.3,0.3], 2), np.tile([0.2,0.2,0.8,0.1,0.1], 1), np.tile([0.3,0.3,0.1,0.6,0.6], 2))), 2, 2), dx=0, dy=0)' + \
                         (('[np.argsort(u_order)]' + ('[:,' if len(v_order) > 1 else '')) if len(u_order) > 1 else ('[' if len(v_order) > 1 else '')) + \
                         ('np.argsort(v_order)]' if len(v_order) > 1 else '')) for u_ in [u] for v_ in [v] for tau in [np.array([-0.2,-0.1,0,0.45,0.55,1,1.1,1.2])] for v_order in [np.argsort(v)] for u_order in [np.argsort(u)]][0],
                4: lambda u,v: [eval('interpolate.bisplev(x= np.array(u_, ndmin=1, copy=False)[u_order], y=np.array(v_, ndmin=1, copy=False)[v_order], tck=(tau, tau, np.concatenate((np.tile([0.3,0.3,0.4,0.3,0.3], 2), np.tile([0.4,0.4,0.8,0.1,0.1], 1), np.tile([0.3,0.3,0.1,0.6,0.6], 2))), 2, 2), dx=0, dy=0)' + \
                         (('[np.argsort(u_order)]' + ('[:,' if len(v_order) > 1 else '')) if len(u_order) > 1 else ('[' if len(v_order) > 1 else '')) + \
                         ('np.argsort(v_order)]' if len(v_order) > 1 else '')) for u_ in [u] for v_ in [v] for tau in [np.array([-0.2,-0.1,0,0.45,0.55,1,1.1,1.2])] for v_order in [np.argsort(v)] for u_order in [np.argsort(u)]][0],
                5: lambda u,v: [eval('interpolate.bisplev(x= np.array(u_, ndmin=1, copy=False)[u_order], y=np.array(v_, ndmin=1, copy=False)[v_order], tck=(tau, tau, np.concatenate((np.tile([0.2,0.2,0.6,0.3,0.3], 2), np.tile([0.6,0.6,0.8,0.1,0.1], 1), np.tile([0.3,0.3,0.1,0.7,0.7], 2))), 2, 2), dx=0, dy=0)' + \
                         (('[np.argsort(u_order)]' + ('[:,' if len(v_order) > 1 else '')) if len(u_order) > 1 else ('[' if len(v_order) > 1 else '')) + \
                         ('np.argsort(v_order)]' if len(v_order) > 1 else '')) for u_ in [u] for v_ in [v] for tau in [np.array([-0.2,-0.1,0,0.45,0.55,1,1.1,1.2])] for v_order in [np.argsort(v)] for u_order in [np.argsort(u)]][0],
                6: lambda u,v: [eval('interpolate.bisplev(x= np.array(u_, ndmin=1, copy=False)[u_order], y=np.array(v_, ndmin=1, copy=False)[v_order], tck=(tau, tau, np.concatenate((np.tile([0.5,0.5,0.5,0.7,0.7,0.7,0.3,0.3,0.3,0.1,0.1,0.1], 3), np.tile([0.7,0.7,0.7,0.6,0.6,0.6,0.1,0.1,0.1,0.4,0.4,0.4], 3), np.tile([0.3,0.3,0.3,0.1,0.1,0.1,0.7,0.7,0.7,0.5,0.5,0.5], 3), np.tile([0.1,0.1,0.1,0.4,0.4,0.4,0.5,0.5,0.5,0.8,0.8,0.8], 3))), 2, 2), dx=0, dy=0)' + \
                         (('[np.argsort(u_order)]' + ('[:,' if len(v_order) > 1 else '')) if len(u_order) > 1 else ('[' if len(v_order) > 1 else '')) + \
                         ('np.argsort(v_order)]' if len(v_order) > 1 else '')) for u_ in [u] for v_ in [v] for tau in [np.concatenate((np.array([-0.2,-0.1,0]), np.linspace(0,1,11)[1:-1], np.array([1,1.1,1.2])))] for v_order in [np.argsort(v)] for u_order in [np.argsort(u)]][0],
                7: lambda u,v: [eval('interpolate.bisplev(x= np.array(u_, ndmin=1, copy=False)[u_order], y=np.array(v_, ndmin=1, copy=False)[v_order], tck=(tau, tau, np.concatenate((np.tile([0.5,0.5,0.5,0.7,0.7,0.7,0.3,0.3,0.3,0.1,0.1,0.1], 3), np.tile([0.7,0.7,0.7,0.6,0.6,0.6,0.1,0.1,0.1,0.4,0.4,0.4], 3), np.tile([0.3,0.3,0.3,0.1,0.1,0.1,0.7,0.7,0.7,0.5,0.5,0.5], 3), np.tile([0.1,0.1,0.1,0.4,0.4,0.4,0.5,0.5,0.5,0.8,0.8,0.8], 3))), 2, 2), dx=0, dy=0)' + \
                         (('[np.argsort(u_order)]' + ('[:,' if len(v_order) > 1 else '')) if len(u_order) > 1 else ('[' if len(v_order) > 1 else '')) + \
                         ('np.argsort(v_order)]' if len(v_order) > 1 else '')) for u_ in [u] for v_ in [v] for tau in [np.array([-2,-1,0,1,17,33,34,50,66,67,83,99,100,101,102])/100] for v_order in [np.argsort(v)] for u_order in [np.argsort(u)]][0],
                8: lambda u,v: [eval('interpolate.bisplev(x= np.array(u_, ndmin=1, copy=False)[u_order], y=np.array(v_, ndmin=1, copy=False)[v_order], tck=(tau, tau, np.concatenate((np.tile([0.5,0.5,0.7,0.7,0.3,0.3,0.1,0.1], 2), np.tile([0.7,0.7,0.6,0.6,0.1,0.1,0.4,0.4], 2), np.tile([0.3,0.3,0.1,0.1,0.7,0.7,0.5,0.5], 2), np.tile([0.1,0.1,0.4,0.4,0.5,0.5,0.8,0.8], 2))), 2, 2), dx=0, dy=0)' + \
                         (('[np.argsort(u_order)]' + ('[:,' if len(v_order) > 1 else '')) if len(u_order) > 1 else ('[' if len(v_order) > 1 else '')) + \
                         ('np.argsort(v_order)]' if len(v_order) > 1 else '')) for u_ in [u] for v_ in [v] for tau in [np.concatenate((np.array([-0.2,-0.1,0]), np.linspace(0,1,7)[1:-1], np.array([1,1.1,1.2])))] for v_order in [np.argsort(v)] for u_order in [np.argsort(u)]][0],
                9: lambda u,v: [eval('interpolate.bisplev(x= np.array(u_, ndmin=1, copy=False)[u_order], y=np.array(v_, ndmin=1, copy=False)[v_order], tck=(tau, tau, np.concatenate((np.tile([0.5,0.7,0.3,0.1], 1), np.tile([0.7,0.6,0.1,0.4], 1), np.tile([0.3,0.1,0.7,0.5], 1), np.tile([0.1,0.4,0.5,0.8], 1))), 2, 2), dx=0, dy=0)' + \
                         (('[np.argsort(u_order)]' + ('[:,' if len(v_order) > 1 else '')) if len(u_order) > 1 else ('[' if len(v_order) > 1 else '')) + \
                         ('np.argsort(v_order)]' if len(v_order) > 1 else '')) for u_ in [u] for v_ in [v] for tau in [np.concatenate((np.array([-0.1,0]), np.linspace(0,1,5)[1:-1], np.array([1,1.1])))] for v_order in [np.argsort(v)] for u_order in [np.argsort(u)]][0],
                10: lambda u,v: [eval('interpolate.bisplev(x= np.array(u_, ndmin=1, copy=False)[u_order], y=np.array(v_, ndmin=1, copy=False)[v_order], tck=(tau, tau, np.concatenate((np.tile([0.5,0.5,0.5,0.7,0.7,0.7,0.3,0.3,0.3,0.1,0.1,0.1], 3), np.tile([0.7,0.7,0.7,0.6,0.6,0.6,0.1,0.1,0.1,0.4,0.4,0.4], 3), np.tile([0.3,0.3,0.3,0.1,0.1,0.1,0.7,0.7,0.7,0.5,0.5,0.5], 3), np.tile([0.1,0.1,0.1,0.4,0.4,0.4,0.5,0.5,0.5,0.8,0.8,0.8], 3))), 2, 2), dx=0, dy=0)' + \
                         (('[np.argsort(u_order)]' + ('[:,' if len(v_order) > 1 else '')) if len(u_order) > 1 else ('[' if len(v_order) > 1 else '')) + \
                         ('np.argsort(v_order)]' if len(v_order) > 1 else '')) for u_ in [u] for v_ in [v] for tau in [np.array([-2,-1,0,1,17,33,34,50,66,67,83,99,100,101,102])/100] for v_order in [np.argsort(v)] for u_order in [np.argsort(u)]][0],
                11: lambda u,v: [eval('interpolate.bisplev(x= np.array(u_, ndmin=1, copy=False)[u_order], y=np.array(v_, ndmin=1, copy=False)[v_order], tck=(tau, tau, np.concatenate((np.tile([0.5,0.5,0.5,0.7,0.7,0.7,0.4,0.4,0.4,0.1,0.1,0.1], 3), np.tile([0.7,0.7,0.7,0.6,0.6,0.6,0.1,0.1,0.1,0.3,0.3,0.3], 3), np.tile([0.4,0.4,0.4,0.1,0.1,0.1,0.7,0.7,0.7,0.5,0.5,0.5], 3), np.tile([0.1,0.1,0.1,0.3,0.3,0.3,0.5,0.5,0.5,0.8,0.8,0.8], 3))), 2, 2), dx=0, dy=0)' + \
                         (('[np.argsort(u_order)]' + ('[:,' if len(v_order) > 1 else '')) if len(u_order) > 1 else ('[' if len(v_order) > 1 else '')) + \
                         ('np.argsort(v_order)]' if len(v_order) > 1 else '')) for u_ in [u] for v_ in [v] for tau in [np.array([-2,-1,0,1,17,33,34,50,66,67,83,99,100,101,102])/100] for v_order in [np.argsort(v)] for u_order in [np.argsort(u)]][0],
               12: lambda u,v: ((stats.norm(0,0.5).cdf(stats.norm(0,1).ppf(u)) * stats.norm(0,0.5).cdf(stats.norm(0,1).ppf(v))) + ((1 - stats.norm(0,0.5).cdf(stats.norm(0,1).ppf(u))) * (1 - stats.norm(0,0.5).cdf(stats.norm(0,1).ppf(v))))) * 0.5,
               }
    return(Graphon(fct=examples[idX],size=size))
#out: graphon

# Define graphon generating function by predefined functions
def byExID2(idX,size=101):
    # idX = id of function (see below), size = fineness of the graphon matrix
    examples = {
                #continuous functions
                1: {'fct': lambda u,v: u*v},
                2: {'fct': lambda u,v: np.exp(-((1-u)**0.7+(1-v)**0.7))},
                3: {'fct': lambda u,v: 1/4*(u**2+v**2+u**0.5+v**0.5)},
                4: {'fct': lambda u,v: 1/2*(u+v)},
                5: {'fct': lambda u,v: 1/(1+np.exp(-10*(u**2+v**2)))},
                6: {'fct': lambda u,v: np.abs(u-v)},
                602: {'fct': lambda u,v: 1- np.abs(u-v)},
                603: {'fct': lambda u,v: np.exp(-np.abs(u-v)) / (1 + np.exp(-np.abs(u-v)))},
                7: {'fct': lambda u,v: 1/(1+np.exp(-(np.maximum(u,v)**2+np.minimum(u,v)**4)))},
                8: {'fct': lambda u,v: np.exp(-np.maximum((1-u),(1-v))**(3/4))},
                9: {'fct': lambda u,v: np.exp(-1/2*(np.minimum((1-u),(1-v))+(1-u)**0.5+(1-v)**0.5))},
                10: {'fct': lambda u,v: np.log(1+0.5*np.maximum(u,v))},
                11: {'fct': lambda u,v: np.minimum(u,v)*((1/2)*(u+v)) + np.maximum(u,v)*(1-((1/2)*(u+v)))},
                12: {'fct': lambda u,v: ((((u**2)+(v**2))/2)**(1/2))},
                13: {'fct': lambda u,v: (1-((((1-u)**2)+((1-v)**2))/2)**(1/2))*((1/2)*(u+v)) + ((((u**2)+(v**2))/2)**(1/2))*(1-((1/2)*(u+v)))},
                14: {'fct': lambda u,v: ((np.asarray(v) * np.ones(1))+(np.asarray(u) * np.ones(1)).reshape(len((np.asarray(u) * np.ones(1))),1)) * np.logical_and(v < 1/2, (np.asarray(u) * np.ones(1)).reshape(len((np.asarray(u) * np.ones(1))),1) < 1/2) + ((np.asarray(v) * np.ones(1))+(np.asarray(u-1) * np.ones(1)).reshape(len((np.asarray(u) * np.ones(1))),1)) * np.logical_and(v >= 1/2, (np.asarray(u) * np.ones(1)).reshape(len((np.asarray(u) * np.ones(1))),1) >= 1/2)},
                15: {'fct': lambda u,v: ((4/3) * u + (2/3) * v) if (v < 1/2 and u < 1/2) else(0 if ((v < 1/2 and u >= 1/2) or (v >= 1/2 and u < 1/2)) else ((2/3) * u + (4/3) * v - 1))},
                1502: {'fct': lambda u,v: np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: (4/3) * u_[:, np.newaxis] + (2/3) * v_[np.newaxis, :])(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: np.zeros((u_.size,v_.size)))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: np.zeros((u_.size,v_.size)))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: (2/3) * u_[:, np.newaxis] + (4/3) * v_[np.newaxis, :] - 1)(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()]},
                16: {'fct': lambda u,v: (1 / (np.sqrt(2*np.pi) * 0.5)) * np.exp((-1/2) * (((u-0.5)**2 + (v-0.5)**2) / 0.5**2)) * (1/0.9)},
                17: {'fct': lambda u,v: ((np.log(u*v + 1) * (1/np.log(2))) + (((0.5)**(1/8) - (np.abs(u*v - 0.5)**(1/2))) * (1/((0.5)**(1/8)))) + ((u * v)**3) + ((1 / (u+v+0.01)**2) * (1/10000)) + (np.sqrt(1 / (u+v+0.01)) * (1/10))) * (1/5)},
                18: {'fct': lambda u,v: 14* (u*v)**3 - 20.5* (u*v)**2 + 7.5* (u*v)},
                19: {'fct': lambda u,v: (1/ (np.maximum(1- (u**2 + v**2)**(1/2), 0)**5 + np.maximum(1- ((1-u)**2 + v**2)**(1/2), 0)**5 + np.maximum(1- (u**2 + (1-v)**2)**(1/2), 0)**5 + np.maximum(1- ((1-u)**2 + (1-v)**2)**(1/2), 0)**5)) * \
                        (np.maximum(1- (u**2 + v**2)**(1/2), 0)**5 *0.8 + (np.maximum(1- ((1-u)**2 + v**2)**(1/2), 0)**5 + np.maximum(1- (u**2 + (1-v)**2)**(1/2), 0)**5) *0.2 + np.maximum(1- ((1-u)**2 + (1-v)**2)**(1/2), 0)**5 *0.85)},
                20: {'fct': lambda u,v: (1/ (np.maximum(1- (u**2 + v**2)**(1/2), 0)**3 + np.maximum(1- ((1-u)**2 + v**2)**(1/2), 0)**3 + np.maximum(1- (u**2 + (1-v)**2)**(1/2), 0)**3 + np.maximum(1- ((1-u)**2 + (1-v)**2)**(1/2), 0)**3)) * \
                        (np.maximum(1- (u**2 + v**2)**(1/2), 0)**3 *0.8 + (np.maximum(1- ((1-u)**2 + v**2)**(1/2), 0)**3 + np.maximum(1- (u**2 + (1-v)**2)**(1/2), 0)**3) *0.2 + np.maximum(1- ((1-u)**2 + (1-v)**2)**(1/2), 0)**3 *0.85)},
                21: {'fct': lambda u,v: ((1-u)*(1-v))**(1/1) * 0.8 + (u*v)**(1/1) * 0.85},
                22: {'fct': lambda u,v: ((1-u)*(1-v))**(2) * 0.8 + (u*v)**(2) * 0.9},
                23: {'fct': lambda u,v: 1 - ((u-1/2)*(v-1/2)*4)**2},
                24: {'fct': lambda u,v: 1 - np.sqrt(((u-1/2)**2 + (v-1/2)**2) * 2)},
                25: {'fct': lambda u,v: 1 - np.sqrt((u-0.8)**2 + (v-0.8)**2) * np.sqrt(2* 0.8**2)},
                26: {'fct': lambda u,v: (3 - np.sqrt((u-0.3)**2 + (v-0.3)**2)**(1/3) - np.sqrt((u-0.75)**2 + (v-0.1)**2)**(1/3) - np.sqrt((u-0.1)**2 + (v-0.75)**2)**(1/3)) / (3 - np.sqrt((.3-0.75)**2 + (.3-0.1)**2)**(1/3) - np.sqrt((.3-0.1)**2 + (.3-0.75)**2)**(1/3))},
                27: {'fct': lambda u,v: (3 - np.sqrt((u-0.3)**2 + (v-0.3)**2)**(1/10) - np.sqrt((u-0.75)**2 + (v-0.1)**2)**(1/10) - np.sqrt((u-0.1)**2 + (v-0.75)**2)**(1/10)) / (3 - np.sqrt((.3-0.75)**2 + (.3-0.1)**2)**(1/10) - np.sqrt((.3-0.1)**2 + (.3-0.75)**2)**(1/10))},
                28: {'fct': lambda u,v: ((1 - np.sqrt(((u-0.5)**2 + (v-0.5)**2) * 2)) * np.sqrt((u+v)/2)) + ((1 - ((np.sqrt(np.sqrt((0.5-u)**2)) + np.sqrt(np.sqrt((0.5-v)**2)))**2) / 2) * (1 - np.sqrt((u+v)/2)))},
                2801: {'fct': lambda u,v: (1 - np.sqrt(((u-0.5)**2 + (v-0.5)**2) * 2))},
                2802: {'fct': lambda u,v: (1 - ((np.sqrt(np.sqrt((0.5-u)**2)) + np.sqrt(np.sqrt((0.5-v)**2)))**2) / 2)},
                29: {'fct': lambda u,v: 0.4*(1-u)*(1-v) + 0.4 * (u*v) + 0.2 * (u * (1-v) + (1-u) * v) + 0.5 * (1 - 2 * ((u-0.5)**2 + (v-0.5)**2))},
                30: {'fct': lambda u,v: 0.4*(1-u)*(1-v) + 0.4 * (u*v) + 0.2 * (u * (1-v) + (1-u) * v) + 0.1 * ((np.exp(-np.sqrt((u-0.5)**2 + (v-0.5)**2) +np.sqrt(0.5))-1) / (np.exp(np.sqrt(0.5))-1))},
                31: {'fct': lambda u,v: [eval('interpolate.bisplev(x= np.array(u_, ndmin=1, copy=False)[u_order], y=np.array(v_, ndmin=1, copy=False)[v_order], tck=(tau, tau, np.concatenate((np.tile([0.5,0.5,0.2,0.3,0.3], 2), np.tile([0.2,0.2,0.8,0.1,0.1], 1), np.tile([0.3,0.3,0.1,0.6,0.6], 2))), 2, 2), dx=0, dy=0)' + \
                         (('[np.argsort(u_order)]' + ('[:,' if len(v_order) > 1 else '')) if len(u_order) > 1 else ('[' if len(v_order) > 1 else '')) + \
                         ('np.argsort(v_order)]' if len(v_order) > 1 else '')) for u_ in [u] for v_ in [v] for tau in [np.array([-0.2,-0.1,0,0.45,0.55,1,1.1,1.2])] for v_order in [np.argsort(v)] for u_order in [np.argsort(u)]][0]},
                #from Y. Zhang 2016
                51: {'fct': lambda u,v: np.sin(5 * np.pi *(u+v-1) +1) /2 + 0.5},
                52: {'fct': lambda u,v: 1 -(1 + np.exp(15 *(0.8 * np.abs(u-v))**(4/5) -0.1))**(-1)},
                53: {'fct': lambda u,v: (u**2 + v**2)/3 * np.cos(1 / (u**2 + v**2)) + 0.15},
                #discrete functions
                101: {'fct': lambda u,v: (0.2 if u < 2/3 else 0.8) if v < 2/3 else (0.8 if u < 2/3 else 0.2), 'splitPos': np.array([0,2/3,1])},
                102: {'fct': lambda u,v: (0.8 if u < 1/2 else 0.2) if v < 1/2 else (0.2 if u < 1/2 else 0.8), 'splitPos': np.array([0,.5,1])},
                103: {'fct': lambda u,v: (0.6 if u < 1/2 else 0.2) if v < 1/2 else (0.3 if u < 1/2 else 0.8), 'splitPos': np.array([0,.5,1])},
                104: {'fct': lambda u,v: (0.8 if u < 1/4 or 2/4 < u < 3/4 else 0.2) if v < 1/4 or 2/4 < v < 3/4 else (0.2 if u < 1/4 or 2/4 < u < 3/4 else 0.8), 'splitPos': np.array([0,1/4,2/4,3/4,1])},
                105: {'fct': lambda u,v: (0.8 if u < 1/3 or 2/3 < u else 0.2) if v < 1/3 or 2/3 < v else (0.2 if u < 1/3 or 2/3 < u else 0.8), 'splitPos': np.array([0,1/3,2/3,1])},
                106: {'fct': lambda u,v: (0.8 if u < 1/4 or 3/4 < u else 0.2) if v < 1/4 or 3/4 < v else (0.2 if u < 1/4 or 3/4 < u else 0.8), 'splitPos': np.array([0,1/4,3/4,1])},
                107: {'fct': lambda u,v: 1 if (u>0.7) and (v>0.7) else (0.6 if (u>0.4) and (v>0.4) else 0), 'splitPos': np.array([0,.4,.7,1])},
                108: {'fct': lambda u,v: (0.55 if u < 1/2 else 0.3) if v < 1/2 else (0.3 if u < 1/2 else 0.55), 'splitPos': np.array([0,1/2,1])},
                109: {'fct': lambda u,v: (0.4 if u < 1/2 else (0.6 if u < 2/3 else 0.9)) if v < 1/2 else ((0.6 if u < 1/2 else \
                        (0.4 if u < 2/3 else 0.7)) if v < 2/3 else (0.9 if u < 1/2 else (0.7 if u < 2/3 else 0.1))), 'splitPos': np.array([0,1/2,2/3,1])},
                #mixture models
                201: {'fct': lambda u, v: ((np.sqrt(u * v) * 2) if (u <= 0.5 and v <= 0.5) else ((v * 0.3) if (u > 0.5 and v <= 0.5) else ((u * 0.3) if (u <= 0.5 and v > 0.5) else (np.sqrt((u - 1 / 2) * (v - 1 / 2)) * 2)))), 'splitPos': np.array([0,1/2,1])},
                202: {'fct': lambda u, v: ((np.sqrt(u * v) * 1) if (u <= 0.5 and v <= 0.5) else ((((u - 0.5) * v) * 1) if (u > 0.5 and v <= 0.5) else ((((v - 0.5) * u) * 1) if (u <= 0.5 and v > 0.5) else (np.sqrt((u - 1 / 2) * (v - 1 / 2)) * 1)))), 'splitPos': np.array([0,.5,1])},
                203: {'fct': lambda u,v: np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: (u_+0.5)*(v_+0.5))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: (u_+0.5)*(v_-0.5))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: (u_-0.5)*(v_+0.5))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: (u_-0.5)*(v_-0.5))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()], 'splitPos': np.array([0,.5,1])},
                204: {'fct': lambda u,v: np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: .3 * 4 * np.dot(u_[:, np.newaxis], v_[np.newaxis, :]) + .3 * 4 * np.dot((.5-u_[:, np.newaxis]), (.5-v_[np.newaxis, :])))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: .2 * 4 * np.dot(u_[:, np.newaxis], (v_[np.newaxis, :]-.5)))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: .2 * 4 * np.dot((u_[:, np.newaxis]-.5), v_[np.newaxis, :]))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: .3 * 4 * np.dot((u_[:, np.newaxis]-.5), (v_[np.newaxis, :]-.5)) + .3 * 4 * np.dot((1-u_[:, np.newaxis]), (1-v_[np.newaxis, :])))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()], 'splitPos': np.array([0,.5,1])},
                2042: {'fct': lambda u,v: np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: .12 * 4 * np.dot(u_[:, np.newaxis], v_[np.newaxis, :]) + .12 * 4 * np.dot((.5-u_[:, np.newaxis]), (.5-v_[np.newaxis, :])))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: .02 * 4 * np.dot(u_[:, np.newaxis], (v_[np.newaxis, :]-.5)))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: .02 * 4 * np.dot((u_[:, np.newaxis]-.5), v_[np.newaxis, :]))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: .15 * 4 * np.dot((u_[:, np.newaxis]-.5), (v_[np.newaxis, :]-.5)) + .15 * 4 * np.dot((1-u_[:, np.newaxis]), (1-v_[np.newaxis, :])))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()], 'splitPos': np.array([0,.5,1])},
                205: {'fct': lambda u,v: np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.25), fctToFct(lambda u_,v_: .35 * (1/.25)**2 * np.dot(u_[:, np.newaxis], v_[np.newaxis, :]) + .35 * (1/.25)**2 * np.dot((.25-u_[:, np.newaxis]), (.25-v_[np.newaxis, :])))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.25)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.25), fctToFct(lambda u_,v_: .2 * 4 * np.dot(u_[:, np.newaxis], (v_[np.newaxis, :]-.25)))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.logical_and(np.atleast_1d(v) > 0.25, np.atleast_1d(v) <= 0.65))))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.25), fctToFct(lambda u_,v_: .2 * 2 * np.dot(u_[:, np.newaxis], (v_[np.newaxis, :]-.65)))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.65)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.logical_and(np.atleast_1d(u) > 0.25, np.atleast_1d(u) <= 0.65)), fctToFct(lambda u_,v_: .2 * 4 * np.dot((u_[:, np.newaxis]-.25), v_[np.newaxis, :]))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.25)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.logical_and(np.atleast_1d(u) > 0.25, np.atleast_1d(u) <= 0.65)), fctToFct(lambda u_,v_: .2 * (1/.4)**2 * np.dot((u_[:, np.newaxis]-.25), (v_[np.newaxis, :]-.25)) + .2 * (1/.4)**2 * np.dot((.65-u_[:, np.newaxis]), (.65-v_[np.newaxis, :])))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.logical_and(np.atleast_1d(v) > 0.25, np.atleast_1d(v) <= 0.65))))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.logical_and(np.atleast_1d(u) > 0.25, np.atleast_1d(u) <= 0.65)), fctToFct(lambda u_,v_: .2 * 3 * np.dot((u_[:, np.newaxis]-.25), (v_[np.newaxis, :]-.65)))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.65)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.65), fctToFct(lambda u_,v_: .2 * 2 * np.dot((u_[:, np.newaxis]-.65), v_[np.newaxis, :]))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.25)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.65), fctToFct(lambda u_,v_: .2 * 3 * np.dot((u_[:, np.newaxis]-.65), (v_[np.newaxis, :]-.25)))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.logical_and(np.atleast_1d(v) > 0.25, np.atleast_1d(v) <= 0.65))))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.65), fctToFct(lambda u_,v_: .3 * (1/.35)**2 * np.dot((u_[:, np.newaxis]-.65), (v_[np.newaxis, :]-.65)) + .3 * (1/.35)**2 * np.dot((1-u_[:, np.newaxis]), (1-v_[np.newaxis, :])))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.65)))[()], 'splitPos': np.array([0,.25,.65,1])},
                206: {'fct': lambda u,v: np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: [it_ if (it_.size > 1) else np.asscalar(it_) for it_ in [np.squeeze(np.repeat([np.repeat(.2, v_.size)], u_.size, axis = 0))]][0])(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: 2.4 * [it_ if (it_.size > 1) else np.asscalar(it_) for it_ in [np.squeeze(np.array([((v_-.5) * ui) for ui in u_]))]][0] + .2)(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: 2.4 * [it_ if (it_.size > 1) else np.asscalar(it_) for it_ in [np.squeeze(np.array([(v_ * ui) for ui in (u_-.5)]))]][0] + .2)(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                 np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: [it_ if (it_.size > 1) else np.asscalar(it_) for it_ in [np.squeeze(np.repeat([np.repeat(.8, v_.size)], u_.size, axis = 0))]][0])(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()], 'splitPos': np.array([0,.5,1])},
                2062: {'fct': lambda u,v: np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: [it_ if (it_.size > 1) else np.asscalar(it_) for it_ in [np.squeeze(np.repeat([np.repeat(.2, v_.size)], u_.size, axis = 0))]][0])(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: .6 * [it_ if (it_.size > 1) else np.asscalar(it_) for it_ in [np.squeeze(np.array([((v_-.5) + ui) for ui in u_]))]][0] + .2)(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: .6 * [it_ if (it_.size > 1) else np.asscalar(it_) for it_ in [np.squeeze(np.array([(v_ + ui) for ui in (u_-.5)]))]][0] + .2)(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: [it_ if (it_.size > 1) else np.asscalar(it_) for it_ in [np.squeeze(np.repeat([np.repeat(.8, v_.size)], u_.size, axis = 0))]][0])(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()], 'splitPos': np.array([0,.5,1])},
                207: {'fct': lambda u,v: np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: np.dot(u_[:, np.newaxis], v_[np.newaxis, :]))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: np.dot(u_[:, np.newaxis], v_[np.newaxis, :]-.5))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: np.dot(u_[:, np.newaxis]-.5, v_[np.newaxis, :]))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: np.dot(u_[:, np.newaxis]-.5, v_[np.newaxis, :]-.5))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()], 'splitPos': np.array([0,.5,1])},
                2072: {'fct': lambda u,v: np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: np.dot(u_[:, np.newaxis], v_[np.newaxis, :]))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: .15*(u_[:, np.newaxis] + v_[np.newaxis, :] -.5))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: .15*(u_[:, np.newaxis] + v_[np.newaxis, :] -.5))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: np.dot(u_[:, np.newaxis]-.5, v_[np.newaxis, :]-.5))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()], 'splitPos': np.array([0,.5,1])},
                208: {'fct': lambda u,v: np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: np.dot(u_[:, np.newaxis], v_[np.newaxis, :]))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: np.dot(.5-u_[:, np.newaxis], .5-(v_[np.newaxis, :]-.5)))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: np.dot(.5-(u_[:, np.newaxis]-.5), .5-v_[np.newaxis, :]))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: np.dot(u_[:, np.newaxis]-.5, v_[np.newaxis, :]-.5))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()], 'splitPos': np.array([0,.5,1])},
                2082: {'fct': lambda u,v: np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: np.dot(u_[:, np.newaxis], v_[np.newaxis, :]))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) <= 0.5), fctToFct(lambda u_,v_: .15*((.5-u_[:, np.newaxis]) + (.5-(v_[np.newaxis, :]-.5))))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: .15*((.5-(u_[:, np.newaxis]-.5)) + (.5-v_[np.newaxis, :])))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) <= 0.5)))[()] + \
                                  np.squeeze(np.dot(np.dot(np.diag(np.atleast_1d(u) > 0.5), fctToFct(lambda u_,v_: np.dot(u_[:, np.newaxis]-.5, v_[np.newaxis, :]-.5))(np.atleast_1d(u),np.atleast_1d(v))), np.diag(np.atleast_1d(v) > 0.5)))[()], 'splitPos': np.array([0,.5,1])},
               }
    attr_ = examples[idX]
    GraphonSpeci = Graphon(fct=attr_['fct'], size=size)
    del attr_['fct']
    for ky_i in attr_.keys():
        exec('GraphonSpeci.' + ky_i + ' = attr_[\'' + ky_i + '\']')
    return(GraphonSpeci)
#out: graphon

# Define graphon generating function by predefined structures
def byExID3(idX,size=101):
    # idX = id of structure (see below), size = fineness of the graphon matrix
    structure = {
                 1: {'u_lim': np.array([0,0.5,1]), 'wMat': np.array([[0.8,0.2],[0.2,0.8]])},
                 2: {'u_lim': np.array([0,0.5,2/3,1]), 'wMat': np.array([[0.4,0.6,0.9],[0.6,0.4,0.7],[0.9,0.7,0.1]])},
                 3: {'u_lim': np.array([0,0.25,0.75,1]), 'wMat': np.array([[0.65,0.3,0.65],[0.1,0.9,0.1],[0.65,0.3,0.65]])},
                 4: {'u_lim': np.array([0,0.5,1]), 'wMat': np.array([[0.65,0.3],[0.1,0.9]])},
                 5: {'u_lim': np.array([0,0.7,1]), 'wMat': np.array([[0.65,0.3],[0.1,0.9]])},
                 6: {'u_lim': np.array([3/12,5/12,7/12,9/12,12/12]), 'wMat': np.array([[0.4,0.9,0.6,0.9,0.4],[0.9,0.1,0.7,0.1,0.9],[0.6,0.7,0.4,0.7,0.6],[0.9,0.1,0.7,0.1,0.9],[0.4,0.9,0.6,0.9,0.4]])},
                 7: {'u_lim': np.array([0,0.7,1]), 'wMat': np.array([[0.8,0.2],[0.2,0.8]])},
                 8: {'u_lim': np.array([0,0.25,0.75,1]), 'wMat': np.array([[0.8,0.2,0.8],[0.2,0.8,0.2],[0.8,0.2,0.8]])},
                 9: {'u_lim': np.array([0,0.5,1]), 'wMat': np.array([[1,0],[0,1]])},
                 10: {'u_lim': np.array([0,0.25,0.5,0.75,1]), 'wMat': np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])},
                 11: {'u_lim': np.array([0,3/12,7/12,1]), 'wMat': np.array([[0.5,0.1,0.2],[0.1,0.7,0.05],[0.2,0.05,0.9]])},
                 12: {'u_lim': np.array([0,0.5,1]), 'wMat': np.array([[0.6,0.2],[0.2,0.9]])},
                 13: {'u_lim': np.array([0,0.7,1]), 'wMat': np.array([[0.6,0.2],[0.2,0.9]])},
                 14: {'u_lim': np.array([0,0.5,1]), 'wMat': np.array([[0.9,0.2],[0.2,0.9]])},
                 15: {'u_lim': np.array([0,1/3,2/3,1]), 'wMat': np.array([[0.9,0.1,0.2],[0.1,0.8,0.3],[0.2,0.3,0.7]])},
                 1502: {'u_lim': np.array([0,3/12,8/12,12/12]), 'wMat': np.array([[0.9,0.1,0.2],[0.1,0.8,0.3],[0.2,0.3,0.7]])},
                 16: {'u_lim': np.array([0,0.5,0.75,0.875,1]), 'wMat': np.diag([0.3]*4) + 0.01*(np.ones((4,4)) - np.identity(4))},
                 17: {'u_lim': np.array([0,0.5,0.75,0.875,1]), 'wMat': np.diag([0.3]*4) + 0.08*(np.ones((4,4)) - np.identity(4))},
                 18: {'u_lim': np.array([0,0.5,0.75,0.875,1]), 'wMat': np.diag([0.5]*4) + 0.08*(np.ones((4,4)) - np.identity(4))},
                 19: {'u_lim': np.array([0,0.5,1]), 'wMat': np.array([[0.6,0.48],[0.48,0.6]])},
                 20: {'u_lim': np.array([0,1]), 'wMat': np.array([[0.5]])},
                 21: {'u_lim': np.array([0,1/4,1/2,1]), 'wMat': np.array([[0.6,0.1,0.3],[0.1,0.5,0.2],[0.3,0.2,0.4]])},
                 22: {'u_lim': np.array([0,1/4,1/2,1]), 'wMat': np.array([[0.9,0.1,0.1],[0.1,0.9,0.1],[0.1,0.1,0.9]])},
                 23: {'u_lim': np.array([0,0.65,1]), 'wMat': np.array([[0.6,0.2],[0.2,0.8]])},
                 2301: {'u_lim': np.array([0,0.1,1]), 'wMat': np.array([[0.6,0.2],[0.2,0.8]])},
                 2302: {'u_lim': np.array([0,0.5,1]), 'wMat': np.array([[0.6,0.2],[0.2,0.8]])},
                 24: {'u_lim': np.array([0,.25,0.5,.75,1]), 'wMat': np.array([[.2,.2,.2,.2],[.2,.2,.2,.8],[.2,.2,.8,.8],[.2,.8,.8,.8]])},
                 2402: {'u_lim': np.array([0,.25,0.5,.75,1]), 'wMat': np.array([[.2,.2,.2,.5],[.2,.2,.5,.8],[.2,.5,.8,.8],[.5,.8,.8,.8]])},
                 50: {'u_lim': np.array([np.sqrt(i/10) for i in range(11)]), 'wMat': np.array([[(i+j)/(2*10) for j in range(1,11)] for i in range(1,11)])},
                 51: {'u_lim': np.linspace(0,1,11), 'wMat': np.array([[(i+j)/(2*10) for j in range(1,11)] for i in range(1,11)])},
                }
    u_lim = structure[idX]['u_lim']
    wMat = structure[idX]['wMat']
    GraphonSpeci = Graphon(fct=lambda u,v: np.squeeze(np.atleast_2d(wMat[np.searchsorted(u_lim, u) - 1, :])[:, np.searchsorted(u_lim, v) - 1])*1,size=size)
    GraphonSpeci.u_lim = u_lim
    GraphonSpeci.wMat = wMat
    GraphonSpeci.n_groups = len(u_lim) - 1
    return(GraphonSpeci)
#out: graphon

# Define graphon by B-spline function
def byBSpline(tau, P_mat=None, theta=None, order=1, size=101):
    # tau = inner knot positions, P_mat/theta = parameters in form of matrix/vector, order = order of the B-splines
    if order == 0:
        if P_mat is None:
            if theta is None:
                raise ValueError('no information about the graphon values')
            nSpline1d = len(tau) -1
            P_mat = theta.reshape((nSpline1d, nSpline1d))
        else:
            if not theta is None:
                warnings.warn('parameter vector theta has not been used')
                print('UserWarning: parameter vector theta has not been used')
            theta = P_mat.reshape(np.prod(P_mat.shape))
        def grFct(x_eval, y_eval):
            vec_x = np.maximum(np.searchsorted(tau, np.array(x_eval, ndmin=1, copy=False)) -1, 0).astype(int)
            vec_y = np.maximum(np.searchsorted(tau, np.array(y_eval, ndmin=1, copy=False)) -1, 0).astype(int)
            return(P_mat[vec_x][:,vec_y])
    else:
        if theta is None:
            if P_mat is None:
                raise ValueError('no information about the graphon values')
            theta = P_mat.reshape(np.prod(P_mat.shape))
        else:
            if not P_mat is None:
                warnings.warn('parameter matrix P_mat has not been used')
                print('UserWarning: parameter matrix P_mat has not been used')
            P_mat = theta.reshape((len(tau) -1, len(tau) -1))
        def grFct(x_eval, y_eval):
            x_eval_order = np.argsort(x_eval)
            y_eval_order = np.argsort(y_eval)
            fct_eval_order=interpolate.bisplev(x= np.array(x_eval, ndmin=1, copy=False)[x_eval_order], y=np.array(y_eval, ndmin=1, copy=False)[y_eval_order], tck=(tau, tau, theta, order, order), dx=0, dy=0)
            return(eval('fct_eval_order' + (('[np.argsort(x_eval_order)]' + ('[:,' if len(y_eval_order) > 1 else '')) if len(x_eval_order) > 1 else ('[' if len(y_eval_order) > 1 else '')) + ('np.argsort(y_eval_order)]' if len(y_eval_order) > 1 else '')))
    GraphonSpeci = Graphon(fct=grFct,size=size)
    GraphonSpeci.tau = tau
    GraphonSpeci.P_mat = P_mat
    GraphonSpeci.theta = theta
    GraphonSpeci.order = order
    return(GraphonSpeci)
#out: graphon

# Define graphon by multiple B-spline functions
def byBSpline_mult(tau_ext_list=None, tau_list=None, P_mat_ext_list=None, P_mat_list=None, order=1, size=101):
    # tau_(ext)_list = inner knot positions (extended or not), P_mat_(ext)_list = parameters in matrix form (extended or not), order = order of the B-splines
    if tau_ext_list is None:
        tau_ext_list = [np.concatenate((np.repeat(-0.1, 2), tau_list[i], np.repeat(1.1, 2))) for i in range(len(tau_list))]
    else:
        if not (tau_list is None):
            warnings.warn('only tau_ext_list is used, tau_list is neglected')
            print('UserWarning: only tau_ext_list is used, tau_list is neglected')
    nSubs = len(tau_ext_list)
    if P_mat_ext_list is None:
        P_mat_ext_list = [[[np.vstack((
            np.zeros((order +1, P_mat_ij.shape[1] + 2* (order +1))),
            np.hstack((np.zeros((P_mat_ij.shape[0], order +1)), P_mat_ij, np.zeros((P_mat_ij.shape[0], order +1)))),
            np.zeros((order +1, P_mat_ij.shape[1] + 2* (order +1)))
        )) for P_mat_ij in [P_mat_list[i][j]]][0] for j in range(nSubs)] for i in range(nSubs)]
    else:
        if not (P_mat_list is None):
            warnings.warn('only P_mat_ext_list is used, P_mat_list is neglected')
            print('UserWarning: only P_mat_ext_list is used, P_mat_list is neglected')
    def grFct(x_eval, y_eval):
        x_eval_order = np.argsort(x_eval)
        y_eval_order = np.argsort(y_eval)
        fct_eval_order = np.sum([interpolate.bisplev(x=np.array(x_eval, ndmin=1, copy=False)[x_eval_order],
                                                     y=np.array(y_eval, ndmin=1, copy=False)[y_eval_order],
                                                     tck=(tau_ext_list[i_], tau_ext_list[j_], P_mat_ext_list[i_][j_].reshape(np.prod(P_mat_ext_list[i_][j_].shape)), order, order), dx=0, dy=0)
                                 for j_ in range(nSubs) for i_ in range(nSubs)], axis=0)
        return (eval('fct_eval_order' +
                     (('[np.argsort(x_eval_order)]' + ('[:,' if len(y_eval_order) > 1 else '')) if
                      (len(x_eval_order) > 1) else ('[' if (len(y_eval_order) > 1) else '')) +
                     ('np.argsort(y_eval_order)]' if (len(y_eval_order) > 1) else '')))
    GraphonSpeci = Graphon(fct=grFct,size=size)
    GraphonSpeci.tau_ext_list = tau_ext_list
    GraphonSpeci.tau_list = tau_list
    GraphonSpeci.P_mat_ext_list = P_mat_ext_list
    GraphonSpeci.P_mat_list = P_mat_list
    GraphonSpeci.nSubs = nSubs
    GraphonSpeci.order = order
    return(GraphonSpeci)
#out: graphon

# Define graphon generating function by predefined structures
def byExID4(idX,size=101):
    # idX = id of structure (see below), size = fineness of the graphon matrix
    structure = {
                 1: {'tau_list': [np.array([0, 0, 0.1, 0.2, 0.2]), np.array([0.2, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0])],
                     'P_mat_list': [[np.repeat(0.4, 9).reshape(3, 3), np.repeat(0.15, 15).reshape(3, 5)], [np.repeat(0.15, 15).reshape(5, 3), np.repeat(0.6, 25).reshape(5, 5)]]},
    }
    GraphonSpeci = byBSpline_mult(tau_list=structure[idX]['tau_list'], P_mat_list=structure[idX]['P_mat_list'],size=size)
    return(GraphonSpeci)
#out: graphon

# Define discrete graphon by random plateaus
def randomGraphon(size=5, equDeg=False):
    # size = number of blocks concerning stochastic block models, equDeg = if degree profile should be equal/constant
    mat=np.random.uniform(0,1,(size,size))
    mat[np.tril_indices(size)]=mat.T[np.tril_indices(size)]
    if equDeg:
        indexVal = 0
        multFact_vec = np.zeros(size)
        while(np.mean((multFact_vec - 1)**2) > 1e-3**2):
            Deg_target = np.mean(mat)
            for i in range(size):
                multFact = Deg_target / np.mean(mat[i])
                multFact_vec[i] = multFact
                multFact = np.min([multFact, 1/np.max(mat[i])])
                mat[i] = mat[i] * multFact
                mat[:,i] = mat[:,i] * multFact
            indexVal += 1
            if indexVal % 10 == 0:
                print('iteration for constancy: ' + indexVal.__str__())
            if indexVal > 100:
                warnings.warn('equalizing degree profile was not successful')
                print('UserWarning: equalizing degree profile was not successful')
                break
    order=[tupl[0] for tupl in sorted(enumerate(mat.sum(axis=0)), key=itemgetter(1))]
    mat=mat[order][:,order]
    return(Graphon(mat=mat))
#out: graphon

# Sort graphon by degree profile
def sortGraphon(graphon,size=1001,segmts=None):
    # graphon = graphon, size = preciseness of information extracting from graphon in, segmts = segments concerning the sorting, if larger than size -> segments decompose into single lines
    mat = graphon.mat if graphon.byMat else fctToMat(graphon.fct,size=size)
    if segmts is None:
        segmts = mat.shape[0]
    limits = np.round(mat.shape[0] * np.arange(segmts+1)/segmts).astype(int)
    intvals = np.array([np.arange(limits[i], limits[i+1]) for i in np.arange(segmts)])
    orderRow = np.argsort([np.sum(np.sum(mat, axis = 1)[elmts]) for elmts in intvals])
    orderCol = np.argsort([np.sum(np.sum(mat, axis = 0)[elmts]) for elmts in intvals])
    return(Graphon(mat = mat[np.concatenate(intvals[orderRow])][:,np.concatenate(intvals[orderCol])]))
#out: graphon

