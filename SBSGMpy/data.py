'''

Create graphs from real-world data.
@author: Benjamin Sischka

'''
import os
import numpy as np
import networkx as nx
import scipy.io
import csv
import warnings
from SBSGMpy.graph import ExtGraph

def create_data(data_):
    ##### Karate
    if data_ == 'karate':
        with open(os.path.realpath('../Data/karate') + '/karate.gml', 'r', newline='') as f_:
            read_data = f_.read()
        cont1 = read_data.split('\ngraph\n')[1].replace('\n', '').replace('node', ', node,').replace('edge', ', edge,').replace(' ', '').replace(',', '', 1)
        cont2 = cont1.strip('][').split(',')
        node_label = [int(cont2[i+1].split('id')[1].replace(']', '')) for i in range(len(cont2)) if (cont2[i] == 'node')]
        edge_list = [[int(node_.replace(']', '')) for node_ in cont2[i+1].split('source')[1].split('target')] for i in range(len(cont2)) if (cont2[i] == 'edge')]
        with open(os.path.realpath('../Data/karate') + '/karate.edges', 'w', newline='') as f_:
            [f_.write(edge_list_i[0].__str__() + ' ' + edge_list_i[1].__str__() + '\n') for edge_list_i in edge_list]
    ##### Political Bloggers 2
    if data_ == 'polblogs_new':
        G = nx.read_gml(os.path.realpath('../Data/polblogs') + '/polblogs2.gml')
        coms = np.array(list(nx.get_node_attributes(G, 'value').values()))
        ## labels: liberals=0, conservatives=1 ;  colors in Adamic et al. (2005): liberals=blue, conservatives=red
        Us_real = np.maximum(np.minimum(coms, 1 - 1e-3), 1e-3)
        ### -> liberals=bluish, conservatives=reddish
        adjMat_dirc = nx.to_numpy_array(G)
        adjMat = np.minimum(1, adjMat_dirc + adjMat_dirc.T)
        np.fill_diagonal(adjMat, 0)
        labels = list(G.nodes)
        idx_nonIsol = np.where(adjMat.sum(axis=0) != 0)[0]
        max_len = len(idx_nonIsol)
        idx_vec = np.array([]).astype(int)
        new_idx = np.array([idx_nonIsol[0]])
        iter_ind = 0
        while len(new_idx) != 0:
            if iter_ind > max_len:
                warnings.warn('too many iterations')
                print('UserWarning: too many iterations')
                break
            new_idx = np.append(new_idx, np.where(adjMat[new_idx[0]] > .1)[0])
            idx_vec = np.append(idx_vec, new_idx[0])
            new_idx = np.unique(new_idx[np.logical_not(np.in1d(new_idx, idx_vec))])
            iter_ind += 1
        idx_vec.sort()
        adjMat_new = adjMat[idx_vec][:, idx_vec]
        Us_real_new = Us_real[idx_vec]
        labels_new = np.array(list(G.nodes))[idx_vec]
        ### save reduced and modified network in new file
        G_new = nx.from_numpy_array(adjMat_new)
        mapping = {list(G_new.nodes())[i]: labels_new[i] for i in range(len(labels_new))}
        G_new = nx.relabel_nodes(G_new, mapping)
        nx.set_node_attributes(G_new, {list(G_new.nodes())[i]: int(coms[idx_vec][i]) for i in range(len(labels_new))}, "value")
        nx.set_node_attributes(G_new, {list(G_new.nodes())[i]: float(Us_real_new[i]) for i in range(len(labels_new))}, "UsReal")
        nx.write_gml(G=G_new, path=os.path.realpath('../Data/polblogs') + '/polblogs2-new.gml')


def GraphFromData(data_, estMethod = None, dir_ = os.path.realpath('..'), addLabels = True, since1992 = False):
    Us_real = None
    node_labels = None
    ##### Karate
    if data_ == 'karate':
        adjMat = nx.to_numpy_array(nx.read_edgelist(os.path.join(dir_, 'Data/karate') + '/karate.edges'))
    ##### Facebook
    if data_ == 'facebook':
        adjMat = nx.to_numpy_array(nx.read_edgelist(os.path.join(dir_, 'Data/facebook/facebook') + '/0.edges'))
    ##### Facebook Complete
    if data_ == 'facebook_full':
        adjMat = nx.to_numpy_array(nx.read_edgelist(os.path.join(dir_, 'Data/facebook1') + '/123.edges'))
    ##### Political Bloggers
    if data_ == 'polblogs':
        adjMat = np.genfromtxt(os.path.join(dir_, 'Data/polblogs2') + '/polblogs2.csv', delimiter=',')
    ##### Political Bloggers 2
    if data_ == 'polblogs_new':
        G = nx.read_gml(os.path.join(dir_, 'Data/polblogs') + '/polblogs2-new.gml')
        adjMat = nx.to_numpy_array(G)
        Us_real = nx.get_node_attributes(G, 'UsReal')  # np.array(list(nx.get_node_attributes(G, 'UsReal').values()))
        node_labels = np.array(list(G.nodes))
    ##### Human Brain Data
    if data_ == 'brain':
        weightMat = scipy.io.loadmat(os.path.join(dir_, 'Data/human_brain') + '/Coactivation_matrix.mat')['Coactivation_matrix']
        adjMat = (weightMat >= 1e-5).astype('int')
        # an edge between two brain regions means that there is at least one task at which they are coactivated
    ##### Santa Fe Collaboration Network
    if data_ == 'santa_fe':
        edge_list = np.zeros((0, 2)).astype('int')
        with open(os.path.join(dir_, 'Data/SantaFeeCollaboration') + '/edge.d', 'r', newline='') as file_:
            file_cont = csv.reader(file_, delimiter='\t')
            header = next(file_cont)
            for line in file_cont:
                edge_list = np.append(edge_list, [[int(line[i]) for i in range(2)]], axis=0)
        min_label = np.min(edge_list)
        max_label = np.max(edge_list)
        net_size = max_label - min_label + 1
        adjMat = np.zeros((net_size, net_size)).astype('int')
        adjMat[edge_list[:, 0] - 1, edge_list[:, 1] - 1] = adjMat[edge_list[:, 1] - 1, edge_list[:, 0] - 1] = 1
    ##### Yeast
    if data_ == 'yeast':
        edge_list = np.zeros((0, 2)).astype('int')
        with open(os.path.join(dir_, 'Data/yeast') + '/Uetz_screen.txt', 'r', newline='') as file_:
            file_cont = csv.reader(file_, delimiter='\t')
            for line in file_cont:
                edge_list = np.append(edge_list, [line], axis=0)
        node_label = np.unique(edge_list)
        net_size = len(node_label)
        label_list = np.hstack((np.arange(net_size).reshape(net_size, 1) + 1, node_label.reshape(net_size, 1)))
        for i in range(net_size):
            edge_list[edge_list == label_list[i, 1]] = label_list[i, 0]
        edge_list = edge_list.astype('int')
        adjMat = np.zeros((net_size, net_size)).astype('int')
        adjMat[edge_list[:, 0] - 1, edge_list[:, 1] - 1] = 1
        adjMat[np.triu_indices(net_size)] = adjMat[np.triu_indices(net_size)] + adjMat.T[np.triu_indices(net_size)]
        adjMat.T[np.triu_indices(net_size)] = adjMat[np.triu_indices(net_size)]
        np.fill_diagonal(adjMat, 0)
        adjMat = np.minimum(adjMat, 1)
    ##### Airports / Aircraft Routes
    if data_ == 'airports':
        ## an edge means there is at least one connection between two airports in one direction (!)
        adjMat = nx.to_numpy_array(nx.read_edgelist(os.path.join(dir_, 'Data/routes_aircraft') + '/routes.edgelist', delimiter=",", nodetype=int))
    ##### Military Alliances
    if data_ == 'alliances':
        if since1992:
            file_cont = open(os.path.join(dir_, 'Data/alliances') + '/country_list.csv', "r")
            reader = csv.reader(file_cont)
            header1 = next(reader)
            country_infoAll = np.zeros((0, len(header1)))
            for line in reader:
                country_infoAll = np.append(country_infoAll, [line], axis=0)
            file_cont.close()
            with open(os.path.join(dir_, 'Data/alliances') + '/alliances_strong_post_g.csv', "r") as file_cont:
                reader = csv.reader(file_cont)
                header2 = next(reader)
                country_info = np.zeros((0, 4))
                for line in reader:
                    country_info = np.append(country_info, [[line[i] for i in range(4)]], axis=0)
            for i in range(country_info.shape[0]):
                country_info[i ,2] = country_infoAll[country_infoAll[: ,4] == country_info[i ,3], 0][0]
            with open(os.path.join(dir_, 'Data/alliances') + '/alliances_strong_post_el.csv') as f_cont:
                ed_list = f_cont.read().splitlines()[1:]
            G_nx1 = nx.read_edgelist(ed_list, delimiter=",", nodetype=int)
            adjMat = nx.to_numpy_array(G_nx1)
            node_labels = np.array([country_info[country_info[:, 0].astype('int') == list(G_nx1.nodes)[i], np.array([2 ,3 ,1])] for i in range(G_nx1.order())])
        else:
            adjMat = np.zeros((0, 257)).astype('int')
            with open(os.path.join(dir_, 'Data/alliances') + '/alliances_strong_post_adjMat_2016.csv') as f_cont:
                reader = csv.reader(f_cont)
                node_labels = next(reader)[1:]
                for line in reader:
                    adjMat = np.append(adjMat, [[int(int_i) for int_i in line[1:]]], axis=0)
            margSum_pos = (adjMat.sum(axis=0) > 0)
            adjMat = adjMat[margSum_pos][:, margSum_pos]
            node_labels = np.array(node_labels)[margSum_pos]
            # remove isolated groups and single connected nodes
            all_other = np.logical_not(np.in1d(node_labels, ['China', 'Cuba', 'North Korea', 'Bosnia and Herzegovina', 'Syria']))
            adjMat = adjMat[all_other][:, all_other]
            node_labels = node_labels[all_other]
    #####
    return(ExtGraph(A = adjMat, Us_real=Us_real, estMethod=estMethod, labels=node_labels if addLabels else None))

