import datetime
from queue import Queue
import sys, copy
import joblib, os
import json as js
from itertools import groupby
from operator import itemgetter
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
# from tqdm import tqdm
from csv import reader
from collections import deque
import matplotlib.pyplot as plt

def is_kcore(G, k):
    if len(G.nodes)>0:
        sorted_deg = sorted(G.degree, key=lambda x: x[1])
        return sorted_deg[0][1]>=k
    else: return False

def k_max(G):
    return sorted(list(nx.core_number(G).items()), key=lambda x:x[1], reverse=True)[0]

def remove_theta(G, query, theta):
    # return the graph that has filtered edges whose weights are smaller than theta
    G_temp = copy.deepcopy(G)
    for (u,v) in G_temp.edges:
        if G_temp[u][v]['weight']<theta:
            G_temp.remove_edge(u,v)
    if query:
        for g in list(nx.connected_components(G_temp)):
            if query in g:
                G_temp = nx.subgraph(G_temp,g)
    return G_temp

def local_k_core(G, query, k):
    # return the maximum local k-core of the query vertex
    filtered_G = nx.Graph()
    G_res = G.copy()
    core_filtered = [a[0] for a in nx.core_number(G_res).items() if a[1]<k]
    G_res.remove_nodes_from(core_filtered)
    for g in list(nx.connected_components(G_res)):
        if query in g:
            filtered_G = nx.subgraph(G_res,g)
    if len(filtered_G.nodes) > 0:
        assert is_kcore(filtered_G,k) ==True
    return filtered_G

def get_V_max(list_G, query, theta, k):
    V_max = -1
    for g in list_G:
        g_theta = remove_theta(g, query, theta)
        g_max = local_k_core(g_theta, query, k)
        if len(g_max.nodes)> V_max:
            V_max = len(g_max.nodes)
    return V_max

def G_induced_by_E_theta(G, theta):
    filtered_edges = []
    for (u,v) in G.edges:
        if G[u][v]['weight']>=theta:
            filtered_edges.append((u,v))
    H = G.edge_subgraph(filtered_edges)
    return H

def S_rel(V_c, T_c, V_max, T_q, alpha):
    if V_c == 0:
        return 0
    return round(alpha*V_c/V_max + (1-alpha)*T_c/T_q,6)

def EEF(list_G_ori, query, theta, k, alpha):
    list_G = copy.deepcopy(list_G_ori)
    V_max = get_V_max(list_G, query, theta, k)
    T_q = len(list_G)
    C_opt = None
    maxS = 0
    lambda_theta = {}
    UBR = {}
    for t, G in enumerate(tqdm(list_G), start = 1):
        lambda_theta[t] = {}
        UBR.setdefault(t,{})
        g_fil = G_induced_by_E_theta(G,theta)
        g_max = local_k_core(g_fil, query, k)
        edges = list(nx.edge_bfs(g_max, query))

        for (v,w) in edges:
            e = tuple(sorted((v,w)))
            if g_max[v][w]['weight']>=theta:
                if t>1 and e in lambda_theta[t-1].keys():
                    lambda_theta[t][e] = lambda_theta[t-1][e] + 1
                else: 
                    lambda_theta[t][e] = 1
        UBR[1][t] = S_rel(2*len(lambda_theta[t].keys())/k, 1, V_max, T_q, alpha)
    ubr_tuple = [(t, ubr) for t, ubr in UBR[1].items()]
    ubr_tuple = sorted(ubr_tuple, key= lambda x: x[1], reverse=True)
    for (t,ubr) in ubr_tuple:
        for d in range(1,t+1):
            E_prime = [e for e, lam in lambda_theta[t].items() if lam>=d]
            UBR[d][t] = S_rel(2*len(E_prime)/k, d, V_max, T_q, alpha)
            if len(E_prime)>=k*(k+1)/2 and UBR[d][t] > maxS:
                C = local_k_core(list_G[t-1].edge_subgraph(E_prime), query, k)
                S_rel = len(C.nodes)*d
                if S_rel>= maxS:
                    maxS = S_rel
                    C_opt = C
    return maxS, C_opt

### Establish Theta-threshold Table

def update_core_by_remove_theta(G,theta,df):
    origin_core = nx.core_number(G)
    G_temp = remove_theta(G,None,theta)
    filtered_core = nx.core_number(G_temp)
    for v,c in origin_core.items():
        if filtered_core[v] < c:
            for i in range(c - filtered_core[v]):
                df.loc[v][c-i] = round(theta*0.1,2)
    return G_temp

def theta_thres_table(G):
    k = k_max(G)[1]
    col_k = ['vertex'] + [i for i in range(1,k+1)]
    init = dict.fromkeys(col_k)
    init['vertex'] = sorted(list(G.nodes))
    df_theta_thres = pd.DataFrame(init)
    df_theta_thres.set_index(['vertex'], inplace=True)
    G_prime = copy.deepcopy(G)
    for theta in range(11):
        G_prime = update_core_by_remove_theta(G_prime, theta*0.1, df_theta_thres)
    df_theta_thres = df_theta_thres.fillna(0)
    return df_theta_thres


### Construct the WCF-Index

class Node:
    def __init__(self, ids, list_v, theta):
        self.ids = ids
        self.vertex = list_v
        self.theta = theta
        self.parent = None
        self.children = set()

    def contains_v(self,v):
        return v in self.vertex

    def add_vertices(self,v):
        self.vertex.extend(v)

    def replace_vertices(self,v):
        self.vertex = v

    def remove_vertices(self,v):
        self.vertex = self.vertex.remove(v)

    def set_parent(self,p_Node_id):
        self.parent = p_Node_id

    def add_children(self,c_Node_id):
        self.children.add(c_Node_id)

    def remove_children(self,c_Node_id):
        self.children.remove(c_Node_id)

    def remove_parent(self):
        self.parent = None
    
    def remove_children(self,c_Node_id):
        self.children.remove(c_Node_id)

    def get_root_in_tree(self,tree):
        if self.parent == None:
            return self
        p_id = self.parent
        p_node = tree[p_id]
        while p_node.parent:
            p_id = p_node.parent
            p_node = tree[p_id]
        return p_node

    def get_subgraph_in_tree(self,tree):
        visited = []
        all_nodes = []
        Q = Queue()
        Q.put(self.ids)
        while not Q.empty():
            X = Q.get()
            visited.append(X)
            all_nodes.extend(tree[X].vertex)
            for Y in tree[X].children:
                Q.put(Y)
        return all_nodes, visited

    def info(self):
        print('Node: {}\nverteices: {}\ntheta: {}\nparent: {}\nchildren: {}'.format(self.ids, self.vertex, self.theta, self.parent, self.children))


def theta_tree(theta_thres_df, G):
    WCF_index = {}
    k = k_max(G)[1]
    label = False
    for k_curr in range(1,k+1):
        theta_tree_k = {}
        theta_tree_k['theta'] = {}
        theta_tree_k['node_id'] = {}
        ids = 0
        merged_ids = set()
        g = theta_thres_df.groupby(k_curr)
        theta_v = sorted(list(g.indices.keys()), reverse=True)
        for theta in theta_v:
            theta_tree_k['theta'][theta] = []
            node_v = g.get_group(theta).index.values.tolist()
            sub_G = nx.subgraph(G,node_v)
            temp_G = remove_theta(sub_G,None,theta)
            for C in list(nx.connected_components(temp_G)):
                merged = False
                X = Node(ids,list(C),theta)
                theta_tree_k['node_id'][ids] = X
                theta_tree_k['theta'][theta].append(ids)
                out_N = list(get_N_of_subgraph(nx.subgraph(G,C), G))
                visited = []
                for v in out_N:
                    if theta_thres_df.loc[v][k_curr]>theta and not merged:
                        nei = [y for y in theta_tree_k['node_id'].values() if y.contains_v(v)]
                        if nei and (nei[0] not in visited):
                            Y = nei[0]
                            visited.append(Y)
                            if label:
                                print('considering neighbor of Node {}, which is Node {}'.format(X.ids, Y.ids))
                            Z = Y.get_root_in_tree(theta_tree_k['node_id'])
                            if Z!=X:
                                if Z.theta > X.theta:
                                    Z.set_parent(X.ids)
                                    X.add_children(Z.ids)
                                else:
                                    Z.add_vertices(X.vertex)
                                    for c_id in X.children:
                                        theta_tree_k['node_id'][c_id].set_parent(Z.ids)
                                    if ids not in merged_ids:
                                        theta_tree_k['node_id'].pop(ids,'merged')
                                        theta_tree_k['theta'][theta].remove(ids)
                                        merged_ids.add(ids)
                                    merged = True   
                ids += 1
        WCF_index[k_curr] = theta_tree_k
    return WCF_index


### WCF-Index Based Query

def get_N_of_subgraph(sub_G, G):
    # return neighbors of the subgraph
    node_N = set()
    for node in sub_G.nodes:
        node_N.update([i for i in G.neighbors(node)])
    return [i for i in node_N if i not in sub_G.nodes]

def is_root(tree, node, theta):
    res = False
    if theta-0.1 < tree['node_id'][node].theta <= theta:
        if tree['node_id'][node].parent is None: res = True
        elif tree['node_id'][tree['node_id'][node].parent].theta<theta:
            res = True
    return res

def return_C1(G, wcf_index, query, theta, k):
    C_1 = None
    if k not in wcf_index.keys():
        return C_1
    tree = wcf_index[k]
    S = [node for node in tree['node_id'].keys() if is_root(tree, node, theta)]
    S = sorted(S, key=lambda x: tree['node_id'][x].theta)
    for root_id in S:
        C_vertices = tree['node_id'][root_id].get_subgraph_in_tree(tree['node_id'])[0]
        if query in C_vertices:
            G_k_max = nx.subgraph(G, C_vertices)
            C_1 = remove_theta(G_k_max,query,theta)
            C_1 = local_k_core(C_1,query,k)
    return C_1

def LCT(mu,M):
    maxLen = 0
    currLen = 0
    for value in M:
        if value >= mu:
            currLen += 1
        else:
            if currLen > maxLen:
                maxLen = currLen
            currLen = 0
    if currLen > maxLen:
        maxLen = currLen
    return maxLen
    
def UBR_wcf(T_i, L_c, V_max, T_q, alpha):
    M = [len(L_c[1][t].nodes) for t in T_i]
    # UBR = max([(alpha*mu/V_max + (1-alpha)*LCT(mu,M)/T_q) for mu in M])
    UBR = max([S_rel(mu, LCT(mu, M), V_max, T_q, alpha ) for mu in M])
    return UBR

def WCF_search(list_G, WCF_indice, query, theta, k, alpha, enable_ubr=True):
    V_max = get_V_max(list_G, query, theta, k)
    T_q = len(list_G)
    L_c = [[nx.Graph()] * (len(list_G)+1) for _ in range(len(list_G)+1)]
    score = [[0] * (len(list_G)+1) for _ in range(len(list_G)+1)]
    maxS = 0
    C_opt = nx.Graph()
    anchored = []
    for t, wcf_index in enumerate(WCF_indice, start=1):
        C_1 = return_C1(list_G[t-1], wcf_index, query, theta, k)
        if C_1 is None:
            anchored.append(t)
        else: L_c[1][t] = C_1
    T_elig = [t+1 for t in range(len(list_G)) if t+1 not in anchored]
    T_s = []
    for _, g in groupby(enumerate(T_elig), lambda x: x[0]-x[1]):
        T_s.append(list(map(itemgetter(1), g)))
    all_ubr = {}
    for T_i in T_s:
        all_ubr[tuple(T_i)] = (UBR_wcf(T_i, L_c, V_max, T_q, alpha))
    sorted_T_s = sorted(all_ubr.keys(), key=lambda x: all_ubr[x], reverse=True)
    for T_i in sorted_T_s:
        if all_ubr[T_i]<= maxS: continue
        for d in range(1, len(T_i)+1):
            M = []
            for t in T_i:
                if d <= (t-T_i[0]+1):
                    if d>1:
                        inter = nx.intersection(L_c[d-1][t-1], L_c[d-1][t])
                        k_core_inter = local_k_core(inter,query,k)
                        if k_core_inter:
                            L_c[d][t] = k_core_inter
                    score[d][t] = S_rel(len(L_c[d][t].nodes), d, V_max, T_q, alpha)
                    print('d={} timestamps and t={} with {} nodes, {} '.format(d,t,len(L_c[d][t].nodes),is_kcore(L_c[d][t],k)))
                    if score[d][t] >= maxS:
                        maxS = score[d][t]
                        C_opt = L_c[d][t]
                    M.append(len(L_c[d][t].nodes))
            # ubr = max([(alpha*mu/V_max + (1-alpha)*(d+LCT(mu,M)-1)/T_q) for mu in M])
            ubr = max([S_rel(mu, LCT(mu, M)-1+d, V_max, T_q, alpha ) for mu in M])
            if ubr <= maxS and enable_ubr:
                break
    return maxS, C_opt, score, L_c


### Maintenance

def MCD(G, v):
    num = 0
    core_dict = nx.core_number(G)
    for w in G.neighbors(v):
        if core_dict[v] <= core_dict[w]:
            num += 1
    return num

def sub_core(G, query_v):
    H = nx.Graph()
    Q = deque([query_v])
    cd = {}
    core_dict = nx.core_number(G)
    k = core_dict[query_v]
    visited = set()
    visited.add(query_v)
    while Q:
        v = Q.popleft()
        H.add_node(v)
        for w in G.neighbors(v):
            if core_dict[w] >= k:
                if v in cd.keys():
                    cd[v] += 1
                else: 
                    cd[v] = 1
                if core_dict[w] == k and (w not in visited):
                    Q.append(w)
                    H.add_edge(v,w)
                    visited.add(w)
    return H, cd

def pure_core(G, query_v):
    H = nx.Graph()
    Q = deque([query_v])
    cd = {}
    core_dict = nx.core_number(G)
    k = core_dict[query_v]
    visited = set()
    visited.add(query_v)
    while Q:
        v = Q.popleft()
        H.add_node(v)
        for w in G.neighbors(v):
            if (core_dict[w] > k) or (core_dict[w]==k and MCD(G,w)>k):
                if v in cd.keys():
                    cd[v] += 1
                else: 
                    cd[v] = 1
                if core_dict[w] == k and (w not in visited):
                    Q.append(w)
                    H.add_edge(v,w)
                    visited.add(w)
    return H, cd

def theta_order_v(G, v):
    res = {}
    v_k_max = nx.core_number(G)[v]
    filtered_G = G.copy()
    for theta in range(10):
        filtered_G = remove_theta(filtered_G,None,theta)
        curr_k = nx.core_number(filtered_G)[v]
        if curr_k < v_k_max:
            for i in range(1,v_k_max+1):
                res[i] = round(theta*0.1,2)
            v_k_max = curr_k
    return res
    
def update_theta_df(theta_df_old, v_list, G_new):
    new_thres = {}
    theta_df_new = theta_df_old.copy()
    for v in v_list:
        new_thres[v] = theta_order_v(G_new, v)
    print(new_thres)
    for v, s in new_thres.items():
        for k, theta in s.items():
            theta_df_new.loc[v,k] = theta
    return theta_df_new

def change_thres_df(theta_thres_df_new, v_list, i,k,theta):
    theta_thres_df_new.loc[v_list[i],k] = theta

def update_wcf_index(theta_thres_df_old, v_list, G_new):
    theta_thres_df_new = theta_thres_df_old.copy()
    new_thres = Parallel(n_jobs=-1)(delayed(theta_order_v)(G_new, v) for v in v_list)
    
    for i, s in enumerate(new_thres):
        for k, theta in s.items():
            theta_thres_df_new.loc[v_list[i], k] = theta

    return theta_tree(theta_thres_df_new, G_new)

def maintenance_core_by_remove_theta(G,theta,df, v_list):
    orign_core = nx.core_number(G)
    G_temp = remove_theta(G,None,theta)
    filtered_core = nx.core_number(G_temp)
    for v in v_list:
        if filtered_core[v] < orign_core[v]:
            for i in range(orign_core[v] - filtered_core[v]):
                df.loc[v][orign_core[v]-i] = round(theta*0.1,2)
    return G_temp

def maintenance_theta_thres_table(theta_thres_df_old, update_v_list, G_new):
    theta_new = theta_thres_df_old.copy()
    G_prime = G_new.copy()
    for theta in range(11):
        # print(theta)
        G_prime = maintenance_core_by_remove_theta(G_prime, theta, theta_new, update_v_list)
    theta_new = theta_new.fillna(0)
    return theta_new

def maintenance_index(v_list, G_new, index_old, df_old):
    thres_change = {}
    thres_max = {}
    index_new = index_old.copy()
    df_new = maintenance_theta_thres_table(df_old, v_list, G_new)
    for v in v_list:
        for k in df_old.loc[v].index:
            if df_old.loc[v, k] != df_new.loc[v, k]:
                thres_change.setdefault(k,{})[v] = (df_old.loc[v, k], df_new.loc[v, k])
                thres_change[k] = dict(sorted(thres_change[k].items(), key=lambda r: r[0], reverse=True))
                thres_max.setdefault(k,0)
                thres_max[k] = max(thres_max[k], df_old.loc[v, k], df_new.loc[v, k])
    for k in thres_max.copy():
        rest_id = [item for sublist in list(index_new[k]['theta'].values()) for item in sublist]
        for theta in index_new[k]['theta'].copy():
            if theta <= thres_max[k]:
                index_new[k]['theta'].pop(theta, None)
        for node_id in index_new[k]['node_id'].copy():
            rest_id = [item for sublist in list(index_new[k]['theta'].values()) for item in sublist]
            if node_id not in rest_id:
                index_new[k]['node_id'].pop(node_id,None)
        for node in index_new[k]['node_id'].values():
            if node.parent not in rest_id:
                node.remove_parent()
            if node.children:
                for child_id in node.children:
                    if child_id not in rest_id:
                        node.remove_children(child_id)

    for k_curr in thres_max.keys():
        g = df_new.groupby(k_curr)
        theta_v = sorted(list(g.indices.keys()), reverse=True)
        ids = 0
        merged_ids = set()
        for theta in theta_v:
            if theta<=thres_max[k_curr]:
                index_new[k_curr]['theta'][theta] = []
                node_v = g.get_group(theta).index.values.tolist()
                sub_G = nx.subgraph(G_new,node_v)
                temp_G = remove_theta(sub_G,None,theta)
                for C in list(nx.connected_components(temp_G)):
                    merged = False
                    X = Node('new'+str(ids),list(C),theta)
                    index_new[k_curr]['node_id']['new'+str(ids)] = X
                    index_new[k_curr]['theta'][theta].append('new'+str(ids))
                    out_N = list(get_N_of_subgraph(nx.subgraph(G_new,C), G_new))
                    visited = []
                    for v in out_N:
                        if df_new.loc[v,k_curr]>theta and not merged:
                            nei = [y for y in index_new[k_curr]['node_id'].values() if y.contains_v(v)]
                            if nei and (nei[0] not in visited):
                                Y = nei[0]
                                visited.append(Y)
                                Z = Y.get_root_in_tree(index_new[k_curr]['node_id'])
                                if Z!=X:
                                    print('the root of {} is {}'.format(Y.vertex,Z.vertex))
                                    if Z.theta > X.theta:
                                        Z.set_parent(X.ids)
                                        X.add_children(Z.ids)
                                        print('Set Node {} parent to be {}'.format(Z.vertex,X.vertex))
                                    else:
                                        Z.add_vertices(X.vertex)
                                        for c_id in X.children:
                                            index_new[k_curr]['node_id'][c_id].set_parent(Z.ids)
                                        if 'new'+str(ids) not in merged_ids:
                                            index_new[k_curr]['node_id'].pop('new'+str(ids),'merged')
                                            index_new[k_curr]['theta'][theta].remove('new'+str(ids))
                                            merged_ids.add('new'+str(ids))
                                            print('Node {} is merged'.format('new'+str(ids)))
                                        merged = True   
                    ids += 1
    return index_new


### Compression

def compress_wcf_indices(wcf_indices, ratio):
    gains = {}
    node_Vs = {}
    for t, wcf_index in enumerate(wcf_indices):
        for k, theta_tree in wcf_index.items():
            for node_id, node in theta_tree['node_id'].items():
                v_comb = tuple(sorted(node.vertex))
                if v_comb in node_Vs.keys():
                    node_Vs[v_comb]['fre'] = node_Vs[v_comb]['fre']+1
                    node_Vs[v_comb]['t'] = node_Vs[v_comb]['t']+[t]
                    node_Vs[v_comb]['k'] = node_Vs[v_comb]['k']+[k]
                    node_Vs[v_comb]['node_id'] = node_Vs[v_comb]['node_id']+[node_id]
                else:
                    node_Vs[v_comb] = {'fre':1, 't':[t], 'k':[k], 'node_id': [node_id]}
    for Vs, Vs_stats in node_Vs.items():
        gain = (len(Vs)-1)*Vs_stats['fre'] - len(Vs)
        if gain > 0:
            gains[Vs] = gain


    gains = dict(sorted(gains.items(), key=lambda item: item[1], reverse=True))
    total_gain = sum(gains.values())
    total_space = sum([len(i)*Vs_stats['fre'] for i, Vs_stats in node_Vs.items()])
    opt_compress = 1-total_gain/total_space
    opt_thres = 1-(1-opt_compress)*ratio
    print(total_gain, total_space, opt_compress, opt_thres)


    auxiliary_table = {}
    compressed_index = wcf_indices.copy()

    obtained_gains = 0
    vir_id = 0
    for Vs, gain in gains.items():
        auxiliary_table['Vir_'+str(vir_id)] = Vs
        Vs_stats = node_Vs[Vs]
        for i, k in enumerate(Vs_stats['k']):
            compressed_index[Vs_stats['t'][i]][k]['node_id'][Vs_stats['node_id'][i]].replace_vertices('Vir_'+str(vir_id))
        obtained_gains += gain
        vir_id += 1
        if 1-obtained_gains/total_space<=opt_thres:
            break

    return compressed_index, auxiliary_table

def indices_to_json(wcf_indices):
    wcf_json = []
    for wcf_index in wcf_indices:
        index_dict = {}
        for k, v in wcf_index.items():
            index_dict[k] = wcf_index[k]['theta'].copy()
            for theta, node_ids in index_dict[k].items():
                index_dict[k][theta] = [wcf_index[k]['node_id'][node_id].vertex for node_id in node_ids]
        wcf_json.append(index_dict)
    return wcf_json
