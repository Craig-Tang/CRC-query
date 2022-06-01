import networkx as nx
import copy
from pathlib import Path
from collections import deque
import pandas as pd
from time import time
from queue import Queue
from itertools import groupby
from operator import itemgetter

#================ Utilities ========================
def get_list_G(dataset, ts, te):
    graph_path = './Data/'+dataset+'/'
    graph_files = list(Path(graph_path).glob('*'))[ts, te]
    list_G = [nx.read_gml(g) for g in graph_files]
    return list_G

def is_kcore(G, k):
    if len(G.nodes)>0:
        sorted_deg = sorted(G.degree, key=lambda x: x[1])
        return sorted_deg[0][1]>=k
    else: return False

def k_max(G):
    return sorted(list(nx.core_number(G).items()), key=lambda x:x[1], reverse=True)[0]

def remove_theta(G, theta, query=None):
    # return the graph that has filtered edges whose weights are smaller than theta
    G_temp = G.copy()
    for (u,v) in G_temp.edges:
        if G_temp[u][v]['weight']<theta:
            G_temp.remove_edge(u,v)
    if query:
        for g in list(nx.connected_components(G_temp)):
            if query in g:
                G_temp = nx.subgraph(G_temp,g)
    return G_temp

def local_k_core(G, query, k):
    max_k_core = nx.k_core(G,k)
    filtered_G = nx.Graph()
    for g in list(nx.connected_components(max_k_core)):
        if query in g:
            filtered_G = nx.subgraph(max_k_core,g)
    return filtered_G

def get_G_max(list_G, query, theta, k, filtered=False):
    if filtered:
        G_max = [local_k_core(g, query, k) for g in list_G]
    else:
        G_max = [local_k_core(remove_theta(g, query, theta), query, k) for g in list_G]
    return G_max

def get_V_max(list_G, k):
    V_max = -1
    V_max = max([len(nx.k_core(g,k)) for g in list_G])
    return V_max

def G_induced_by_E_theta(G, theta):
    filtered_edges = [(u,v) for (u,v) in G.edges if G[u][v]['weight']>=theta]
    H = G.edge_subgraph(filtered_edges)
    return H

def cal_S_rel(V_c, T_c, V_max, T_q, alpha):
    aa = (1+alpha*alpha)*(V_c/V_max*T_c/T_q)/(alpha*alpha*V_c/V_max+T_c/T_q)
    return aa

#================ EEF ========================


def bfs_lambda_theta(list_G, theta, k, V_max, source=None):
    def edge_id(edge):
        return (frozenset(edge[:2]),) + edge[2:]
    lambda_theta = {}
    UBR = {}
    for t, G in enumerate(list_G, start = 1):
        lambda_theta[t] = {}
        UBR.setdefault(t,{})
        visited_nodes = {source}
        visited_edges = set()
        queue = deque([(source, list(G.edges(source)))])
        while queue:
            parent, children_edges = queue.popleft()
            for edge in children_edges:
                child = edge[1]
                if (child not in visited_nodes) and (child in G.nodes):
                    if G.degree[child]>=k:
                        visited_nodes.add(child)
                        queue.append((child, list(G.edges(child))))
                    else: 
                        G.remove_node(child)
                        continue
                edgeid = edge_id(edge)
                if edgeid not in visited_edges and  edge in G.edges:
                    visited_edges.add(edgeid)
                    if G.get_edge_data(*edge)['weight']>=theta:
                        e = tuple(sorted(edge))
                        if t>1 and e in lambda_theta[t-1].keys():
                            lambda_theta[t][e] = lambda_theta[t-1][e] + 1
                        else: 
                            lambda_theta[t][e] = 1
                    else: G.remove_edge(*edge)   
        UBR[1][t] = cal_S_rel(2*len(lambda_theta[t].keys())/k, 1, V_max, len(list_G), 1)
    return lambda_theta, UBR

def EEF(list_G_ori, query, theta, k, V_max, alpha=1):
    start = time()
    list_G = copy.deepcopy(list_G_ori)
    C_opt = nx.Graph()
    maxS = 0
    duration = (-1,-1)
    T_q = len(list_G)
    lambda_theta, UBR= bfs_lambda_theta(list_G, theta, k, V_max, source=query)

    ubr_tuple = [(t, ubr) for t, ubr in UBR[1].items()]
    ubr_tuple = sorted(ubr_tuple, key= lambda x: x[1], reverse=True)
    for (t,ubr) in ubr_tuple:
        for d in range(1,t+1):
            E_prime = [e for e, lam in lambda_theta[t].items() if lam>=d]
            UBR[d][t] = cal_S_rel(2*len(E_prime)/k, d, V_max, T_q, alpha)
            if len(E_prime)>=k*(k+1)/2 and UBR[d][t] > maxS:
                C = local_k_core(list_G[t-1].edge_subgraph(E_prime), query, k)
                S_rel = cal_S_rel(len(C.nodes), d, V_max, T_q, alpha)
                if S_rel>= maxS:
                    maxS = S_rel
                    C_opt = C
                    duration = (t-d+1,t)
                    # print('best d',d, 'ending at', t)
    end = time()
    print('Running time of EEF query:', end-start)
    print('CRC identified with size {} and time interval [{},{}]'.format(len(C_opt.nodes), duration[0], duration[1]))
    return maxS, C_opt, duration

#======================== WCF =========================

### Establish Theta-threshold Table

def filter_theta(G, query, theta):
    G_temp = G.copy()
    for (u,v) in G_temp.edges:
        if G_temp[u][v]['weight']*10<(theta+1):
            G_temp.remove_edge(u,v)
    if query:
        for g in list(nx.connected_components(G_temp)):
            if query in g:
                G_temp = nx.subgraph(G_temp,g)
    return G_temp

def update_core_by_remove_theta(G,theta,df):
    origin_core = nx.core_number(G)
    G_temp = filter_theta(G,None,theta)
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
        G_prime = update_core_by_remove_theta(G_prime, theta, df_theta_thres)
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
            temp_G = remove_theta(sub_G,theta)
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
                            # if label:
                            #     print('considering neighbor of Node {}, which is Node {}'.format(X.ids, Y.ids))
                            Z = Y.get_root_in_tree(theta_tree_k['node_id'])
                            if Z!=X:
                                if Z.theta > X.theta:
                                    Z.set_parent(X.ids)
                                    X.add_children(Z.ids)
                                else:
                                    Z.add_vertices(X.vertex)
                                    for c_id in X.children:
                                        theta_tree_k['node_id'][c_id].set_parent(Z.ids)
                                        Z.add_children(c_id)
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
            C_1 = remove_theta(G_k_max,theta, query)
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
    UBR = max([cal_S_rel(mu, LCT(mu, M), V_max, T_q, alpha ) for mu in M])
    return UBR

def WCF_search(list_G, WCF_indice, query, theta, k, V_max, alpha=1):
    start = time()
    T_q = len(list_G)
    all_size = [[0] * (len(list_G)+1) for _ in range(len(list_G)+1)]
    L_c = [[nx.Graph()] * (len(list_G)+1) for _ in range(len(list_G)+1)]
    score = [[-1] * (len(list_G)+1) for _ in range(len(list_G)+1)]
    maxS = 0
    OptD = (-1,-1)
    C_opt = nx.Graph()
    anchored = []
    for t, wcf_index in enumerate(WCF_indice, start=1):
        C_1 = return_C1(list_G[t-1], wcf_index, query, theta, k)
        if C_1 is None:
            anchored.append(t)
        else: 
            L_c[1][t] = C_1
            score[1][t] = -len(C_1.nodes)
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
                            all_size[d][t] = len(k_core_inter.nodes)
                    score[d][t] = cal_S_rel(len(L_c[d][t].nodes), d, V_max, T_q, alpha)
                    # print('d={} timestamps and t={} with {} nodes, {} '.format(d,t,len(L_c[d][t].nodes),is_kcore(L_c[d][t],k)))
                    if score[d][t] >= maxS:
                        maxS = score[d][t]
                        C_opt = L_c[d][t]
                        OptD = (t-d+1,t)
                    M.append(len(L_c[d][t].nodes))
            # ubr = max([(alpha*mu/V_max + (1-alpha)*(d+LCT(mu,M)-1)/T_q) for mu in M])
            ubr = max([cal_S_rel(mu, LCT(mu, M)-1+d, V_max, T_q, alpha ) for mu in M])
            if ubr <= maxS:
                break
    end = time()
    print('Running time of WCF query:', end-start)
    print('CRC identified with size {} and time interval [{},{}]'.format(len(C_opt.nodes), OptD[0], OptD[1]))
    return maxS, C_opt, score, L_c, OptD