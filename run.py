import click
from time import time
import networkx as nx
import CRC
from joblib import Parallel, delayed


@click.command()
@click.option('--dataset', prompt='Dataset name(str)', help='The name of the dataset')
@click.option('--theta', prompt='Theta(float)', help='The value of the parameter Theta')
@click.option('--k', prompt='K(int)', help='The value of the parameter K')
@click.option('--query', prompt='Query(str)', help='The value of the Query vertex')
@click.option('--alpha', prompt='Alpha(float)', help='The value of the parameter Alpha')
@click.option('--start_time', prompt='T_s(int)', help='The start timestamp(included, start from 0)')
@click.option('--end_time', prompt='T_e(int)', help='The end timestamp(excluded)')
@click.option('--method', prompt='Type one number to chose the algorithm (int): [1]EEF; [2]WCF;', help='Reliable community search methods')

def query(dataset, theta, k, query, method, alpha, start_time, end_time):
    theta = float(theta)
    k = int(k)
    ts = int(start_time)
    te = int(end_time)
    alpha = float(alpha)

    list_G = CRC.get_list_G(dataset, ts, te)
    V_MAX = CRC.get_V_max(list_G, k)

    if method == "1":
        maxS, C_opt, duration = CRC.EEF(list_G, query, theta, k, V_MAX, alpha)
        file_name = '{}.output-{}-{}-{}-{}_EEF'.format(dataset, theta, k, query, alpha)
    if method == "2":
        start = time()
        theta_thres_all = []
        wcf_indices = []
        theta_thres_all = Parallel(n_jobs=-1)(delayed(CRC.theta_thres_table)(g) for g in list_G)
        wcf_indices = Parallel(n_jobs=-1)(delayed(CRC.theta_tree)(theta_thres_all[i],g) for i,g in enumerate(list_G))
        end = time()
        print("Index construction time:", end-start)

        maxS, C_opt, score, L_c, duration = CRC.WCF_search(list_G, wcf_indices, query, theta, k, V_MAX, alpha)
        file_name = '{}.output-{}-{}-{}-{}_WCF'.format(dataset, theta, k, query, alpha)
    
    f = open('./Output/'+file_name, 'w')
    f.writelines('Score: ' + str(maxS) +'\n')
    f.write('Interval: ' + str(duration) +'\n')
    f.write('Nodes: ' + str(list(C_opt.nodes)))
    f.close()
    print('CRC output at: ', file_name)

if __name__ == '__main__':
    query()