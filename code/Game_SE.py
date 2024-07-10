from datetime import datetime
import math
from typing import List, Dict

import numpy as np


class GAME_SE:

    def __init__(self):
        self.nodes: List[int]  = []
        self.comm_vol: List[float] = []
        self.comm_cut: List[float] = []
        self.node_degree: List[float] = []
        self.node_comm:List[int] = []
        self.stop_move:List[float] = []
        self.global_stop:List[float] = []
        self.vol = 0.
        self.node_num = 0
        self.comm_num = 0
        self.adj = None
        self.sort_node = None
        self.div:Dict = {}

    def SE_1D(self,adj):
        vol = np.sum(adj)
        degree = np.sum(adj, axis=1)
        return np.sum(-(degree[degree > 0] /vol) * np.log2(degree[degree > 0] /vol))

    def partition_init(self, adj, current_mgs_num):
        self.node_degree = list(np.sum(adj, axis=1))
        self.vol = np.sum(adj)
        self.adj = adj

        for comm, nodes in self.div.items():
            com_vol = 0.
            com_cut = 0.
            for node in nodes:
                com_vol += self.node_degree[node]
                n_in = np.sum(self.adj[node, nodes])
                com_cut += self.node_degree[node] - n_in
            self.comm_vol[comm] = com_vol
            self.comm_cut[comm] = com_cut

        new_list = [0.]*current_mgs_num
        new_node =[self.node_num+i for i in range(current_mgs_num) ]

        # self.stop_move += new_list
        self.global_stop  = [x + 1 for x in self.global_stop]
        self.global_stop += new_list
        self.node_num += current_mgs_num
        self.nodes += new_node
        self.sorted_nodes = sorted(enumerate(self.node_degree), key=lambda it: it[1], reverse=True)

        for n in range(current_mgs_num):
            new_n = self.comm_num+n
            self.node_comm.append(new_n)
            self.comm_vol.append(self.node_degree[new_n])
            self.comm_cut.append(self.node_degree[new_n])


        self.comm_num += current_mgs_num
        self.stop_move = [0.] * self.node_num


    def SE_2D(self, d, g, v, n_in:float =0, leave=True):

        '''
        keep state
        '''
        if n_in < 1e-8 and math.fabs(d-g) < 1e-8 and math.fabs(d-v) <1e-8:
            return  0.

        if leave:
            '''
            leave community
            '''

            _g = g - d + (2 * n_in)
            _v = v - d

        else:

            '''
            join community
            '''
            _g = g
            _v = v

            g = _g + d - (2 * n_in)
            v = _v + d
        try:
            se =  - (_g/self.vol)*math.log2(_v/self.vol) + (g/self.vol)*math.log2(v/self.vol) + (_v/self.vol)*math.log2(_v/v) +(d/self.vol)*math.log2(self.vol/v)
        except Exception as e:
            print(f"g:{g}, _g:{_g}, v:{v}, _v:{_v}, vol:{self.vol}, d:{d}, n_in:{n_in}")
        return se

    def be_new_comm(self, node,s_node_in, d):
        # update clust C_k
        comm = self.node_comm[node]
        self.comm_vol[comm] -= d
        self.comm_cut[comm] = self.comm_cut[comm] + 2 * s_node_in - d

        # new cluster {x}
        self.node_comm[node] = self.comm_num
        self.comm_vol.append(d)
        self.comm_cut.append(d)

        self.comm_num +=1



    def to_new_comm(self, node, t_comm, node_in, s_node_in, d):
        s_comm = self.node_comm[node]
        self.comm_vol[s_comm] -= d
        self.comm_cut[s_comm] = self.comm_cut[s_comm] + 2 * s_node_in - d

        self.node_comm[node] = t_comm
        self.comm_vol[t_comm] += d
        self.comm_cut[t_comm] = self.comm_cut[t_comm] - 2 * node_in + d
    def gaming(self, max_item=10,patience=1, fix_window=2 ,verbose=False):
        if verbose:
            start = datetime.now()
            print("-" * 30)
            print('Start game')
        for it in range(max_item):
            if verbose:
                print("-" * 20)
                ind = 0
            delta_sum = 0
            move = 0
            for node, d in self.sorted_nodes:
                if verbose:
                    ind += 1
                    print('\rgaming {}: {:.2f}%'.format(it, ind/self.node_num*100), end="", flush=True)

                # if self.stop_move[node] >= patience or self.global_stop[node] >=fix_window :
                #     continue

                if  self.global_stop[node] >=fix_window :
                    continue

                s_comm = self.node_comm[node]

                adj_div = {}
                s_n_in = 0.
                relate_nodes = np.where(self.adj[node] > 0)[0]
                for r_node in relate_nodes:
                    r_comm = self.node_comm[r_node]
                    if r_comm != s_comm:
                        if r_comm not in adj_div:
                            adj_div[r_comm] = 0.
                        adj_div[r_comm] += self.adj[node][r_node]
                    else:
                        s_n_in += self.adj[node][r_node]

                se_be_new_comm = self.SE_2D(d, self.comm_cut[s_comm], self.comm_vol[s_comm], s_n_in)

                node_in = None
                delta_min = 0
                t_comm = None
                for new_comm, n_in in adj_div.items():
                    tar_g = self.comm_cut[new_comm]
                    tar_v = self.comm_vol[new_comm]
                    se_to_new_comm = se_be_new_comm - self.SE_2D(d, tar_g, tar_v, n_in, False)

                    if se_to_new_comm < delta_min:
                        delta_min = se_to_new_comm
                        t_comm = new_comm
                        node_in = n_in



                if delta_min  < 0:
                    delta_sum += delta_min
                    self.to_new_comm(node, t_comm, node_in,s_n_in, d)
                    move +=1

                elif se_be_new_comm  < 0:
                    delta_sum += se_be_new_comm
                    self.be_new_comm(node,s_n_in, d)
                    move +=1

                else:
                    self.stop_move[node] += 1

            if verbose:
                end = datetime.now()
                print("\n")
                print("time consuming:{}".format(end-start),end="  ")
                print("#move: {}".format(move),end="  ")
                print("#sum delta SE: {}".format(self.vol),end="  ")
                clusters = self.get_clusters()
                print("cluster num: {}".format(len(clusters)))

            if move == 0:
                break



    def get_clusters(self):
        divsion = {}
        for node, comm in enumerate(self.node_comm):
            if comm not in divsion:
                divsion[comm] = []
            divsion[comm].append(self.nodes[node])
        self.div = divsion
        return list(divsion.values())

