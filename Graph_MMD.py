from grakel import Graph
from collections import Counter
import pandas as pd
import numpy as np


def Build_graph(dataset):
    '''
    Convert the dataset into a list of Graph instance
    '''
    states = pd.unique(dataset["State"])
    states.sort()
    node_label = {num: num for num in states}
    node_label["nan"] = 'nan'
    dataset = dataset.pivot(index="ID",columns="Time",values="State")
    max_len = dataset.shape[1]
    G = []
    for index, data in dataset.iterrows():
        length = data.last_valid_index()
        
        #edges = set()
        edges = Counter()

        if length == max_len: # 
            for i in range(length-1):
                edges[(data.iloc[i],data.iloc[i+1])] += 1
            edges[(data.iloc[i+1],'nan')] += 1
        else:
            for i in range(length):
                edges[(data.iloc[i],data.iloc[i+1])] += 1
        
        #pdb.set_trace()
        G.append(Graph(dict(edges),edge_labels=edges,node_labels=node_label))
        
    return G

def MMD(G1,G2,Kernel_class):
    '''
    The input G are two list of Graph instance,
    The implementation can be two different size, but I add an assert statement.
    Kernel_class is an abstract class name, now work with EdgeHistogram.
    '''
    
    assert len(G1) == len(G2)
    
    k = Kernel_class(normalize=True)
    K = k.fit_transform(G1+G2)
    
    Kx = K[0:len(G1),0:len(G1)]
    Ky = K[-len(G2):,-len(G2):]
    Kxy = K[len(G1):,0:len(G1)]
    
        
    return 1.0 / (len(G1) * (len(G1) - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (len(G2) * (len(G2) - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (len(G1) * len(G2)) * Kxy.sum()




from grakel.kernels import EdgeHistogram

real = pd.read_csv("Real-Sequence-Data.csv")
syn = pd.read_csv("Syn-Sequence-Data.csv") 
G1 = Build_graph(syn)
G2 = Build_graph(real)


import time
start = time.perf_counter()
print(MMD(G1,G2,EdgeHistogram))
finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')
