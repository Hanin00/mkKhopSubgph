import networkx as nx
import pickle
import sys
import dgl
import torch
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

# 중첩으로 만들어서 노드 개수 확인하는 코드 짜고, 일단 임의로 hop수가 많을 것 같은거.. 슥 샥 보기..)


# 1000개 그래프에서 각 노드의 neighbor가 10개 이상인 것


with open("data/networkx1000_attr.pickle", "rb") as fr:
    netX = pickle.load(fr)
netX = netX[:1000]
omg = netX[289]
print(omg)
print(omg.nodes(data=True))

sys.exit()



with open("data/networkx1000_mo.pickle", "rb") as fr:
    netX = pickle.load(fr)
netX = netX[:1000]
omg = netX[491]
print(omg)
#
# #이미지 nx.draw로 띄워 확인
nx.draw(omg, with_labels=True)
plt.show()
plt.figure(figsize=[15, 7])

sys.exit()
#
#
# #for gId in range(len(netX)) :
# for gId in range(100) :
#    nxG = dgl.from_networkx(netX[gId],["f0","f1","f2"])
#    subG = dgl.khop_out_graph(nxG,2)
#    print("gId : ", gId, "subG : ", subG)
#
# print(netX[5])

# plt.figure(figsize=[15,7])
# nx.draw(netX[5], with_labels=True)
# plt.show()
# sys.exit()



# G = netX[0]
#
# upper10 = []
# maxNodeNum = []
# for gId in tqdm(range(len(netX))) :
#     maxNodeInGraph = []
#     G = netX[gId]
#     for i in G.nodes:
#         a = []  # 한 그래프에서 노드 하나의 neighbor 개수
#         for n in G.neighbors(i):
#             a.append(n)
#         maxNodeInGraph.append(len(a))
#     maxNodeNum.append(max(maxNodeInGraph))
#     if max(maxNodeInGraph) >= 10 :
#         upper10.append((gId,(max(maxNodeInGraph))))
#         # plt.figure(figsize=[15,7])
#         # nx.draw(G, with_labels=True)
#         # plt.show()
#         #break
# print(len(maxNodeNum))
# print(max(maxNodeNum))
# print(upper10)
# print(len(upper10))




# nxG = dgl.from_networkx(G)
# h0, _ = dgl.khop_out_subgraph(nxG, 0,1)
# G0 = dgl.to_networkx(h0)
# plt.figure(figsize=[15,7])
# nx.draw(G0,  with_labels = True)
#
# plt.show()



