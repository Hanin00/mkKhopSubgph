import sys
import numpy as np
import pandas as pd
import torch
import csv

import torch_geometric.utils

import util as ut
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import pickle



gList = []
imgCnt = 1200
start = time.time()
with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
end = time.time()
print(f"파일 읽는데 걸리는 시간 : {end - start:.5f} sec") # 파일 읽는데 걸리는 시간 : 24.51298 sec
# 이거 아님 igList = [49,59,155,240,283,298,316,370,402,431,511,519,646,676,744,866,929,930,1101,1115,1135,1173,1195]
igList = [50,60,156,241,284,299,317,371,403,432,512,520,647,677,745,867,930,931,1102,1116,1136,1174,1196]

# a = ut.AllEdges(data,393)
# print(a)
# b = ut.AllEdges(data,394)
# print(b)
# c = ut.AllEdges(data,395)
# print(c)

objNamesList = []
for imgId in tqdm(range(imgCnt)):
    objectIds, objectNames = ut.AllNodes(data, imgId)
    objNamesList += objectNames
objNamesList = list(set(objNamesList))
totalEmbDict = ut.FeatEmbeddPerTotal(objNamesList)
for i in tqdm(range(imgCnt)):
    if i in igList :
        continue
    else :
        objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(data,i)
        # networkX graph 객체 생성 ---
        objIdSet, objNameList = ut.AllNodes(data, i)
        df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, })
        gI = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')
        nodesList = list(gI.nodes)

        # node attribute 부여 ---
        embDict = ut.MatchDictImage(objIdSet,objNameList,totalEmbDict)
        if i == 294 :
            print(embDict.keys())
            print(embDict.values())

            nx.set_node_attributes(gI, embDict, "attr")  # node attribute 부여
            plt.figure(figsize=[15, 7])
            nx.draw(gI, with_labels=True)
            plt.show()
            break

        nx.set_node_attributes(gI, embDict, "attr") # node attribute 부여

        # graph에서 노드 id 0부터 시작하도록 ---
        listA = list(set(objId + subjId))
        listIdx = range(len(listA))
        dictIdx = {name: value for name, value in zip(listA, listIdx)}
        gI = nx.relabel_nodes(gI, dictIdx)
        nx.set_node_attributes(gI, 1, "weight")
        gList.append(gI)