import time
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import sys
import dgl
import torch


def imgShow(nexG):
    nx.draw(nexG, with_labels=True)
    plt.show()


with open("./data/networkx1000_mo.pickle", "rb") as fr:
    networkXSet = pickle.load(fr)
networkXSet = networkXSet[:1000]

# nexG = networkXSet[0]
# hop = 2

'''
    nx - graph.subgraph의 nodes에 iterable 넣을 수 있음 
    -> neigbor로 subgraph 생성 가능
    모든 노드에 대해 1hop의 subgraph 생성 가능
    이때, 노드 개수로 자르기 위해서는 iterater를 list로 변경해 
    랜덤하게 nodeId를 뽑아 서브그래프를 만들 수 있음

    이때 문제점 : 2hop의 subgraph는 어떻게 만들 수 있는지?

    원본 그래프 데이터 형태 : networkx Graph
    netGraph -> torch.geometric 으로 변경 후 khop grpah로 n-hop그래프 생성가능
    이때 2-hop 설정 시 1 hop 만 있는 그래프에서 사용되지 않는 노드듣도 호출됨을 확인함

    아래 코드를 통해 neighbor 를 통해 subgraph를 만들고 실제로 그래프를 확인해봄
'''

# 0번 노드에 대한 1-hop 노드리스트와 기존 그래프의 edge를 참고해 subgraph를 생성
# imgShow(nexG)

'''
    그래프 하나 기준 하나의 노드에 대한 1-hop subgraph
    (이웃 노드간의 관계도 모두 고려)
'''
def OneHopSubgph(nexG, targetNd) :
    nodeList = list(nexG.neighbors(targetNd))
    nodeList.append(targetNd)
    nxSub = nexG.subgraph(nodeList)

    return nxSub

# # OneHopSubgph Ex
# nexG = networkXSet[0]
# targetNd = 25
# nxSub = OneHopSubgph(nexG, targetNd)
# imgShow(nxSub)

'''
    그래프 하나에 대한 K hop graph
    neighbor를 이용해 subgraph를 만들 nodeList 추출
'''
def TwoHopSubgph(nexG, targetNd) :

    nodeList = []
    oneHopNeighbor = list(nexG.neighbors(targetNd))
    nodeList.append(targetNd)
    nodeList += oneHopNeighbor

    for neiNodeId in oneHopNeighbor :
        nodeList += list(nexG.neighbors(neiNodeId))

    nodeList = list(set(nodeList))
    nxSub = nexG.subgraph(nodeList)

    return nxSub

# # TwoHopSubgph Ex
# nexG = networkXSet[0]
# targetNd = 8
# nxSub = TwoHopSubgph(nexG, targetNd)
# imgShow(nxSub)


'''
    KhopSubGph 만들기 전에 일단 한 그래프에 대해 1 또는 2hop graph 생성
'''
def subGsPerGph(nexG, khop) :
    subGList = []

    sortedNodes = sorted(nexG.nodes)
    if khop == 1:
        for targetNd in sortedNodes:
            subGList.append(OneHopSubgph(nexG, targetNd))
        return subGList
    elif khop == 2:
        for targetNd in sortedNodes:
            subGList.append(TwoHopSubgph(nexG, targetNd))
        return subGList
    else:
        for targetNd in sortedNodes:
            subGList.append(KHopSubgph(nexG, targetNd,khop))
        return subGList


# # subGsPerGph Ex
# nexG = networkXSet[0]
# hop = 2
# subGList = subGsPerGph(nexG, hop)
# print(subGList[0])
# print(subGList[0].nodes(data=True))
# imgShow(subGList[0])
# imgShow(subGList[1])

'''
    subgraph의 node가 10개 이상인 것과 아닌 것 리스트 반환
'''
def split10Nd(subGList) :
    max9nodes = []
    upper10 = []
    for subG in subGList:
        if len(subG.nodes) >= 10:
            upper10.append(subG)
        else:
            max9nodes.append(subG)
    return max9nodes, upper10

# # split10Nd Ex
# subGList = subGsPerGph(nexG, hop)
# max9nodes, upper10 = split10Nd(subGList)
# print('subGList : ', len(subGList))
# print('upper10 : ', len(upper10))
# print('max9nodes : ', len(max9nodes))
# print('')
# print('upper10[0].nodes : ',len(upper10[0].nodes))
# print('max9nodes[0].nodes : ',len(max9nodes[0].nodes))


'''
    모든 노드에 대해서 subgraph를 추출하는게 아닌
    전체 노드개수의 절반만 target node로 삼아서 subgraph로 만들고, 
    이때 target node는 random하게 추출
'''
import random

def siftTargetNd(nexG, khop) :
    subGList = []
    nodesNum = len(nexG.nodes)
    sortedNodes = [random.randint(0,nodesNum) for value in range(0, nodesNum//2)]

    if khop == 1:
        for targetNd in sortedNodes:
            subGList.append(OneHopSubgph(nexG, targetNd))
        return subGList
    elif khop == 2:
        for targetNd in sortedNodes:
            subGList.append(TwoHopSubgph(nexG, targetNd))
        return subGList
    else:
        for targetNd in sortedNodes:
            subGList.append(KHopSubgph(nexG, targetNd,khop))
        return subGList

# # siftTargetNd Ex
# nexG = networkXSet[0]
# hop = 2
# subGList = siftTargetNd(nexG, hop)
# print('subgraph num : ', len(subGList))
# print(subGList[0])
# print(subGList[0].nodes(data=True))

#todo split upper 10 nodes Graph

# todo 전체 그래프에 대한 subgraph 생성 함수
'''
    
    dataset pickle 저장
    hop 수 1/2에 대해서 둘 다 한번에 저장? 따로 저장해서 걍 따로 쓰게..^^
    
    hop 수 parameter로 받아야함
'''
#parameter
nexG = networkXSet[0]
hop = 2





# nodeList = list(nexG.neighbors(targetNd))
# nodeList.append(targetNd)
# nxSub = nexG.subgraph(nodeList)


''' 
    target node를 대상으로, hop 기반의 random subgraph 생성
    그래프 하나를 기준으로, 타겟 노드 1개에 대해 1홉
    -> 이거 재귀적으로 쓸 생각.. 
    -> 재귀 풀어서 개선 가능한지 확인 필요
'''

#min/Max Nodes Num은 내부에서 random.randInt로 만들어줘도 될 듯. Max만 10아래면 될 일
#subGraph Node
def random1hopSubgraph(nexG, targetNd, minNodes, maxNodes):
    nodeList = list(nexG.neighbors(targetNd))
    nodeList.append(targetNd)
    nodeList = sorted(nodeList)

    # 10보다 노드 리스트의 수가 적은 경우, 노드 리스트 수로 최대 노드 개수를 변경함
    # 근데 이 최대 노드 개수가 min 값보다 작은 경우, re
    maxNodes = min(len(nodeList), maxNodes)  # 지정한 노드 수보다 neighbor가 더 적을 경우 노드 최대 개수를 neighbor 수로 변경
    # neighbor가 매우 적을 경우(3개 이하) 그냥 넘김
    if minNodes > len(nodeList):
        print('minNodes : ', minNodes)
        print('maxNodes : ', maxNodes)
        return
    else :
        subNodeNum = random.randint(minNodes, maxNodes) #서브그래프의 노드 개수를 랜덤하게 지정
        # random.sample은 중복 허용 안됨
        siftedNodes = random.sample(nodeList, subNodeNum)
        siftedNodes.append(targetNd)
        nxSub = nexG.subgraph(siftedNodes) #랜덤한 노드 수를 갖는 타겟 노드와 1홉인 subgraph
        return nxSub

'''
    2hop 기준 random subgraph
'''
def random2hopSubgraph(nexG, targetNd, minNodes, maxNodes):

    nodeList = list(nexG.neighbors(targetNd))
    nodeList.append(targetNd)
    #subgraph가 될 nodeList 갱신

    newTgNd = random.choice(nodeList)
    newNeighborList = list(nexG.neighbors(newTgNd))
    khopSubNodeList = list(set(nodeList + newNeighborList))
    khopSubNodeList = sorted(khopSubNodeList)

    # 10보다 노드 리스트의 수가 적은 경우, 노드 리스트 수로 최대 노드 개수를 변경함
    # 근데 이 최대 노드 개수가 min 값보다 작은 경우, re
    maxNodes = min(len(khopSubNodeList), maxNodes)  # 지정한 노드 수보다 neighbor가 더 적을 경우 노드 최대 개수를 neighbor 수로 변경
    # neighbor가 매우 적을 경우(3개 이하) 그냥 넘김
    if minNodes > len(khopSubNodeList):
        print('minNodes : ', minNodes)
        print('maxNodes : ', maxNodes)
        return
    else :
        subNodeNum = random.randint(minNodes, maxNodes) #서브그래프의 노드 개수를 랜덤하게 지정
        # random.sample은 중복 허용 안됨

        siftedNodes = random.sample(khopSubNodeList, subNodeNum)
        nxSub = nexG.subgraph(siftedNodes) #랜덤한 노드 수를 갖는 타겟 노드와 1홉인 subgraph
        return nxSub



'''
    neighbor List를 받아서 처리하면 될 듯?
   k-hop 기준으로
'''
def randomKhopSubgraph(nexG, targetNd, minNodes, maxNodes, khop):

    nodeList = list(nexG.neighbors(targetNd))
    nodeList.append(targetNd)
    #subgraph가 될 nodeList 갱신
    targetNdList = []
    targetNdList.append(targetNd) #맨 처음 타겟 노드
    khopSubNodeList = []
    khopSubNodeList += nodeList

    for i in range(khop-1) :
        newTgNd = random.choice(khopSubNodeList) # n홉의 n번째 targetnode
        targetNdList.append(newTgNd)
        newNeighborList = list(nexG.neighbors(newTgNd))
        khopSubNodeList += newNeighborList
        khopSubNodeList = list(set(khopSubNodeList))
        khopSubNodeList = sorted(khopSubNodeList)

    targetNdList = list(set(targetNdList)) #targetnode 중복 제거

    # [x for x in a if x not in b]
    # 10보다 노드 리스트의 수가 적은 경우, 노드 리스트 수로 최대 노드 개수를 변경함
    # 근데 이 최대 노드 개수가 min 값보다 작은 경우, re
    maxNodes = min(len(khopSubNodeList), maxNodes)  # 지정한 노드 수보다 neighbor가 더 적을 경우 노드 최대 개수를 neighbor 수로 변경
    # neighbor가 매우 적을 경우(3개 이하) 그냥 넘김
    if minNodes > len(khopSubNodeList):
        print('minNodes : ', minNodes)
        print('maxNodes : ', maxNodes)
        return
    else :
        subNodeNum = maxNodes-len(targetNdList)
        #subNodeNum = random.randint(minNodes, maxNodes)-len(targetNdList)
        if subNodeNum < 1 :
            subNodeNum = 10-len(targetNdList)
        #subNodeNum = random.randint(minNodes, maxNodes)-len(targetNdList)

        print('minNodes : ', minNodes)
        print('maxNodes : ', maxNodes)


        # targenNode는 무조건 들어가야하니까 개수 제외하고,
        #서브그래프의 노드 개수를 랜덤하게 지정
        # random.sample은 중복 허용 안됨
        # targetNode들과 targetNode의 neighbor 중 랜덤으로 추출된 노드들이 subgraph의 Node가 되도록 추가
        print('subNodeNum : ', subNodeNum)
        print('khopSubNodeList : ',khopSubNodeList)
        print('targetNdList : ', targetNdList)
        #d = random.sample(khopSubNodeList, subNodeNum)
        #khopSubNodeList에 targetNode가 없어야함
        khopSubNodeList = [x for x in khopSubNodeList if x not in targetNdList]

        print('khopSubNodeList : ',khopSubNodeList)

        subNodeNum = min(len(khopSubNodeList), subNodeNum)

        d = random.sample(khopSubNodeList, subNodeNum)
        siftedNodes = targetNdList + d
        #siftedNodes = random.sample(khopSubNodeList, subNodeNum) + targetNdList
        nxSub = nexG.subgraph(siftedNodes) #랜덤한 노드 수를 갖는 타겟 노드와 1홉인 subgraph
        return nxSub



'''
    전체 노드에 대해서 neighbor 개수가 min(=random)/max(=10)인 subgraph, 
    단 1/2hop을 우선으로 만들고, 
    1홉과 2홉이상의 neighbor List 갱신은 다름
'''
#RandomSubgph 생성 / 일단 1/2hop 그래프 생성

def RandomSubgph(nexG, minNodes, maxNodes,khop) :
    subGList = []
    nexGNodes = list(sorted(nexG.nodes))
    targetNdList = []
    if khop == 1:
        for targetNd in nexGNodes:
            subG = random1hopSubgraph(nexG, targetNd,minNodes, maxNodes)
            if subG != None :
                subGList.append(subG)
                targetNdList.append(targetNd)
        return subGList
    else :
        for targetNd in nexGNodes:
            print('targetNd : ', targetNd)
            subG = randomKhopSubgraph(nexG, targetNd, minNodes, maxNodes,khop)
            if subG != None:
                subGList.append(subG)
                targetNdList.append(targetNd)
        return targetNdList, subGList



# 노드 당 최대 khop 의 노드 개수를 갖는 randomwalk subgraph
def RWKhopSubgraph(nexG, minNodes, maxNodes, khop):
    subGList = []
    targetNdList = []
    nexGNodes = list(sorted(nexG.nodes))
    for targetNd in nexGNodes:
        print(targetNd)
        targetNdList1Sub = []
        originTId = targetNd
        targetNdList1Sub.append(targetNd)
        # hop 수 지정
        for i in range(khop):
            exListNum = len(targetNdList1Sub)
            nodeList = list(nexG.neighbors(targetNd))
            targetNd = random.choice(nodeList)
            targetNdList1Sub = list(set(targetNdList1Sub))

            targetNdList1Sub.append(targetNd)

            # 리프 노드인 경우
            # err : 노드 두 개에서 왔다갔다 하는 경우 : ex target == new target가 같을 때로 식별 가능
            if exListNum == len(targetNdList1Sub):
                #nxSub = nexG.subgraph(targetNdList1Sub)
                subGList.append(nexG.subgraph(targetNdList1Sub))
                targetNdList.append(originTId)
                break
                # 노드 개수가 maxNodes(=10)개 인 경우
            if len(targetNdList1Sub) == maxNodes:
                nxSub = nexG.subgraph(targetNdList1Sub)
                subGList.append(nexG.subgraph(targetNdList1Sub))
                targetNdList.append(originTId)
                break

    return targetNdList, subGList




minNodes = 2
maxNodes = 10
khop = 5
targetNd = 28
#nexG 그래프 1개에서 targetNd 2번을 기준으로 3개에서 10개 사이의 노드 수를 갖는 1hop sub graph
targetNdList, subGList = RWKhopSubgraph(nexG, minNodes, maxNodes,khop)

print(targetNdList[0])
print(subGList[0])
print(subGList[0].nodes(data=True))
imgShow(subGList[0])

print(targetNdList[1])
print(subGList[1])
print(subGList[1].nodes(data=True))
imgShow(subGList[1])


print(targetNdList[2])
print(subGList[2])
print(subGList[2].nodes(data=True))
imgShow(subGList[2])
