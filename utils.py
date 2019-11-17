import argparse
import pandas as pd
import numpy as np


def codeLabels(yTrain, yTest, disorder): #ABIDE, ADHD, PTSD, ADNI
    if disorder == "ADNI":
        yTrain[yTrain == 'Controls'] = 0
        yTrain[yTrain == 'EMCI']     = 1
        yTrain[yTrain == 'LMCI']     = 2
        yTrain[yTrain == 'AD']       = 3

        yTest[yTest   == 'Controls'] = 0
        yTest[yTest   == 'EMCI']     = 1
        yTest[yTest   == 'LMCI']     = 2
        yTest[yTest   == 'AD']       = 3
    if disorder == "ADHD":
        yTrain[yTrain == 'Controls'] = 0
        yTrain[yTrain == 'ADHD-C']   = 1
        yTrain[yTrain == 'ADHD-H']   = 2
        yTrain[yTrain == 'ADHD-I']   = 3

        yTest[yTest   == 'Controls'] = 0
        yTest[yTest   == 'ADHD-C']   = 1
        yTest[yTest   == 'ADHD-H']   = 2
        yTest[yTest   == 'ADHD-I']   = 3
    if disorder == "ABIDE":
        yTrain[yTrain == 'Controls'] = 0
        yTrain[yTrain == 'Aspergers']= 1
        yTrain[yTrain == 'Autism']   = 2

        yTest[yTest   == 'Controls'] = 0
        yTest[yTest   == 'Aspergers']= 1
        yTest[yTest   == 'Autism']   = 2
    if disorder == "PTSD":
        yTrain[yTrain == 'Controls'] = 0
        yTrain[yTrain == 'PCS_PTSD'] = 1
        yTrain[yTrain == 'PTSD']     = 1

        yTest[yTest   == 'Controls'] = 0
        yTest[yTest   == 'PCS_PTSD'] = 1
        yTest[yTest   == 'PTSD']     = 1



def square2Vec(img, vecLength):
    imgVec = img.flatten()
    vec = imgVec[:vecLength]
    return vec

def vector2Conn(indices, xPath, yPath, disorder, strengthVector):
    matSize = {"ADNI":200, "ABIDE":200,"ADHD":190, "PTSD":125}
    
    M = np.zeros((matSize[disorder], matSize[disorder]))
    for __ in indices:
        index = __ - 1

        M[xPath[index]-1, yPath[index]-1] = strengthVector[index]
    return M

#def passTopPaths(connVec, top = args.topPaths):
def passTopPaths(connVec, top):   
    length = connVec.shape[0]

    temp = np.zeros(connVec.shape)
    indices = np.argsort(connVec)

    for i in range(length-top, length):
        ind = indices[i]
        temp[ind] = connVec[ind]

    for i in range(top):
        ind = indices[i]
        temp[ind] = connVec[ind]

    return temp

#def saveEdgeFile(img, idx, heatmap_method, clampedNeuron):
def saveEdgeFile(img, idx, heatmap_method, clampedNeuron, topPaths, dataset,xPath, yPath, map = "all"):

    vec = square2Vec(img, vecLength=len(idx))
    vec = passTopPaths(vec, top = topPaths)


    conn = vector2Conn(idx, xPath, yPath, disorder = dataset, strengthVector = vec)

    conn = np.transpose(conn)
    conns = list(conn)



    # Saving Negative and Postive Edges Together
    conn_str = ''
    for lconn in conns:
        for index in range(len(lconn)):
            item = lconn[index]
            item = abs(item)
            if item == 0.0: item = int(item)
            if index == 0:
                conn_str = conn_str + str(item)
            else:
                conn_str = conn_str +'\t'+ str(item)
        conn_str = conn_str + '\n'

       

    # Saving Separte Relevance Edges
    conn_str_pos = ''
    conn_str_neg = ''

    for lindex in range(len(conns)):
        lconn = list(conns[lindex])

        for index in range(len(lconn)):
            item = lconn[index]

            if item == 0.0:
                item = int(item)
                if index == 0:
                    conn_str_pos = conn_str_pos + str(item)
                    conn_str_neg = conn_str_neg + str(item)
                else:
                    conn_str_pos = conn_str_pos + '\t'+ str(item)
                    conn_str_neg = conn_str_neg + '\t'+ str(item)


            else:
                if index == 0:
                    if item > 0:
                        conn_str_pos = conn_str_pos + str(item)
                        conn_str_neg = conn_str_neg + '0'
                    if item < 0:
                        conn_str_pos = conn_str_pos + '0'
                        conn_str_neg = conn_str_neg + str(-1*item)
                if index != 0:
                    if item > 0:
                        conn_str_pos = conn_str_pos +'\t'+ str(item)
                        conn_str_neg = conn_str_neg +'\t'+ '0'
                    if item < 0:
                        conn_str_pos = conn_str_pos +'\t'+ '0'
                        conn_str_neg = conn_str_neg +'\t'+ str(-1*item)

        conn_str_pos = conn_str_pos + '\n'
        conn_str_neg = conn_str_neg + '\n'

    if map == "abs":
        file1 = open('Edge/'+ heatmap_method +'_'+ dataset+'_l'+clampedNeuron+'.edge', 'w')
        file1.write(conn_str)
        file1.close()
    elif map == "pos":
        file2 = open('Edge/'+ heatmap_method +'_'+ dataset+'_l'+clampedNeuron+'_pos.edge', 'w')
        file2.write(conn_str_pos)
        file2.close()
    elif map == "neg":
        file3 = open('Edge/'+ heatmap_method +'_'+ dataset+'_l'+clampedNeuron+'_neg.edge', 'w')
        file3.write(conn_str_neg)
        file3.close()
    elif map == "all":
        file1 = open('Edge/'+ heatmap_method +'_'+ dataset+'_l'+clampedNeuron+'.edge', 'w')
        file1.write(conn_str)
        file1.close()
        file2 = open('Edge/'+ heatmap_method +'_'+ dataset+'_l'+clampedNeuron+'_pos.edge', 'w')
        file2.write(conn_str_pos)
        file2.close()
        file3 = open('Edge/'+ heatmap_method +'_'+ dataset+'_l'+clampedNeuron+'_neg.edge', 'w')
        file3.write(conn_str_neg)
        file3.close()


def avgMap(allTestMaps):
    heatmap = np.mean(allTestMaps, axis = 0)
    return heatmap


