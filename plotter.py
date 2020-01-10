import numpy as np
import matplotlib.pyplot as plt
import ipdb as ipdb
import os

'''
directory = 'Results/Part1/PTSD/'

filenames = os.listdir(directory)


fig=plt.figure()
row = 3

for index, thisFile in enumerate(filenames):
    thisImg = plt.imread(directory + thisFile)

    col = index

    if index >= 4:
        row = 2
        col = index - 4
    if index >= 8:
        row = 3
        col = index - 8
    fig.add_subplot(row, 4 , index+1)

    plt.imshow(thisImg)
    plt.axis('off')
plt.savefig('test.png', bbox_inches='tight')

'''


import numpy as np
import bottleneck as bn

def pass_topX_2D(arr, X, verbose=False):
    idx = bn.argpartition(arr, arr.size-X, axis=None)[-X:]
    width = arr.shape[1]
    idx = [divmod(i, width) for i in idx]
    idx.sort(key = lambda tup: tup[0])
    if verbose == True:
        print("The sorted 2D indices = ", idx)
    mat = np.zeros(arr.shape)
    for item in idx:
        mat[item[0], item[1]] = arr[item[0], item[1]]
    return idx

np.random.seed(47)
arr = np.random.rand(5,5)
mat = pass_topX_2D(arr, 5, verbose=True)

print(arr)
print(mat) 