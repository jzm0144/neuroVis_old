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
