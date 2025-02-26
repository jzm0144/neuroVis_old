"""
Created on Sunday Jan 9th 19:31:41 2020
@author: Janzaib Masood
"""
from __future__ import print_function
import warnings
warnings.simplefilter('ignore')
import argparse
import os, sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from math import *
import ipdb as ipdb
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
import innvestigate 
from Transformer import Transformer
from utils import *
from models import trainModel

parser = argparse.ArgumentParser()
parser.add_argument("ageMatchUnmatch", type=str, help = "Enter ageMatched or ageUnmatched")
parser.add_argument("dataset", type=str, help="Dataset you want to train on \nADNI, ABIDE, ADHD or PTSD?")
parser.add_argument("heatmapNumber", type=int, help="Generate heatmap for the exmaple id")
parser.add_argument("topPaths", type=int, help="Most significant Paths")
parser.add_argument("label", type=int, help="Generate heatmap for what class? ")
args = parser.parse_args()

print("The Dataset is                    = ", args.dataset)
print("The age                           = ", args.ageMatchUnmatch)
print("Generate Heatmap for Data Example = ", args.heatmapNumber)
print("Num of Paths in Heatmap           = ", args.topPaths)
print("Clamped Neuron                    = ", args.label)

path = os.getcwd()
trainPath = os.getcwd()+ "/Data/" +args.ageMatchUnmatch+"/"+ args.dataset + "_train_data.xlsx"
testPath  = os.getcwd()+ "/Data/" +args.ageMatchUnmatch+"/"+ args.dataset + "_test_data.xlsx"
clf = Transformer(trainPath, testPath, verbose=True)
xTrain, yTrain = clf.getTrainData()
xTest, yTest   = clf.getTestData()
idx, xPath, yPath = clf.getPaths()
codeLabels(yTrain = yTrain, yTest = yTest, disorder = args.dataset)

# Bring all Data in range 0.0 and 1.0
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain = (xTrain + 1)/2
xTest  = (xTest  + 1)/2

num_classes = len(np.unique(yTrain))
print("Number of Classes", num_classes)


# Input Image Dimensions (The reshaped reduced Connectivity Matrix)
img_rows, img_cols = xTrain.shape[1:]
print("Image Size = ("+str(img_rows)+", "+str(img_cols),")")
if K.image_data_format() == 'channels_first':
    xTrain = xTrain.reshape(xTrain.shape[0], 1, img_rows, img_cols)
    xTest  = xTest.reshape(xTest.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    xTrain = xTrain.reshape(xTrain.shape[0], img_rows, img_cols, 1)
    xTest  = xTest.reshape(xTest.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#Accuracy
# convert class vectors to binary class matrices
yTrain = keras.utils.to_categorical(yTrain, num_classes)
yTest = keras.utils.to_categorical(yTest, num_classes)

'''
trainModel(input_shape
          ,xTrain, yTrain
          ,xTest, yTest
          ,args.ageMatchUnmatch
          ,args.dataset
          ,num_classes)
'''

model = load_model('Models/'+ args.ageMatchUnmatch+"_"+args.dataset+'.h5')
# Strip softmax layer
model_no_softmax = innvestigate.utils.model_wo_softmax(model)



#Decide your inputs
inputs = xTest[:,:,:,:]
outs   = yTest[:]
preds = model.predict(inputs)
for i in range(len(preds)):
    print("Out= ",outs[i], "  Pred   ",preds[i])

input_max = 1.0
input_min = 0.0
noise_scale = (input_max - input_min) * 0.005

heatmaps = [
      # NAME                                  OPT.PARAMS                     TITLE
      # Show input.

      # Gradient Family        
      ("gradient",                      {}                                 ,  "Gradient"),
      #("smoothgrad",                    {"augment_by_n": 1,
      #                                   "noise_scale": noise_scale,
      #                                   "postprocess": "square"}          ,  "SmoothGrad"),
      #("input_t_gradient",              {}                                 ,  "Input * Gradient"),
      #("integrated_gradients",          {"reference_inputs": input_min,
      #                                   "steps": 16}                      ,  "Integrated Gradients"),

      #("deconvnet",                     {}                                 ,  "Deconvnet"),
      #("guided_backprop",               {}                                 ,  "Guided Backprop"),

      # LRP Family
      #("deep_taylor.bounded",           {"low": input_min,
      #                                   "high": input_max}                ,  "DeepTaylor"),
      #("lrp.z",                         {}                                 ,  "LRP-Z"),
      #("lrp.epsilon",                   {"epsilon": 1}                     ,  "LRP-Epsilon"),
      #("lrp.sequential_preset_a_flat",  {"epsilon": 1}                     ,  "LRP-PresetAFlat"),
      #("lrp.sequential_preset_b_flat",  {"epsilon": 1}                     ,  "LRP-PresetBFlat"),

      # State of the Art Methods
      #("occlusion",                     {}                                 ,  "Occlusion Map"),
      #("lime",                          {}                                 ,  "Lime Method"),
      #("shapley",                       {}                                 ,  "Shapely Values"),
      #("Meaningful Perturbation",       {}                                 ,  "Meaningful Perturbation")
]



# The Node File
temp = {"ADNI":200, "ABIDE":200,"ADHD":190, "PTSD":125}
nodeFile = args.dataset + str(temp[args.dataset]) + ".node"

# ------------------------------  Part 1  ------------------------------------
# Generate Heatmaps of the Given Example Number
for heatmap in heatmaps:
    #Create the analyzers
    analyzer = innvestigate.create_analyzer(heatmap[0],
                                            model_no_softmax,
                                            neuron_selection_mode = "index",
                                            **heatmap[1])

    # Generate the heatmaps
    analysis = analyzer.analyze(inputs, args.label)

    thisHeatmap = analysis[args.heatmapNumber, :, :, 0]

    
    # Save the Heatmap Edge files
    edgeFilePath = saveEdgeFile(thisHeatmap,
                 idx,
                 heatmap[0],
                 str(args.label),
                 args.topPaths,
                 args.dataset,
                 xPath,
                 yPath,
                 map = "pos",
                 edgeDir = "Edge/Part1/",
                 exampleHNum = str(args.heatmapNumber),
                 predNeuron=str(np.argmax(preds[args.heatmapNumber])),
                 actualNeuron=str(np.argmax(outs[args.heatmapNumber])))


    plotBrainNet(nodePath = "Node2/"+nodeFile,
                 edgePath = edgeFilePath,
                 outputPath = 'Results/Part1/'+args.dataset,
                 configFile = 'config.mat')
# ----------------------------------------------------------------------------
'''
# ------------------------------  Part 2  ------------------------------------
# -------------- Avg of Each Heatmap Method For All Examples -----------------
# ----------------- Edges saved in Directory Edge/Part2 ----------------------

for heatmap in heatmaps:
    #Create the analyzers
    analyzer = innvestigate.create_analyzer(heatmap[0],
                                            model_no_softmax,
                                            neuron_selection_mode = "index",
                                            **heatmap[1])    
    # Generate the heatmaps
    analysis = analyzer.analyze(inputs, args.label)

    # mean heatmap
    meanHeatmap = avgMap(analysis[:,:,:,0])


    # Save the Mean Heatmap Edge files
    edge = saveEdgeFile(img = meanHeatmap,
                         idx = idx,
                         heatmap_method = "mean_"+heatmap[0],
                         clampedNeuron = str(args.label),
                         topPaths = args.topPaths,
                         dataset = args.dataset,
                         xPath = xPath,
                         yPath = yPath,
                         map = "pos",
                         edgeDir = "Edge/Part2/")

    # Also save the BrainNet png files
    plotBrainNet(nodePath = "Node2/"+nodeFile,
                 edgePath = edge,
                 outputPath = 'Results/Part2/'+args.dataset,
                 configFile = 'config.mat')

# ----------------------------------------------------------------------------
# ------------------------------  Part 3  ------------------------------------
# -------------- Avg of All Heatmaps For the Same Example --------------------
# ----------------- Edges saved in Directory Edge/Part3 ----------------------
mapFrame = np.zeros((len(heatmaps),inputs.shape[0],inputs.shape[1],inputs.shape[2]))


for index, heatmap in enumerate(heatmaps):
    #Create the analyzers
    analyzer = innvestigate.create_analyzer(heatmap[0],
                                            model_no_softmax,
                                            neuron_selection_mode = "index",
                                            **heatmap[1])    
    # Generate the heatmaps
    analysis = analyzer.analyze(inputs, args.label)

    # Heatmaps of this current method are put in this spot of the mapFrame
    mapFrame[index,:,:,:] = analysis[:,:,:,0]


meanHeatmap_for_given_example = np.mean(mapFrame[:,args.heatmapNumber, :, :], axis=0)

# Save the Mean Heatmap Edge files
edge = saveEdgeFile(img = meanHeatmap_for_given_example,
                     idx = idx,
                     heatmap_method = "meanHeatmap",
                     clampedNeuron = str(args.label),
                     topPaths = args.topPaths,
                     dataset = args.dataset,
                     xPath = xPath,
                     yPath = yPath,
                     map = "pos",
                     edgeDir = "Edge/Part3/",
                     exampleHNum = str(args.heatmapNumber))

# Also save the BrainNet png files
plotBrainNet(nodePath = "Node2/"+nodeFile,
             edgePath = edge,
             outputPath = 'Results/Part3/'+args.dataset,
             configFile = 'config.mat')
# ----------------------------------------------------------------------------


# ------------------------------  Part 4  ------------------------------------
# --------- step1: Calc all Heatmaps for the Same Example --------------------
# --------- step2: Sparsify: Let only top-X paths pass thruough --------------
# --------- step3: Calc Binary Intersection of path occurences ---------------
# --------- step4: Calc Mean and Element-wise multiply with Binary Intersetion



mapFrame = np.zeros((len(heatmaps),inputs.shape[0],inputs.shape[1],inputs.shape[2]))
X = 150

for index, heatmap in enumerate(heatmaps):
    #Step1: Calc all Heatmaps for the same example  
    analyzer = innvestigate.create_analyzer(heatmap[0],
                                            model_no_softmax,
                                            neuron_selection_mode = "index",
                                            **heatmap[1])    
    analysis = analyzer.analyze(inputs, args.label)

    #Step2: Sparsify: Let only top-X paths pass through
    Maps = analysis[:,:,:,0].copy()
    for __ in range(Maps.shape[0]):
        Maps[__,:,:] = pass_topX_2D(Maps[__,:,:], X, verbose=False)
    mapFrame[index, :, :, :] = Maps.copy()
BinaryMask = mapFrame.copy()
BinaryMask[BinaryMask > 0] = 1


#Step3: Calc Binary Intersection of path occurrences
intMask = np.ones(BinaryMask.shape[1:])
for __ in range(BinaryMask.shape[0]):
    intMask = np.multiply(intMask[:,:,:], BinaryMask[__,:,:,:])

#Step4: Calc Mean and Element-wise multiply with Binary Intersection
#FinalHeatmaps = np.multiply(intMask, np.mean(mapFrame, axis=0))


# Save the Mean Heatmap Edge files
edge = saveEdgeFile(img = intMask[args.heatmapNumber,:,:],#FinalHeatmaps[args.heatmapNumber,:,:],
                     idx = idx,
                     heatmap_method = "intHeatmap",
                     clampedNeuron = str(args.label),
                     topPaths = args.topPaths,
                     dataset = args.dataset,
                     xPath = xPath,
                     yPath = yPath,
                     map = "pos",
                     edgeDir = "Edge/Part4/",
                     exampleHNum = str(args.heatmapNumber))


# Also save the BrainNet png files
plotBrainNet(nodePath = "Node2/"+nodeFile,
             edgePath = edge,
             outputPath = 'Results/Part4/'+args.dataset,
             configFile = 'config.mat')

# ----------------------------------------------------------------------------
'''
