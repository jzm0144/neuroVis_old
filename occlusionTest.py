import copy
import numpy as np


def occlusionMap(image, model, occluding_size = 1, occluding_pixel = 0, occluding_stride = 1, class_label = 0):

    row = image.shape[0]
    col  = image.shape[1]

    out = model.predict(image)
    prediction_label =np.argmax(out)

    heatmap = np.zeros((row, col))
    
    for rowPix in range(row):
        for colPix in range(col):

            # Getting the image copy, applying the occluding window and classifying it again:
            input_image = copy.copy(image)
            input_image[rowPix, colPix] =  occluding_pixel            
            out = model.predict(im)
            prob = out[class_label]
            heatmap[rowPix, colPix] = prob
            print('scanning position (%s, %s)'%(h,w))
            heatmap[h,w] = prob

    return heatmap
    
Occlusion_exp(image_path, occluding_size, occluding_pixel, occluding_stride)