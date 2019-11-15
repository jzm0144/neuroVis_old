# Chane of Plan---- Lime does not make sense for Connectivity

from __future__ import print_function
import warnings
warnings.simplefilter('ignore')


import os, sys
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import lime
from lime import lime_image
import ipdb as ipdb

print('Currently Using:', keras.__version__)

inet_model = inc_net.InceptionV3()

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)


# Let's see the top 5 predictions for some image
images = transform_img_fn([os.path.join('cat_mouse.jpg')])
# I'm dividing by 2 and adding 0.5 because of how this Inception represents images
#plt.imshow(images[0] / 2 + 0.5)
preds = inet_model.predict(images)
for x in decode_predictions(preds)[0]:
    print(x)
#plt.show()


# Creating a lime explainer
explainer = lime_image.LimeImageExplainer()


# Hide color is the color for a superpixel turned OFF.
# Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
explanation = explainer.explain_instance(images[0],
                                         inet_model.predict,
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=10)


image, mask = explanation.get_image_and_mask(106,
                                            positive_only=False,
                                            num_features=10,
                                            hide_rest=False)
explanation.image[explanation.image < 0] = 0
explanation.image = 255*explanation.image
explanation.image = explanation.image.astype(int)

ipdb.set_trace()





