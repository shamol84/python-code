# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:16:52 2015

@author: mah228
"""

##image process 2

import Image
import PIL
from PIL import Image
import numpy as np
import pandas as pd




im1 = Image.open('H:/building-energy-paper/elsarticle/elsarticle/fig/images.jpg')
im2 = Image.open('H:/building-energy-paper/elsarticle/elsarticle/fig/images2.jpg')
im2 = Image.open('H:/building-energy-paper/elsarticle/elsarticle/fig/Presentation1/images2.jpg')
im2 = Image.open('H:/building-energy-paper/elsarticle/elsarticle/fig/Presentation1/images3.jpg')
im2 = Image.open('H:/building-energy-paper/elsarticle/elsarticle/fig/Presentation1/images4.jpg')
im2 = Image.open('H:/building-energy-paper/elsarticle/elsarticle/fig/Presentation1/images2.jpg')
im3 = Image.open('H:/building-energy-paper/elsarticle/elsarticle/fig/Presentation1/images3.jpg')
im4 = Image.open('H:/building-energy-paper/elsarticle/elsarticle/fig/Presentation1/images4.jpg')

###################
#setup a standard image size; this will distort some images but will get everything into the same shape
STANDARD_SIZE = (300, 167)
def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]
    
    #####################
pd1 = pd.DataFrame(img1)
##############

i = np.asarray(im1)
i2 = np.asarray(im2)
i3 = np.asarray(im3)
i4 = np.asarray(im4)

#y=threshold(i)
eiar =str(i.tolist())
ar=i.tolist()
pd1=pd.DataFrame(ar)
q=str(eiar)
pd2 = pd.DataFrame(q)


 lineToWrite = eiar+'\n'


y = reduce(lambda x,y:x+y,a)
#eiar = str(i.tolist())

pd1 = pd1.str.lstrip('[').str.rstrip(']')


numberArrayExamples = open('H:/building-energy-paper/elsarticle/elsarticle/fig/numArEx.txt','a')

numberArrayExamples.write(eiar)




for i in range(3):
    # select only data observations with cluster label == i
    ds = pd1[np.where(labels==i)]
    # plot the data observations
    pyplot.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    pyplot.setp(lines,ms=15.0)
    pyplot.setp(lines,mew=2.0)
pyplot.show()


############edge detection
from scipy import misc
fimg = misc.imread("H:/building-energy-paper/elsarticle/elsarticle/fig/images.jpg")

from skimage import color
gimg = color.colorconv.rgb2grey(fimg)



#from matplotlib import imshow
from skimage import measure
contours = measure.find_contours(gimg, 0.8)
import matplotlib.pyplot as plt
 
for n, contour in enumerate(contours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)