# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:11:28 2015

@author: mah228
"""

###test code for image classification ##

### image split
## 
## install PIL in conda: conda install PIL

import Image
import os
im = Image.open("H:/building-energy-paper/elsarticle/elsarticle/fig/fig2.png")

def imgCrop(im):
    
    box = (0, 0, 1000, 1000)
    region = im.crop(box)
    path="H:/building-energy-paper/elsarticle/elsarticle/fig/cropped.png"
    region.save(path)
    
    
imgCrop(im)




def crop(p1,height,width,i):
    im = Image.open(p1)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            #k=0
            #path="H:/building-energy-paper/elsarticle/elsarticle/fig/path="H:/building-energy-paper/elsarticle/elsarticle/fig/cropped"+str(i)+".png""
            path="H:/building-energy-paper/elsarticle/elsarticle/fig/Presentation1/cropped"+str(i)+str(j)+".png"
            #k = k+1
            a.save(path)
            
            
def threshold(imageArray):
    balanceAr = []
    newAr = imageArray
    for eachRow in imageArray:
        for eachPix in eachRow:
            avgNum = reduce(lambda x, y: x + y, eachPix[:3]) / len(eachPix[:3])
            balanceAr.append(avgNum)
            balance = reduce(lambda x, y: x + y, balanceAr) / len(balanceAr)
    return balance

 

def threshold(imageArray):
    balanceAr = []
    newAr = imageArray
    for eachRow in imageArray:
        for eachPix in eachRow:
            avgNum = reduce(lambda x, y: x + y, eachPix[:3]) / len(eachPix[:3])
            balanceAr.append(avgNum)
    balance = reduce(lambda x, y: x + y, balanceAr) / len(balanceAr)
    for eachRow in newAr:
        for eachPix in eachRow:
            if reduce(lambda x, y: x + y, eachPix[:3]) / len(eachPix[:3]) > balance:
                eachPix[0] = 255
                eachPix[1] = 255
                eachPix[2] = 255
                eachPix[3] = 255
            else:
                eachPix[0] = 0
                eachPix[1] = 0
                eachPix[2] = 0
                eachPix[3] = 255
    return newAr           
        
import numpy
import matplotlib
matplotlib.use('Agg')
from scipy.cluster.vq import *
import pylab
pylab.close()
 
# generate 3 sets of normally distributed points around
# different means with different variances
pt1 = numpy.random.normal(1, 0.2, (100,2))
pt2 = numpy.random.normal(2, 0.5, (300,2))
pt3 = numpy.random.normal(3, 0.3, (100,2))
 
# slightly move sets 2 and 3 (for a prettier output)
pt2[:,0] += 1
pt3[:,0] -= 0.5
 
xy = numpy.concatenate((pt1, pt2, pt3))
 
# kmeans for 3 clusters
res, idx = kmeans2(numpy.array(zip(xy[:,0],xy[:,1])),3)
 
colors = ([([0.4,1,0.4],[1,0.4,0.4],[0.1,0.8,1])[i] for i in idx])
 
# plot colored points
pylab.scatter(xy[:,0],xy[:,1], c=colors)
 
# mark centroids as (X)
pylab.scatter(res[:,0],res[:,1], marker='o', s = 500, linewidths=2, c='none')
pylab.scatter(res[:,0],res[:,1], marker='x', s = 500, linewidths=2)
pylab.savefig('H:/building-energy-paper/elsarticle/elsarticle/fig/images.jpg')
