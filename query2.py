"""
queryEval.py

DO NOT MODIFY ANY CODES IN THIS FILE
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED!
I DON'T CARE YOUR CODE SO BUGGY


@author: John See, 2017
@modified by: Lai Kuan, Wong, 2018
@fixed by: Kun Shun, Goh, 2018

"""
import os
import cv2
import numpy as np
import pickle
import sys, getopt
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq
from computeFeatures import computeFeatures, computeFeatures_baseline
from computeDistances import computeDistances
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO

# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
# dbpath = 'C:\\Users\\aquas\\Documents\\VIP\\as2\\plantdb'
dbpath = '/home/gkhnkrhmt/Desktop/images'
# dbpath= 'C:\\Users\\Koh\\Desktop\\1151101808_Assignment-2\\plantdb\\train'
queryfile = "/home/gkhnkrhmt/Desktop/Resim/12.jpg"

# create a arrays for precision and recall
precision_bow = []
precision_tfidf = []
precision_base = []
recall_bow = []
recall_tfidf = []
recall_base = []

# read query image file
img = cv2.imread(queryfile)
query_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(query_img) , plt.title("Query İmage")



#====================================================================
# Bag-of-word Features
#====================================================================

featvect = []  # empty list for holding features
FEtime = np.zeros(500)

# load pickled features
fv = pickle.load(open("bow.pkl", "rb") )
print('BoW features loaded')

# Compute features
newfeat = computeFeatures(query_img)
# Load cookbook
codebook = pickle.load(open("codebook.pkl", "rb"))
code, distortion = vq(newfeat, codebook)
# Map features to label and obtain BoW
k = codebook.shape[0]
bow_hist, _ = np.histogram(code, k, normed=True)
# Update newfeat to BoW
newfeat = bow_hist


# insert new feat to the top of the feature vector stack
fv = np.insert(fv, 0, newfeat, axis=0)

# find all pairwise distances
D = computeDistances(fv)


# *** Evaluation ----------------------------------------------------------

# number of images to retrieve
nRetrieved = 50

# access distances of all images from query image (first image), sort them asc
nearest_idx = np.argsort(D[0, :]);
closest_distance1 = D[0][nearest_idx[1]]
print(nearest_idx)
img1 = mpimg.imread(queryfile)
img2 = mpimg.imread("/home/gkhnkrhmt/Desktop/images/"+str(nearest_idx[1])+".jpg")
fig = plt.figure()
a = fig.add_subplot(1, 4, 1)
imgplot_1 = plt.imshow(img1)
a.set_title('Query')
a = fig.add_subplot(1, 4, 2)
plt.xlabel('Distance: '+str(closest_distance1))
imgplot = plt.imshow(img2)
a.set_title('Bow  Closest')

#cv2.imshow("Query İmage ", img1)
#cv2.imshow("Bow Closest Image and Distance : "+str(closest_distance1), img2)
# quick way of finding category label for top K retrieved images
retrievedCats = np.uint8(np.floor(((nearest_idx[1:nRetrieved+1])-1)/50));

#for curve  500
retrievedCurve = np.uint8(np.floor(((nearest_idx[1:500+1])-1)/50));



#====================================================================
# TD-IDF Features
#====================================================================

featvect = []  # empty list for holding features
FEtime = np.zeros(500)

# load pickled features
fv = pickle.load(open("tfidf.pkl", "rb") )
print('TF-IDF features loaded')

# read query image file
img = cv2.imread(queryfile)
query_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



# Compute features
newfeat = computeFeatures(query_img)
# Load cookbook
codebook = pickle.load(open("codebook.pkl", "rb"))
code, distortion = vq(newfeat, codebook)
# Map features to label and obtain BoW
k = codebook.shape[0]
bow_hist, _ = np.histogram(code, k, normed=True)
# Update newfeat to BoW
newfeat = bow_hist


# insert new feat to the top of the feature vector stack
fv = np.insert(fv, 0, newfeat, axis=0)

# find all pairwise distances
D = computeDistances(fv)

nearest_idx = np.argsort(D[0, :]);
closest_distance2 = D[0][nearest_idx[1]]
print(nearest_idx)
img3 = mpimg.imread("/home/gkhnkrhmt/Desktop/images/"+str(nearest_idx[1])+".jpg")
a = fig.add_subplot(1, 4, 3)
plt.xlabel('Distance: '+str(closest_distance2))
imgplot_2 = plt.imshow(img3)
a.set_title('Tf-idf  Closest')
#cv2.imshow("Tf-idf Closest Image and Distance : "+str(closest_distance2), img2)
# quick way of finding category label for top K retrieved images
retrievedCats = np.uint8(np.floor(((nearest_idx[1:nRetrieved+1])-1)/50));

#for curve  500
retrievedCurve = np.uint8(np.floor(((nearest_idx[1:500+1])-1)/50));

#====================================================================
# Baseline Features
#====================================================================

featvect = []  # empty list for holding features
FEtime = np.zeros(500)

# load pickled features
fv = pickle.load(open("base.pkl", "rb") )
print('Baseline features loaded')

# read query image file
img = cv2.imread(queryfile)
query_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Compute features
newfeat = computeFeatures_baseline(query_img)

# insert new feat to the top of the feature vector stack
fv = np.insert(fv, 0, newfeat, axis=0)

# find all pairwise distances
D = computeDistances(fv)



nRetrieved = 50

# access distances of all images from query image (first image), sort them asc
nearest_idx = np.argsort(D[0, :]);
closest_distance3 = D[0][nearest_idx[1]]
print(nearest_idx)
img4 = mpimg.imread("/home/gkhnkrhmt/Desktop/images/"+str(nearest_idx[1])+".jpg")
a = fig.add_subplot(1, 4, 4)
imgplot_3 = plt.imshow(img4)
a.set_title('Baseline  Closest')
plt.xlabel('Distance: '+str(closest_distance3))
plt.savefig("mm_model1.png")
plt.show()
#cv2.imshow("Baseline Features Closest Image and Distance : "+str(closest_distance3), img3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# quick way of finding category label for top K retrieved images
retrievedCats = np.uint8(np.floor(((nearest_idx[1:nRetrieved+1])-1)/50));

#for curve  500
retrievedCurve = np.uint8(np.floor(((nearest_idx[1:500+1])-1)/50));


