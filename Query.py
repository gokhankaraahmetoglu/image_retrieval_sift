import cv2
import matplotlib.pyplot as plt
import pickle
from computeFeatures import computeFeatures, computeFeatures_baseline
from scipy.cluster.vq import vq
import numpy as np
from computeDistances import computeDistances
import matplotlib.image as mpimg


class Query:
    def read_query(self, queryfile):
        # read query image file
        img = cv2.imread(queryfile)
        query_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(query_img), plt.title("Query İmage")
        return query_img

    # Bag-of-word Features
    def compute_bow_features(self, query_img, queryfile,nearest_ids,closest_dists):
        # load pickled features
        fv = pickle.load(open("bow.pkl", "rb"))
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

        # access distances of all images from query image (first image), sort them asc
        nearest_idx = np.argsort(D[0, :]);
        nearest_ids.append(nearest_idx[1])
        closest_distance1 = D[0][nearest_idx[1]]
        closest_dists.append(closest_distance1)



    # TD-IDF Features
    def compute_tfidf_features(self, queryfile, nearest_ids , closest_dists):
        # load pickled features
        fv = pickle.load(open("tfidf.pkl", "rb"))
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
        nearest_ids.append(nearest_idx[1])
        closest_distance2 = D[0][nearest_idx[1]]
        closest_dists.append(closest_distance2)


    # Baseline Features
    def compute_baseline_features(self, queryfile, nearest_ids, closest_dists):

        # load pickled features
        fv = pickle.load(open("base.pkl", "rb"))
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

        # access distances of all images from query image (first image), sort them asc
        nearest_idx = np.argsort(D[0, :])
        nearest_ids.append(nearest_idx[1])
        closest_distance3 = D[0][nearest_idx[1]]
        closest_dists.append(closest_distance3)
        print(nearest_idx)
        print(nearest_ids)
        print(closest_dists)
