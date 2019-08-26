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
    def compute_bow_features(self, query_img, queryfile):
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
        closest_distance1 = D[0][nearest_idx[1]]
        print(nearest_idx)
        img1 = mpimg.imread(queryfile)
        img2 = mpimg.imread("images/" + str(nearest_idx[1]) + ".jpg")
        fig = plt.figure()
        a = fig.add_subplot(1, 4, 1)
        imgplot_1 = plt.imshow(img1)
        a.set_title('Query')
        a = fig.add_subplot(1, 4, 2)
        plt.xlabel('Distance: ' + str(closest_distance1))
        imgplot = plt.imshow(img2)
        a.set_title('Bow  Closest')
        return fig

    # TD-IDF Features
    def compute_tfidf_features(self, queryfile, fig):
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
        closest_distance2 = D[0][nearest_idx[1]]
        print(nearest_idx)
        img3 = mpimg.imread("images/" + str(nearest_idx[1]) + ".jpg")
        a = fig.add_subplot(1, 4, 3)
        plt.xlabel('Distance: ' + str(closest_distance2))
        imgplot_2 = plt.imshow(img3)
        a.set_title('Tf-idf  Closest')
        return fig

    # Baseline Features
    def compute_baseline_features(self, queryfile, fig):
        """

        :param queryfile:
        :param fig:
        :return:
        """
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
        nearest_idx = np.argsort(D[0, :]);
        closest_distance3 = D[0][nearest_idx[1]]
        print(nearest_idx)
        img4 = mpimg.imread("images/" + str(nearest_idx[1]) + ".jpg")
        a = fig.add_subplot(1, 4, 4)
        imgplot_3 = plt.imshow(img4)
        a.set_title('Baseline  Closest')
        plt.xlabel('Distance: ' + str(closest_distance3))
        fig.set_size_inches((12, 12), forward=False)
        plt.savefig("results/mm_model13.png", format="png")
        plt.show()
