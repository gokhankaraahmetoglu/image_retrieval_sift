import pickle
from scipy.cluster.vq import kmeans, vq
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import cv2
from computeFeatures import computeFeatures, computeFeatures_baseline
import os
import matplotlib.pyplot as plt


class Features:
    def input_img(self, dbpath, feat, base_feat):
        for idx in range(500):
            # Load and convert image
            img = cv2.imread(os.path.join(dbpath, str(idx + 1) + ".jpg"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Compute SIFT features for each keypoints
            feat.append(computeFeatures(img))

            # Compute baseline features for each image
            base_feat.append(computeFeatures_baseline(img))

            print('Extracting features for image #%d' % idx)

    """
    feat.shape=1x500    , her bir elemanı keypointx128 lik dizidir. (her elemanın descriptorx128 'inden dolayı)
    base_feat.shape=1x500 , her bir elemanı 1x192 lik dizidir. (3 tnae 64lük histogramdan dolayı)
    """

    def compute_codebook(self, feat):
        """

        :param feat: SIFT's descriptors (feat.shape=1x500)
        :return:
        """
        # Stack all features together
        alldes = np.vstack(feat)
        """alldes.shape = 4410913x128 , tüm descriptorların sayısı x128 """
        k = 50
        # Perform K-means clustering
        alldes = np.float32(alldes)  # convert to float, required by kmeans and vq functions
        codebook, _ = kmeans(alldes, k)
        """codebook.shape=kx128 yani 50x128 oluyor. distortion.shape=1 (result döndürüyor?)
           her 128 sütun için 4410913 tane keypointten 50 merkezi nokta belirliyor."""
        # Save codebook as pickle file
        pickle.dump(codebook, open("codebook.pkl", "wb"))
        """vstack yaparak 4milyon x128 li bir shape elde ederiz . Bunu yapmamızın sebebi 4milyon yani tüm keypointler
        içerisinden k-means ile 50x128 tane centroid döndürmek . 
        vstack yapmazsak k-means 500x1 lik bir diziden 50x1 'lik shape döndürecektir . 500 tane içinden 50 tane centroid döndüre-
        cektir. 
        """

    # Bag-of-word Features
    def compute_bow(self, feat):
        """

        :param feat:
        :return:
        """
        # Create Bag-of-word list
        bow = []
        codebook = pickle.load(open("codebook.pkl", "rb"))
        k = codebook.shape[0]
        plot_it = True
        for f in feat:
            code, _ = vq(f, codebook)
            """f'teki descriptorlar gruplanıp code'a yazılır . code.shape=1xkeypoint_sayısı"""
            bow_hist, _ = np.histogram(code, k, normed=True)
            """gruplanmış descriptor'lar alınır ve histograma tabii tutulur. bow_hist.shape= bow_hist.shape=1x50"""
            bow.append(bow_hist)
            if plot_it == True:
                plot_it = False
                self.plot_histogram(bow_hist, k)

        """bow.shape=1x500"""
        # Stack them together
        all_bow = np.vstack(bow)

        # pickle your features (bow)
        pickle.dump(all_bow, open("bow.pkl", "wb"))
        print('Bag-of-words features pickled!')

    # TF-IDF Features
    def compute_tfidf(self):
        # first all_bow loading
        all_bow = pickle.load(open("bow.pkl", "rb"))
        # td-idf weighting
        transformer = TfidfTransformer(smooth_idf=True)
        t = transformer.fit_transform(all_bow).toarray()

        # normalize by Euclidean (L2) norm before returning
        t = normalize(t, norm='l2', axis=1)

        # pickle your features (tfidf)
        pickle.dump(t, open("tfidf.pkl", "wb"))
        print('TF-IDF features pickled!')

    # Baseline Features
    def compute_baseline(self, base_feat):
        base_feat = np.vstack(base_feat)  # shape=500x192

        # pickle your features (baseline)
        pickle.dump(base_feat, open("base.pkl", "wb"))
        print('Baseline features pickled!')

    def plot_histogram(self, bow_hist, k):

        bars = ['k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7', 'k=8', 'k=9', 'k=10', 'k=11', 'k=12', 'k=13', 'k=14',
                'k=15', 'k=16',
                'k=17', 'k=18', 'k=19', 'k=20', 'k=21', 'k=22', 'k=23', 'k=24', 'k=25', 'k=26', 'k=27', 'k=28', 'k=29',
                'k=30',
                'k=31', 'k=32', 'k=33', 'k=34', 'k=35', 'k=36', 'k=37', 'k=38', 'k=39', 'k=40', 'k=41', 'k=42', 'k=43',
                'k=44', 'k=45',
                'k=46', 'k=47', 'k=48', 'k=49', 'k=50']
        y = np.arange(len(bars))
        plt.bar(y, bow_hist, color='g')
        plt.xticks(y, bars)
        plt.savefig("results/histogram/normalized_hist1.png")
        plt.show()
