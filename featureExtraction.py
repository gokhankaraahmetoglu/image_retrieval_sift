import argparse

from Features import Features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default='images/',
                        help='Path of the image database')

    args = parser.parse_args()

    dbpath = args.db_path

    # List of features that stores
    feat = []
    base_feat = []

    features = Features()

    features.input_img(dbpath,feat,base_feat)

    features.compute_codebook(feat)

    features.compute_bow(feat)

    features.compute_tfidf()

    features.compute_baseline(base_feat)



