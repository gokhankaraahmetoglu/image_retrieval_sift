import argparse

from Query import Query

# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
# dbpath= 'C:\\Users\\Koh\\Desktop\\1151101808_Assignment-2\\plantdb\\train'
queryfile = ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default='data/12.jpg',
                        help='Path of the query image ')

    args = parser.parse_args()

    queryfile = args.db_path

    query = Query()

    query_img = query.read_query(queryfile)

    fig_a = query.compute_bow_features(query_img, queryfile)

    fig_a = query.compute_tfidf_features(queryfile, fig_a)

    query.compute_baseline_features(queryfile, fig_a)
    # ====================================================================

    # ====================================================================
