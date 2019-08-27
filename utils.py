import matplotlib.pyplot as plt
import matplotlib.image as mpimg
titles = ['Query','Bow  Closest','Tf-idf  Closest','Baseline  Closest']

def save_figs(queryfile,nearest_ids,closest_dists):
    for i in range(4):
        if i ==0:
            img1 = mpimg.imread(queryfile)
            fig = plt.figure()
            a = fig.add_subplot(1, 4, i+1)
            imgplot_1 = plt.imshow(img1)
            a.set_title(titles[i])
        else:
            img2 = mpimg.imread("images/" + str(nearest_ids[i-1]) + ".jpg")
            a = fig.add_subplot(1, 4, i+1)
            plt.xlabel('Distance: ' + str(closest_dists[i-1]))
            imgplot = plt.imshow(img2)
            a.set_title(titles[i])

    fig.set_size_inches((12, 12), forward=False)
    plt.savefig("results/mm_model14.png", format="png")
    plt.show()