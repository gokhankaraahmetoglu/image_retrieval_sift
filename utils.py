import matplotlib.pyplot as plt
import matplotlib.image as mpimg

titles = ['Query', 'Bow  Closest', 'Tf-idf  Closest', 'Baseline  Closest']


def save_figs(queryfile, nearest_ids, closest_dists):
    fig = plt.figure()
    for i in range(3):
        if i == 0:
            img1 = mpimg.imread(queryfile)
            a = fig.add_subplot(1, 4, i + 1)
            plt.imshow(img1)
            a.set_title(titles[0])

        img2 = mpimg.imread("images/" + str(nearest_ids[i]) + ".jpg")
        a = fig.add_subplot(1, 4, i + 2)
        plt.xlabel('Distance: ' + str(closest_dists[i]))
        plt.imshow(img2)
        a.set_title(titles[i + 1])

    fig.set_size_inches((12, 12), forward=False)
    plt.savefig("results/mm_model15.png", format="png")
    plt.show()
