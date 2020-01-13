"""
The most efficient would be to use OpenCV's
    cv::flann::hierarchicalClustering .
   But we do NOT have Python bindings to it.

See if you have the time:
    http://opencvpython.blogspot.ro/2013/01/k-means-clustering-3-working-with-opencv.html

Other ideas at:
    - https://stackoverflow.com/questions/1793532/how-do-i-determine-k-when-using-k-means-clustering
        - "Basically, you want to find a balance between two variables:
            the number of clusters (k) and the average variance of the
            clusters."
        - "First build a minimum spanning tree of your data. Removing the K-1
            most expensive edges splits the tree into K clusters, so you can
            build the MST once, look at cluster spacings / metrics for various
            K, and take the knee of the curve.
            This works only for Single-linkage_clustering, but for that it's
            fast and easy. Plus, MSTs make good visuals."

    - https://stackoverflow.com/questions/15376075/cluster-analysis-in-r-determine-the-optimal-number-of-clusters


https://en.wikipedia.org/wiki/Variance

NOT useful:
    http://classroom.synonym.com/calculate-average-variance-extracted-2842.html
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch

import common
import config

colors = ['b', 'r', 'g', 'y', "w", "magenta", "brown", "pink", "orange",
          "purple"]


def sqr(r):
    return r * r


"""
This uses SciPy - see scipy-ref.pdf, Section 3.1.1 .
I guess this is similar to cv::flann::hierarchicalClustering() .
Note: the number of clusters is dependent on the threshold var defined below.

IMPORTANT: We return only the elements from Z that are part of the MEANINGFUL
    clusters obtained with hierarchicalClustering().

See if you have the time:
  http://nbviewer.ipython.org/github/herrfz/dataanalysis/tree/master/data/
    http://nbviewer.ipython.org/github/herrfz/dataanalysis/blob/master/week4/clustering_example.ipynb
"""

# TODO: n is not used, remove or use.
def hierarchical_clustering(z, n):
    n = len(z)
    common.DebugPrint("hierarchical_clustering(): n = %d" % n)

    # Note: Z is not standard list, but a numpy array
    if len(z) < 10:  # or Z == []:
        common.DebugPrint("hierarchical_clustering(): Bailing out of "
                          "hierarchical clustering since too few elements "
                          "provided (and I guess we could have issues)")
        return []

    # Vector of (N choose 2) pairwise Euclidian distances
    d_sch = sch.distance.pdist(z)
    d_max = d_sch.max()

    # This parameter is CRUCIAL for the optimal number of clusters generated
    # This parameter works better for the videos from Lucian
    threshold = 0.05 * d_max

    """
    I did not find much information on the linkage matrix (linkage_matrix), but
        from my understanding it is the direct result of the hierarchical
        clustering, which is performed by recursively splitting clusters,
        forming a dendrogram forest of trees (see if you have time
            https://stackoverflow.com/questions/5461357/hierarchical-k-means-in-opencv-without-knowledge-of-k
            "a forest of hierarchical clustering trees").
      The linkage matrix is stores on each row data for a clustered point:
            - the last element in the row is the leaf in the dendrogram tree
                forest the point belongs to. The leaf does not really tell you
                to which final cluster the point belongs to - (IMPORTANT) for
                this, we have the function sch.fcluster().
      See if you have the time (for some better understanding):
        https://stackoverflow.com/questions/11917779/how-to-plot-and-annotate-hierarchical-clustering-dendrograms-in-scipy-matplotlib

      See doc:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

      See if you have the time:
        https://stackoverflow.com/questions/16883412/how-do-i-get-the-subtrees-of-dendrogram-made-by-scipy-cluster-hierarchy
    """
    linkage_matrix = sch.linkage(d_sch, "single")
    common.DebugPrint("linkage_matrix = %s" % str(linkage_matrix))

    # Inspired from https://stackoverflow.com/questions/7664826/how-to-get-flat-clustering-corresponding-to-color-clusters-in-the-dendrogram-cre
    index_cluster = sch.fcluster(linkage_matrix, threshold, "distance")
    common.DebugPrint("index_cluster = %s" % str(index_cluster))

    c_max = -1

    # We "truncate" later the ending zeros from num_elems
    num_elems = [0] * (n + 1)
    # IMPORTANT: It appears the ids of the clusters start from 1, not 0
    for e in index_cluster:
        # print "e = %s" % str(e)
        num_elems[e] += 1
        if c_max < e:
            c_max = e

    # c_max is the MAXIMUM optimal number of clusters after the hierarchical
    # clustering is performed.
    common.DebugPrint("c_max (the MAX id of final clusters) = %d" % c_max)

    num_elems = num_elems[0 : c_max + 1]
    common.DebugPrint("num_elems = %s" % str(num_elems))
    """
    # We can also use:
    num_elems.__delslice__(c_max + 1, len(num_elems))
    but it's sort of deprecated
        - see http://docs.python.org/release/2.5.2/ref/sequence-methods.html
    """

    num_clusters = 0
    for e in num_elems:
        if e != 0:
            num_clusters += 1

    common.DebugPrint("num_clusters (the optimal num of clusters) = %d" %
                      num_clusters)
    assert num_clusters == c_max

    num_clusters_above_threshold = 0
    for i in range(c_max + 1):
        if num_elems[i] >= \
          config.THRESHOLD_NUM_NONMATCHED_ELEMENTS_IN_CLUSTER:
            common.DebugPrint("num_elems[%d] = %d" % (i, num_elems[i]))
            num_clusters_above_threshold += 1

    common.DebugPrint("num_clusters_above_threshold = %d" %
                      num_clusters_above_threshold)

    # TODO: Move this to config?
    return_only_biggest_cluster = False
    if return_only_biggest_cluster:
        # TODO: find biggest cluster - sort them after num_elems, etc
        res = []
        for i in range(n):
            # We start numbering the clusters from 1.
            if index_cluster[i] == num_clusters:
                res.append(z[i])
    else:
        res = {}
        for i in range(n):
            if num_elems[index_cluster[i]] >= \
              config.THRESHOLD_NUM_NONMATCHED_ELEMENTS_IN_CLUSTER:
                if index_cluster[i] not in res:
                    res[index_cluster[i]] = []
                res[index_cluster[i]].append(z[i])

    if config.USE_GUI and config.DISPLAY_PYTHON_CLUSTERING:
        # We clear the figure and the axes
        plt.clf()
        plt.cla()

        # Plot the data
        for i in range(n):  # index_cluster:
            # print "Z[i, 0] = %.2f, Z[i, 1] = %.2f" % (Z[i, 0], Z[i, 1])
            try:
                col_cluster = colors[index_cluster[i]]
            except:  # IndexError: list index out of range
                col_cluster = 2
            plt.scatter(z[i, 0], z[i, 1], c=col_cluster)

        plt.xlabel(
            "Height. (num_clusters = %d, num_clusters_above_threshold = %d)" %
            (num_clusters, num_clusters_above_threshold))

        plt.ylabel("Weight")

        # From http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.axis
        v = plt.axis()
        # We invert the y to have 0 up and x axis to have 0
        v = (0, v[1], v[3], 0)
        plt.axis(v)

        plt.show()

    return res


"""
This uses Python and OpenCV's cv2 module.
Note: unfortunately, cv::flann::hierarchicalClustering()
    doesn't have Python bindings, so we have to sort of
    implement it :) .
"""


def hierarchical_clustering_with_cv2_unfinished(z, n):
    # We choose an ~optimal number of clusters

    min_validity = 1000000
    min_validity_k = -1

    for k in range(2, 10 + 1):
        a = [None] * k

        """
        Inspired a bit from
            https://www.google-melange.com/gsoc/project/google/gsoc2013/abidrahman2/43002,
            \source\py_tutorials\py_ml\py_kmeans\py_kmeans_opencv\py_kmeans_opencv.rst
        """

        # Define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        t1 = float(cv2.getTickCount())

        ret, label, center = cv2.kmeans(z, k, criteria, 10,
                                        cv2.KMEANS_RANDOM_CENTERS)

        t2 = float(cv2.getTickCount())
        my_time = (t2 - t1) / cv2.getTickFrequency()
        common.DebugPrint(
            "HierarchicalClusteringWithCV2_UNFINISHED(): "
            "cv2.kmeans() took %.5f [sec]" % my_time)

        common.DebugPrint("ret = %s" % str(ret))
        common.DebugPrint("label = %s" % str(label))
        common.DebugPrint("center = %s" % str(center))

        # Now separate the data, Note the flatten()
        for i in range(k):
            a[i] = z[label.ravel() == i]

        common.DebugPrint("A[0] = %s" % str(a[0]))
        common.DebugPrint("A[0][:, 0] = %s" % str(a[0][:, 0]))
        common.DebugPrint("A[0][:, 1] = %s" % str(a[0][:, 1]))

        """
        Following Section 3.2 from http://www.csse.monash.edu.au/~roset/papers/cal99.pdf :
        See if you have the time, for further ideas:
            https://stackoverflow.com/questions/15376075/cluster-analysis-in-r-determine-the-optimal-number-of-clusters
        """
        intra = 0
        for i in range(k):
            # Gives exception: "TypeError: only length-1 arrays can be converted to Python scalars"
            for x in range(len(a[i])):
                intra += np.square(a[i][x, 0] - center[i, 0]) + \
                            np.square(a[i][x, 1] - center[i, 1])
        intra /= n

        dist_min = 1000000
        for i in range(k):
            for j in range(i + 1, k):
                dist = np.square(center[i, 0] - center[j, 0]) + \
                        np.square(center[i, 1] - center[j, 1])
                """
                dist = sqr(center[i, 0] - center[j, 0]) + \
                        sqr(center[i, 1] - center[j, 1])
                """
                common.DebugPrint("dist = %s" % str(dist))
                if dist < dist_min:
                    dist_min = dist
        inter = dist_min

        """
        We want to minimize intra (clusters be dense) and
            maximize inter (clusters be distant from one another).
        """
        validity = intra / inter

        if min_validity > validity:
            min_validity = validity
            min_validity_k = k

        if config.USE_GUI:
            # We clear the figure and the axes
            plt.clf()
            plt.cla()

            # Plot the data
            for i in range(k):
                """
                Note: A[0][:,0] (i.e., [:,0] is a numpy-specific
                    "split"-operator, not working for standard Python lists.
                """
                plt.scatter(a[i][:, 0], a[i][:, 1], c=colors[i])

            plt.scatter(center[:, 0], center[:, 1], s=80, c="b", marker="s")

            plt.xlabel(
                "Height. Also: k=%d, intra=%.1f, inter=%.1f, validity = %.4f" %
                (k, intra, inter, validity))

            plt.ylabel("Weight")

            plt.show()

        """
        TODO!!!! Implement section 4 from http://www.csse.monash.edu.au/~roset/papers/cal99.pdf:
         See "when we require the number of
            clusters to be increased, we split the cluster
            having maximum variance, so the k-means
            procedure is given good starting cluster centres."
        """
        # !!!!TODO: .... DO THE IMPLEMENTATION, WHITE BOY

    common.DebugPrint("IMPORTANT: min_validity_k = %d" % min_validity_k)

