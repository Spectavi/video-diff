"""
 * We build the kd-tree from the quads returned by
 *    findquads for the reference video (with multi_scale_harris.py).
 *
 * See README for details of the flow in the pipeline.
"""

import sys

import cv2
import numpy as np

import common
import config
import findquads

if config.KDTREE_IMPLEMENTATION == 0:
    from scipy import spatial
    

def multiscale_quad_tree(r_path, threshold, scale_index):
    common.DebugPrint(
        "Entered multiscale_quad_tree(scale_index=%d)" % scale_index)

    t1 = float(cv2.getTickCount())

    found_files = False

    try:
        all_quads = np.load("all_quads%d.npz" % scale_index)['arr_0']
        all_id = np.load("all_id%d.npz" % scale_index)['arr_0']
        all_cen = np.load("all_cen%d.npz" % scale_index)['arr_0']
        all_max = np.load("all_max%d.npz" % scale_index)['arr_0']
        all_ori = np.load("all_ori%d.npz" % scale_index)['arr_0']
        n_d = np.load("n_d%d.npz" % scale_index)['arr_0']
        found_files = True

        common.DebugPrint("multiscale_quad_tree(): loaded the NPZ files for "
                          "this scale.")
        print("multiscale_quad_tree(scale_index=%d): time before starting the "
              "processing for the given query video = %s" %
              scale_index, common.GetCurrentDateTimeStringWithMilliseconds())
    except:
        common.DebugPrintErrorTrace()

    if not found_files:
        RD = r_path

        # OpenCV's KD-tree implementation does NOT accept float64
        all_quads = np.array([]).astype(np.float32)
        all_cen = np.array([]).astype(np.float32)
        all_id = np.array([]).astype(np.float32)
        all_max = np.array([]).astype(np.float32)
        all_ori = np.array([]).astype(np.float32)

        n_d = np.zeros((len(RD), 1))

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multiscale_quad_tree(): n_d.shape = "
                              "%s" % str(n_d.shape))
            common.DebugPrint("multiscale_quad_tree(): n_d = %s" % str(n_d))

        # Alex: for each reference video frame we compute the quads
        for iFor in range(len(RD)):
            # Alex: IMPORTANT: This loads into pp the multiscale Harris feature
            # saved in file r_path+RD(i).name
            # load harris locations of image (already computed)
            pp = r_path[iFor]

            if iFor % 10 == 0:
                common.DebugPrint("multiscale_quad_tree(): iFor = %d" % iFor)
                common.DebugPrint("multiscale_quad_tree():   scale_index = "
                                  "%s" % str(scale_index))
                common.DebugPrint("multiscale_quad_tree():   threshold = "
                                  "%s" % str(threshold))

            # Alex: We put in points the rows of pp having the
            # 3rd element == scale_index, and we take out the 3rd column of pp
            # [out,cen,maxdis,ori]=findquads(pp(pp(:,3)==scale_index,1:2),threshold,1)
            points = pp[pp[:, 2] == scale_index, 0:2]
            out, cen, maxdis, ori = findquads.findquads(points, threshold, 1)

            n_d[iFor] = out.shape[0]

            temp = np.zeros((out.shape[0], 1)) + iFor

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("Initially:")

                common.DebugPrint(
                    "  multiscale_quad_tree(): all_quads.shape = %s" % str(
                        all_quads.shape))
                common.DebugPrint(
                    "  multiscale_quad_tree(): all_quads = %s" % str(all_quads))

                common.DebugPrint(
                    "  multiscale_quad_tree(): all_max.shape = %s" % str(
                        all_max.shape))
                common.DebugPrint(
                    "  multiscale_quad_tree(): all_max = %s" % str(all_max))

                common.DebugPrint(
                    "  multiscale_quad_tree(): maxdis.shape = %s" % str(
                        maxdis.shape))
                common.DebugPrint(
                    "  multiscale_quad_tree(): maxdis = %s" % str(maxdis))

                common.DebugPrint(
                    "  multiscale_quad_tree(): all_ori.shape = %s" % str(
                        all_ori.shape))

                common.DebugPrint(
                    "  multiscale_quad_tree(): ori.shape = %s" % str(ori.shape))
                common.DebugPrint(
                    "  multiscale_quad_tree(): ori = %s" % str(ori))

                common.DebugPrint(
                    "  multiscale_quad_tree(): all_cen.shape = %s" % str(
                        all_cen.shape))
                common.DebugPrint(
                    "  multiscale_quad_tree(): all_cen = %s" % str(all_cen))

                common.DebugPrint(
                    "  multiscale_quad_tree(): cen.shape = %s" % str(cen.shape))
                common.DebugPrint(
                    "  multiscale_quad_tree(): cen = %s" % str(cen))

                common.DebugPrint(
                    "  multiscale_quad_tree(): out.shape = %s" % str(out.shape))
                common.DebugPrint(
                    "  multiscale_quad_tree(): out = %s" % str(out))
                #pass

            if out.size == 0:
                assert(cen.size == 0)
                assert(maxdis.size == 0)
                assert(ori.size == 0)
                continue

            """
            It crashes at
                all_quads = np.r_[all_quads, out]
            with "ValueError: array dimensions must agree except for d_0"
            because:
                multiscale_quad_tree(): out = []
                multiscale_quad_tree(): out.shape = (2, 0)
            """

            if all_quads.size == 0:
                all_quads = out.copy()
            else:
                all_quads = np.r_[all_quads, out]

            if all_cen.size == 0:
                all_cen = cen.copy()
            else:
                all_cen = np.r_[all_cen, cen]

            if all_id.size == 0:
                all_id = temp.copy()
            else:
                all_id = np.r_[all_id, temp]

            if all_max.size == 0:
                all_max = maxdis.copy()
            else:
                all_max = np.r_[all_max, maxdis]

            if all_ori.size == 0:
                all_ori = ori.copy()
            else:
                all_ori = np.r_[all_ori, ori]

        try:
            np.savez_compressed("all_quads%d" % scale_index, all_quads)
            np.savez_compressed("all_id%d" % scale_index, all_id)
            np.savez_compressed("all_cen%d" % scale_index, all_cen)
            np.savez_compressed("all_max%d" % scale_index, all_max)
            np.savez_compressed("all_ori%d" % scale_index, all_ori)
            np.savez_compressed("n_d%d" % scale_index, n_d)
        except:
            common.DebugPrintErrorTrace()

        if all_quads.size == 0:
            t2 = float(cv2.getTickCount())
            my_time = (t2 - t1) / cv2.getTickFrequency()
            common.DebugPrint("multiscale_quad_tree() "
                              "took %.6f [sec]" % my_time)

            return None, all_id, all_cen, all_max, all_ori, n_d, all_quads

    t_build1 = float(cv2.getTickCount())

    if config.KDTREE_IMPLEMENTATION == 0:
        tree = spatial.KDTree(all_quads)
    elif config.KDTREE_IMPLEMENTATION == 1:
        # TODO: try to use exact NN-search for the kd-tree -
        #  see http://docs.opencv.org/trunk/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html
        all_quads = all_quads.astype(np.float32)
        tree = cv2.flann_Index(features=all_quads, params=config.FLANN_PARAMS)

    t_build2 = float(cv2.getTickCount())
    my_time_build = (t_build2 - t_build1) / cv2.getTickFrequency()
    print("multiscale_quad_tree(): KD-tree build "
          "took %.6f [sec]" % my_time_build)

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("At the end:")
        common.DebugPrint(
            "  multiscale_quad_tree(): all_id.shape = %s" % str(all_id.shape))
        common.DebugPrint("  multiscale_quad_tree(): all_id = %s" % str(all_id))

        common.DebugPrint(
            "  multiscale_quad_tree(): all_cen.shape = %s" % str(all_cen.shape))
        common.DebugPrint(
            "  multiscale_quad_tree(): all_cen = %s" % str(all_cen))

        common.DebugPrint(
            "  multiscale_quad_tree(): all_max.shape = %s" % str(all_max.shape))
        common.DebugPrint(
            "  multiscale_quad_tree(): all_max = %s" % str(all_max))

        common.DebugPrint(
            "  multiscale_quad_tree(): all_ori.shape = %s" % str(all_ori.shape))
        common.DebugPrint(
            "  multiscale_quad_tree(): all_ori = %s" % str(all_ori))

        common.DebugPrint(
            "  multiscale_quad_tree(): n_d.shape = %s" % str(n_d.shape))
        common.DebugPrint("  multiscale_quad_tree(): n_d = %s" % str(n_d))

        common.DebugPrint(
            "  multiscale_quad_tree(): all_quads.shape = %s" % str(
                all_quads.shape))
        common.DebugPrint(
            "  multiscale_quad_tree(): all_quads = %s" % str(all_quads))

        common.DebugPrint(
            "  multiscale_quad_tree(): all_quads.shape before "
            "kd-tree = %s" % str(all_quads.shape))

        try:
            common.DebugPrint(
                "multiscale_quad_tree(): sys.getsizeof(tree) = %s" % str(
                    sys.getsizeof(tree)))
        except:
            pass
            common.DebugPrintErrorTrace()

    t2 = float(cv2.getTickCount())
    my_time = (t2 - t1) / cv2.getTickFrequency()
    print("multiscale_quad_tree() took %.6f [sec]" % my_time)

    return tree, all_id, all_cen, all_max, all_ori, n_d, all_quads

