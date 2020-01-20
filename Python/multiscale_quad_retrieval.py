import math
import numpy as np

import cv2

import common
import config
import findquads
import spatial_consistency
import Matlab

FILTER = True
USE_GPS_COORDINATES = False

if config.USE_MULTITHREADING:
    import multiprocessing

    class Globals:
        r_quadsTree = None
        r_harlocs = None
        q_harlocs = None
        md_threshold = None
        st_threshold = None
        all_ori = None
        all_id = None
        all_max = None
        all_cen = None
        nos = None
        scale_index = None
        crop_flag = None
        sequence = None
        rd_start = None
        rd_end = None
        maxdis = None
        maxori = None
        tolers = None
    g = Globals()


"""
When parallelizing multiscale_quad_retrieval(), we will obtain
  slightly different results for crossref (etc) - see crossref.txt.
  Note that when running in serial (without a POOL), we obtain the same result ALWAYS.
  (less relevant: if we run only 1 process in pool instead of 3, we get
     fewer changes in crossref.txt - NOT sure if this means something).
   It appears the reason is the fact I am running the FLANN KD-tree implementation,
     which is an approximate NN search library, employing randomization
     (see http://answers.opencv.org/question/32664/flannbasedmatcher-returning-different-results/).
     I guess the random number sequence, when having the same seed, evolves
        differently when serial VS in parallel
       threads, so the results tend to be different.
"""


def iteration_standalone_mqr(query_frame):
    r_quads_tree = g.r_quadsTree
    r_harlocs = g.r_harlocs
    q_harlocs = g.q_harlocs
    md_threshold = g.md_threshold
    st_threshold = g.st_threshold
    all_ori = g.all_ori
    all_id = g.all_id
    all_max = g.all_max
    all_cen = g.all_cen
    nos = g.nos
    scale_index = g.scale_index
    crop_flag = g.crop_flag
    sequence = g.sequence
    rd_start = g.rd_start
    rd_end = g.rd_end
    maxdis = g.maxdis
    maxori = g.maxori
    tolers = g.tolers

    """
    We make pp reference the desired multiharloc list for the query video
        frame query_frame
    """
    pp = q_harlocs[query_frame]

    """
    Alex: for the query frame query_frame we retrieve, for scale scale_index, the
        harris features in var points.
      Then we build the quads from points.
      Then for each quad (4 float values) we query the corresponding scale
        kd-tree, and we get the indices.
        Then we build the histogram and compute idf, ....!!!!

     Note: scale is 1 for original frame resolution and the higher
        we go we have lower image resolutions (we go higher in the
        Guassian pyramid I think).
    """
    points = pp[pp[:, 2] == scale_index, 0:2]
    qout, qcen, qmaxdis, qori = findquads.findquads(points, md_threshold, 0)

    common.DebugPrint("multiscale_quad_retrieval(): query_frame = %d, "
                      "qout.shape = %s" % (query_frame, str(qout.shape)))

    space_xy = np.zeros((qcen.shape[0], 2 * len(r_harlocs))) + np.nan
    votes = np.zeros((len(r_harlocs), 1))
    assert isinstance(tolers, float)

    """
    We substitute queryFrameQuad - 1 with queryFrameQuad, since we want
        to number arrays from 0 (not from 1 like in Matlab).
    """
    for queryFrameQuad in range(qout.shape[0]):
        """
        Matlab's polymorphism is really bugging here: although it's
            normally a float, tolers is considered to be a size 1 vector...
            so len(tolers) == 1
        """
        """
        We substitute tol_i - 1 with tol, since we want
            to number arrays from 0 (not from 1 like in Matlab).
        """
        for tol_i in range(1):
            tol = tolers

            # default for first PAMI with tol= 0.1 approximately

            # NOTE: SciPy's KDTree finds a few more results, in some cases,
            #    than the Matlab code from Evangelidis.
            # tol is a scalar representing the radius of the ball
            if config.KDTREE_IMPLEMENTATION == 0:
                idx = r_quads_tree.query_ball_point(qout[queryFrameQuad, :],
                                                    tol)
            elif config.KDTREE_IMPLEMENTATION == 1:
                pt = qout[queryFrameQuad, :]
                pt = np.array([[pt[0], pt[1], pt[2], pt[3]]], dtype=np.float32)
                retval, idx, dists = r_quads_tree.radiusSearch(
                    query=pt,
                    radius=(tol ** 2),
                    maxResults=NUM_MAX_ELEMS,
                    params=search_params)
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint(
                        "multiscale_quad_retrieval(): radiusSearch's retval "
                        "(at query_frame=%d, queryFrameQuad=%d) is %d\n" % (
                         query_frame, queryFrameQuad, retval))
                idx = idx[0]
                dists = dists[0]
                idx = idx[: retval]
                dists = dists[: retval]

            if common.MY_DEBUG_STDOUT:
                print("multiscale_quad_retrieval(): "
                      "qout[queryFrameQuad, :] = %s" % str(
                       qout[queryFrameQuad, :]))
                print("multiscale_quad_retrieval(): "
                      "idx = %s" % str(idx))
                print("multiscale_quad_retrieval(): "
                      "tol = %s" % str(tol))
                if config.KDTREE_IMPLEMENTATION == 0:
                    print("multiscale_quad_retrieval(): "
                          "r_quads_tree.data[idx] = %s" %
                          str(r_quads_tree.data[idx]))

            # We print the distances to the points returned in idx
            a = qout[queryFrameQuad, :]
            idx = np.array(idx)

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("multiscale_quad_retrieval(): "
                                  "all_max.shape = %s" % str(all_max.shape))
                common.DebugPrint("multiscale_quad_retrieval(): "
                                  "qmaxdis.shape = %s" % str(qmaxdis.shape))
                common.DebugPrint("multiscale_quad_retrieval(): "
                                  "qmaxdis = %s" % str(qmaxdis))
                common.DebugPrint("multiscale_quad_retrieval(): "
                                  "qori.shape = %s" % str(qori.shape))
                common.DebugPrint("multiscale_quad_retrieval(): "
                                  "qori = %s" % str(qori))

            if len(idx) == 0:
                # NOT A GOOD IDEA: continue
                # idx = np.array([])
                dis_idx = np.array([])
                ori_idx = np.array([])
            else:
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "queryFrameQuad = %s" % str(
                                       queryFrameQuad))
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "all_max[idx] = %s" % str(all_max[idx]))
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "qmaxdis[queryFrameQuad] = %s" % str(
                                       qmaxdis[queryFrameQuad]))

                dis_idx = np.abs(
                    qmaxdis[queryFrameQuad] - all_max[idx]) < maxdis

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "dis_idx = %s" % str(dis_idx))

                idx = idx[dis_idx]

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "idx (after idx = idx[dis_idx]) = "
                                      "%s" % str(idx))

                ori_idx = np.abs(qori[queryFrameQuad] - all_ori[idx]) < maxori

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "ori_idx = %s" % str(ori_idx))

                idx = idx[ori_idx]

            # IMPORTANT ###################################################
            # IMPORTANT ###################################################
            # IMPORTANT ###################################################
            # spatio-temporal consistency
            # IMPORTANT ###################################################
            # IMPORTANT ###################################################
            # IMPORTANT ###################################################

            if idx.size > 0:
                # Normally crop_flag == 0
                if crop_flag == 0:
                    dy = qcen[queryFrameQuad, 0] - all_cen[idx, 0]
                    dx = qcen[queryFrameQuad, 1] - all_cen[idx, 1]
                    D = dy ** 2 + dx ** 2
                    co_idx = D < pow(st_threshold, 2)
                    idx = idx[co_idx]
                else:
                    """
                    We substitute iii - 1 with iii, since we want
                        to number arrays from 0 (not from 1 like in Matlab).
                    """
                    for iii in range(len(idx)):
                        space_xy[queryFrameQuad,
                        (all_id[idx[iii]] - rd_start) * 2: (all_id[idx[
                            iii] - 1] - rd_start) * 2 + 1] = \
                            all_cen[idx[iii], :]

                # It has to be an np.array because we multiply it with a scalar
                histo_range = np.array(range(rd_start, rd_end + 1))
                hh = Matlab.hist(x=all_id[idx], binCenters=histo_range)

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "hh = %s" % (str(hh)))
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "hh.shape = %s" % (str(hh.shape)))
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "all_id.shape = %s" % (str(all_id.shape)))
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "idx = %s" % (str(idx)))
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "idx.shape = %s" % (str(idx.shape)))

                # nz can be computed more optimally
                # nz=find(hh~=0) # nz can be computed more optimally
                # np.nonzero() always returns a tuple, even if it contains
                # 1 element since hh has only 1 dimension
                nz = np.nonzero(hh != 0)[0]
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "nz = %s" % (str(nz)))
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "nz.shape = %s" % (str(nz.shape)))

                if nz.size > 0:
                    my_val = pow(math.log10(float(len(r_harlocs)) / len(nz)), 2)

                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "len(r_harlocs) = "
                                          "%d" % len(r_harlocs))
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "len(nz) = %d" % len(nz))
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "my_val = %.5f" % my_val)

                    # PREVIOUSLY
                    votes[nz, tol_i] = votes[nz, tol_i] + my_val

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("multiscale_quad_retrieval(): "
                          "votes.shape = %s" % (str(votes.shape)))
        common.DebugPrint("multiscale_quad_retrieval(): "
                          "votes = %s" % (str(votes)))

    return query_frame, np.ravel(votes)


"""
From http://www.mathworks.com/help/matlab/matlab_prog/symbol-reference.html:
    Dot-Dot-Dot (Ellipsis) - ...
    A series of three consecutive periods (...) is the line continuation 
    operator in MATLAB.
      Line Continuation
      Continue any MATLAB command or expression by placing an ellipsis at the 
      end of the line to be continued:
"""

NUM_MAX_ELEMS = 100000
# Gives fewer results than scipy's tree.query_ball_point when we have 65K
# features
search_params = dict(checks=1000000000)

# returns Votes_space, HH
# Alex: r_harlocs and q_harlocs are the corresponding lists of harlocs computed
"""
md_threshold = max-distance threshold used to build quads out of Harris features
st_threshold = threshold value for spatio-temporal consistency (coherence)
all_ori, all_id, all_max, all_cen = orientation, reference frame ids, 
max distances, respectively centroids coordinates of each reference quad for
scale scale_index
"""


def multiscale_quad_retrieval(r_quads_tree, r_harlocs, q_harlocs, md_threshold,
                              st_threshold, all_ori, all_id, all_max, all_cen,
                              nos, scale_index, crop_flag, sequence):
    common.DebugPrint("Entered multiscale_quad_retrieval(): "
                      "md_threshold = %s, st_threshold = %s." %
                      (str(md_threshold), str(st_threshold)))

    assert len(r_harlocs) != 0
    assert len(q_harlocs) != 0

    try:
        votes_space = np.load("votes_space%d.npz" % scale_index)['arr_0']
        HH = np.load("HH%d.npz" % scale_index)['arr_0']
        return votes_space, HH
    except:
        common.DebugPrintErrorTrace()

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("multiscale_quad_retrieval(): r_quads_tree = %s" %
                          str(r_quads_tree))
        common.DebugPrint(
            "multiscale_quad_retrieval(): len(r_harlocs) = %d" % len(r_harlocs))
        common.DebugPrint(
            "multiscale_quad_retrieval(): r_harlocs = %s" % str(r_harlocs))
        common.DebugPrint(
            "multiscale_quad_retrieval(): q_harlocs = %s" % str(q_harlocs))
        common.DebugPrint(
            "multiscale_quad_retrieval(): md_threshold = %s" % str(
                md_threshold))
        print("multiscale_quad_retrieval(): st_threshold = %s" % str(
            st_threshold))
        common.DebugPrint(
            "multiscale_quad_retrieval(): all_id = %s" % str(all_id))
        common.DebugPrint("multiscale_quad_retrieval(): all_id.shape = %s" % (
            str(all_id.shape)))
        common.DebugPrint(
            "multiscale_quad_retrieval(): sequence = %s" % str(sequence))
        print("multiscale_quad_retrieval(): crop_flag = %s" % str(crop_flag))

    t1 = float(cv2.getTickCount())

    if scale_index > nos:
        assert scale_index <= nos

    # TODO: take out rd_start
    rd_start = 0
    rd_end = len(r_harlocs) - 1

    j = 1

    """
    Inspired from
      https://stackoverflow.com/questions/17559140/matlab-twice-as-fast-as-numpy
        BUT doesn't help in this case:
    votes_space = np.asfortranarray(np.zeros( (len(RD), len(QD)) ))
    """
    votes_space = np.zeros((len(r_harlocs), len(q_harlocs)))

    # Make a distinct copy of HH from votes_space...
    # TODO: use MAYBE even np.bool - OR take it out
    HH = np.zeros((len(r_harlocs), len(q_harlocs)), dtype=np.int8)

    # it helps to make more strict the threshold as the scale goes up
    tolers = 0.1 - float(scale_index) / 100.0

    maxdis = 3 + scale_index
    maxori = 0.25

    # TODO: I am using multiprocessing.Poll and return votes the dispatcher
    #  assembles the results, but the results are NOT the same with the serial
    #  case - although they look pretty decent, but they seem to be
    #  suboptimal - dp_alex returns suboptimal cost path for
    #  USE_MULTITHREADING == True instead of False.
    #         (Note: running under the same preconditions
    #             multiscale_quad_retrieval I got the same results in dp_alex().
    """
    if False: #config.USE_MULTITHREADING == True:
        global g
        g.r_quads_tree = r_quads_tree
        g.r_harlocs = r_harlocs
        g.q_harlocs = q_harlocs
        g.md_threshold = md_threshold
        g.st_threshold = st_threshold
        g.all_ori = all_ori
        g.all_id = all_id
        g.all_max = all_max
        g.all_cen = all_cen
        g.nos = nos
        g.scale_index = scale_index
        g.crop_flag = crop_flag
        g.sequence = sequence
        g.RD_start = RD_start
        g.RD_end = RD_end
        g.maxdis = maxdis
        g.maxori = maxori
        g.tolers = tolers

        #Start worker processes to use on multi-core processor (able to run
        #   in parallel - no GIL issue if each core has it's own VM)
    
        pool = multiprocessing.Pool(processes=config.numProcesses)
        print("multiscale_quad_retrieval(): Spawned a pool of %d workers" %
                                config.numProcesses)

        listParams = range(0, len(q_harlocs)) #!!!!TODO: use counterStep, config.initFrame[indexVideo]

        #res = pool.map(iteration_standalone_mqr, listParams)
        # See https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing.pool
        res = pool.map(func=iteration_standalone_mqr, iterable=listParams,
                       chunksize=1)

        print("Pool.map returns %s" % str(res)) #x0.size + 1

        # From https://medium.com/building-things-on-the-internet/40e9b2b36148
        #    close the pool and wait for the work to finish
        pool.close()
        pool.join()

        # Doing the "reduce" phase after the workers have finished :)
        assert len(res) == len(q_harlocs)
        for query_frame, resE in enumerate(res):
            resEIndex = resE[0]
            resE = resE[1]
            assert resEIndex == query_frame
            # Gives: "ValueError: output operand requires a reduction, but reduction is not enabled"
            #votes_space[:, query_frame - 1] = votes
            votes_space[:, query_frame] = resE

        for query_frame in range(len(q_harlocs)):
            if crop_flag == 0:
                HH[:, query_frame] = 1
            else:
                HH[:, query_frame] = spatial_consistency.spatial_consistency(space_xy,
                                            qcen, len(r_harlocs), st_threshold, crop_flag)

        try:
            np.savez_compressed("votes_space%d" % scale_index, votes_space)
            np.savez_compressed("HH%d" % scale_index, HH)
        except:
            common.DebugPrintErrorTrace()

        return votes_space, HH
        """

    """
    We substitute q - 1 with q, since we want
      to number arrays from 0 (not from 1 like in Matlab).
    """
    for query_frame in range(len(q_harlocs)):
        common.DebugPrint("multiscale_quad_retrieval(): Starting iteration "
                          "query_frame = %d" % query_frame)

        """
        We make pp reference the desired multiharloc list for the query video
           frame query_frame
        """
        pp = q_harlocs[query_frame]

        points = pp[pp[:, 2] == scale_index, 0:2]
        qout, qcen, qmaxdis, qori = findquads.findquads(points, md_threshold, 0)

        if common.MY_DEBUG_STDOUT:
            print("multiscale_quad_retrieval(): query_frame = %d, "
                  "qout.shape (number of quads for query frame query_frame) = "
                  "%s" % (query_frame, str(qout.shape)))

        space_xy = np.zeros((qcen.shape[0], 2 * len(r_harlocs))) + np.nan
        votes = np.zeros((len(r_harlocs), 1))

        assert isinstance(tolers, float)

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multiscale_quad_retrieval(): quads of query "
                              "frame %d are: " % query_frame)
            common.DebugPrint("  qout = %s" % str(qout))

        """
        Alex: for each quad (4 floats) of the query frame from Harris feature of
        scale scale_index
          Note: all_id stores the reference frame id for each quad descriptor.
        """
        """
        We substitute queryFrameQuad - 1 with queryFrameQuad, since we want
            to number arrays from 0 (not from 1 like in Matlab).
        """
        for queryFrameQuad in range(qout.shape[0]):
            common.DebugPrint("multiscale_quad_retrieval(): Starting iteration "
                              "queryFrameQuad = %d" % queryFrameQuad)
            """
            Matlab's polymorphism is really bugging here: although it's
                normally a float, tolers is considered to be a size 1 vector...
                so len(tolers) == 1
            """
            """
            We substitute tol_i - 1 with tol, since we want
                to number arrays from 0 (not from 1 like in Matlab).
            """
            for tol_i in range(1):
                tol = tolers

                # default for first PAMI with tol= 0.1 approximately
                # NOTE: SciPy's KDTree finds a few more results, in some cases,
                #    than the Matlab code from Evangelidis.
                # tol is a scalar representing the radius of the ball
                if config.KDTREE_IMPLEMENTATION == 0:
                    idx = r_quads_tree.query_ball_point(qout[queryFrameQuad, :],
                                                        tol)
                elif config.KDTREE_IMPLEMENTATION == 1:
                    pt = qout[queryFrameQuad, :]
                    pt = np.array([[pt[0], pt[1], pt[2], pt[3]]],
                                  dtype=np.float32)
                    retval, idx, dists = r_quads_tree.radiusSearch(
                        query=pt,
                        radius=(tol ** 2),
                        maxResults=NUM_MAX_ELEMS,
                        params=search_params)
                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "radiusSearch's retval (at "
                                          "query_frame=%d, queryFrameQuad=%d) "
                                          "is %d" %
                                          (query_frame, queryFrameQuad, retval))
                    idx = idx[0]
                    dists = dists[0]
                    """
                    Note: retval is the number of neighbors returned from the 
                    radiusSearch().
                    But the idx and the dists can have more elements than the
                    returned retval.
                    """
                    idx = idx[: retval]
                    dists = dists[: retval]

                if common.MY_DEBUG_STDOUT:
                    print("multiscale_quad_retrieval(): "
                          "qout[queryFrameQuad, :] = %s" %
                          str(qout[queryFrameQuad, :]))
                    print("multiscale_quad_retrieval(): "
                          "idx = %s" % str(idx))
                    print("multiscale_quad_retrieval(): "
                          "dists = %s" % str(dists))
                    print("multiscale_quad_retrieval(): "
                          "tol = %s" % str(tol))
                    if config.KDTREE_IMPLEMENTATION == 0:
                        print("multiscale_quad_retrieval(): "
                              "r_quads_tree.data[idx] = %s" %
                              str(r_quads_tree.data[idx]))

                if common.MY_DEBUG_STDOUT:
                    a = qout[queryFrameQuad, :]
                    if config.KDTREE_IMPLEMENTATION == 0:
                        for myI, index in enumerate(idx):
                            b = r_quads_tree.data[index]
                    else:
                        pass
                idx = np.array(idx)

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "all_max.shape = %s" % str(all_max.shape))
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "qmaxdis.shape = %s" % str(qmaxdis.shape))
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "qmaxdis = %s" % str(qmaxdis))
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "qori.shape = %s" % str(qori.shape))
                    common.DebugPrint("multiscale_quad_retrieval(): "
                                      "qori = %s" % str(qori))

                if len(idx) == 0:
                    # NOT A GOOD IDEA: continue
                    dis_idx = np.array([])
                    ori_idx = np.array([])
                else:
                    if common.MY_DEBUG_STDOUT:
                        print("multiscale_quad_retrieval(): "
                              "queryFrameQuad = %s" % str(queryFrameQuad))
                        print("multiscale_quad_retrieval(): "
                              "all_max[idx] = %s" % str(all_max[idx]))
                        print("multiscale_quad_retrieval(): "
                              "qmaxdis[queryFrameQuad] = %s" %
                              str(qmaxdis[queryFrameQuad]))

                    if USE_GPS_COORDINATES:
                        # We look only at a part of the reference video
                        """
                        Since in some cases the video temporal alignment is
                            difficult to do due to similar portions in the
                            trajectory (see the drone videos, clip 3_some_lake)
                            we "guide" the temporal alignment by restricting
                            the reference frame search space - this is useful
                            when we have the geolocation (GPS) coordinate for
                            each frame.
                        """
                        if common.MY_DEBUG_STDOUT:
                            print("multiscale_quad_retrieval(): "
                                  "all_id = %s" % str(all_id))

                        if all_id.ndim == 2:
                            # TODO: put this at the beginning of the
                            #  function
                            assert all_id.shape[1] == 1
                            """
                            We flatten the array all_id
                              Note: We don't use order="F" since it's
                                    basically 1-D array
                            """
                            all_id = np.ravel(all_id)

                        # TODO: put start and end frame in config - or compute
                        #  it from geolocation
                        sub_idx = np.logical_and((all_id[idx] >= 2030 - 928),
                                                 (all_id[idx] <= 2400 - 928))
                        idx = idx[sub_idx]

                        if common.MY_DEBUG_STDOUT:
                            print("multiscale_quad_retrieval(): "
                                  "all_id = %s" % str(all_id))
                            print("multiscale_quad_retrieval(): "
                                  "sub_idx = %s" % str(sub_idx))
                            print("multiscale_quad_retrieval(): "
                                  "idx = %s" % str(idx))

                    if FILTER:
                        dis_idx = np.abs(
                            qmaxdis[queryFrameQuad] - all_max[idx]) < maxdis

                        if common.MY_DEBUG_STDOUT:
                            common.DebugPrint("multiscale_quad_retrieval(): "
                                              "dis_idx = %s" % str(dis_idx))

                        idx = idx[dis_idx]

                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "idx (after idx = idx[dis_idx]) = "
                                          "%s" % str(idx))

                    if FILTER:
                        ori_idx = np.abs(
                            qori[queryFrameQuad] - all_ori[idx]) < maxori

                        if common.MY_DEBUG_STDOUT:
                            common.DebugPrint("multiscale_quad_retrieval(): "
                                              "ori_idx = %s" % str(ori_idx))

                        idx = idx[ori_idx]

                # IMPORTANT ###################################################
                # IMPORTANT ###################################################
                # IMPORTANT ###################################################
                # spatio-temporal consistency
                # IMPORTANT ###################################################
                # IMPORTANT ###################################################
                # IMPORTANT ###################################################

                if idx.size > 0:
                    if crop_flag == 0:
                        if FILTER:
                            """
                            Alex: this is a simple procedure of eliminating 
                            False Positive (FP) matches, as presented in 
                            Section 4.2 of TPAMI 2013 paper.
                            Basically it filters out quad matches that have
                            centroids st_threshold away from the query quad.
                            Note: all_cen are the controids of all reference
                                quads.
                            """
                            dy = qcen[queryFrameQuad, 0] - all_cen[idx, 0]
                            dx = qcen[queryFrameQuad, 1] - all_cen[idx, 1]
                            D = dy ** 2 + dx ** 2
                            co_idx = D < pow(st_threshold, 2)
                            idx = idx[co_idx]
                    else:
                        """
                        We substitute iii - 1 with iii, since we want
                            to number arrays from 0 (not from 1 like in Matlab).
                        """
                        for iii in range(len(idx)):
                            space_xy[queryFrameQuad,
                            (all_id[idx[iii]] - rd_start) * 2: (all_id[idx[
                                iii] - 1] - rd_start) * 2 + 1] = \
                                all_cen[idx[iii], :]

                    # It has to be an np.array because we multiply it with a
                    # scalar
                    histo_range = np.array(range(rd_start, rd_end + 1))
                    hh = Matlab.hist(x=all_id[idx], binCenters=histo_range)

                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "hh = %s" % (str(hh)))
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "hh.shape = %s" % (str(hh.shape)))
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "all_id.shape = %s" % (
                                              str(all_id.shape)))
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "idx = %s" % (str(idx)))
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "idx.shape = %s" % (str(idx.shape)))

                    # % nz can be computed more optimally
                    nz = np.nonzero(hh != 0)[0]
                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "nz = %s" % (str(nz)))
                        common.DebugPrint("multiscale_quad_retrieval(): "
                                          "nz.shape = %s" % (str(nz.shape)))

                    if nz.size > 0:
                        my_val = pow(
                            math.log10(float(len(r_harlocs)) / len(nz)), 2)

                        if common.MY_DEBUG_STDOUT:
                            common.DebugPrint("multiscale_quad_retrieval(): "
                                              "len(r_harlocs) = %d" % len(
                                               r_harlocs))
                            common.DebugPrint("multiscale_quad_retrieval(): "
                                              "len(nz) = %d" % len(nz))
                            common.DebugPrint("multiscale_quad_retrieval(): "
                                              "my_val = %.5f" % my_val)
                        # PREVIOUSLY
                        votes[nz, tol_i] = votes[nz, tol_i] + my_val

        if common.MY_DEBUG_STDOUT:
            print("multiscale_quad_retrieval(): "
                  "votes.shape = %s" % (str(votes.shape)))
            if (np.abs(votes) < 1.0e-10).all():
                print("multiscale_quad_retrieval(): votes = 0 (all zeros)")
            else:
                print("multiscale_quad_retrieval(): votes = %s" % (str(votes)))

        # Note: since votes is basically a 1-D vector, we don't use the
        # Fortran order
        votes_space[:, query_frame] = np.ravel(votes)

        if crop_flag == 0:
            HH[:, query_frame] = 1
        else:
            HH[:, query_frame] = spatial_consistency.spatial_consistency(
                space_xy,
                qcen, len(r_harlocs), st_threshold, crop_flag)

    if common.MY_DEBUG_STDOUT:
        print("multiscale_quad_retrieval(scale_index=%d): "
              "votes_space =\n%s" % (scale_index, str(votes_space)))

    try:
        np.savez_compressed("votes_space%d" % scale_index, votes_space)
        np.savez_compressed("HH%d" % scale_index, HH)
    except:
        common.DebugPrintErrorTrace()

    t2 = float(cv2.getTickCount())
    my_time = (t2 - t1) / cv2.getTickFrequency()
    print("multiscale_quad_retrieval() took %.6f [sec]" % my_time)

    return votes_space, HH
