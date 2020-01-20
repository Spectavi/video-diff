import common
import config
import Matlab

import math
import sys

import numpy as np

import cv2

CAUSAL_DO_NOT_SMOOTH = True


def rand_index(max_index, a_len):
    if a_len > max_index:
        index = np.array([])
        return

    index = np.zeros((1, a_len))
    available = range(1, max_index)

    """
    From Matlab help:
      r = rand(n) returns an n-by-n matrix
        containing pseudorandom values drawn from the standard uniform 
        distribution on the open interval (0,1).
      r = rand(m,n) or r = rand([m,n]) returns an m-by-n matrix.
    """
    rs = np.ceil(np.random.rand(1, len) *
                 range(max_index, max_index - a_len + 1 + -1, -1))

    for p in range(1, len + 1):
        while rs[p - 1] == 0:
            rs[p - 1] = np.ceil(np.random.rand(1) * (max_index - p + 1))

        index[p - 1] = available[rs[p - 1]]
        available[rs[p - 1]] = np.array([])

    return index

# TODO: Finish this method or delete it.
"""
# It seems ransac_line is NEVER called, since nop == 0 in causal()
def ransac_line(pts, iterNum, thDist, thInlrRatio):
    # RANSAC Use RANdom SAmple Consensus to fit a line
    #  RESCOEF = RANSAC(PTS,ITERNUM,THDIST,THINLRRATIO) PTS is 2*n matrix including
    #   n points, ITERNUM is the number of iteration, THDIST is the inlier
    #   distance threshold and ROUND(THINLRRATIO*SIZE(PTS,2)) is the inlier number threshold. The final
    #   fitted line is y = alpha*x+beta.
    #   Yan Ke @ THUEE, xjed09@gmail.com
    #
    #   modified by georgios evangelidis

    # TODO: not finished implementing it since it's not used
    assert False

    sampleNum = 2

    ptNum = pts.shape[1]
    thInlr = np.round(thInlrRatio * ptNum)

    inlrNum = np.zeros((1, iterNum))
    theta1 = np.zeros((1,iterNum))
    rho1 = np.zeros((1, iterNum))

    for p in range(1, iterNum + 1):
        #% 1. fit using 2 random points
        sampleIdx = rand_index(ptNum, sampleNum)

        #ptSample = pts(:,sampleIdx)
        ptSample = pts[:, sampleIdx - 1]

        #d = ptSample(:,2)-ptSample(:,1)
        d = ptSample[:, 1] - ptSample[:, 1]

        #d=d/norm(d) #% direction vector of the line
        d = d / npla.norm(d) #% direction vector of the line

        #% 2. count the inliers, if more than thInlr, refit else iterate
        #n = [-d(2),d(1)] #% unit normal vector of the line
        n = np.c_[-d[1], d[0]] #% unit normal vector of the line

        #dist1 = n*(pts-repmat(ptSample(:,1),1,ptNum))
        dist1 = n * (pts - repmat(ptSample[:, 0], 1, ptNum)) # TODO: check more

        inlier1 = find(abs(dist1) < thDist)

        #inlrNum(p) = length(inlier1)
        inlrNum[p - 1] = len(inlier1)

        #if length(inlier1) < thInlr, continue end
        if len(inlier1) < thInlr:
            continue

        #ev = princomp(pts(:,inlier1)')
        ev = princomp(pts[:, inlier1].T)

        #d1 = ev(:,1)
        d1 = ev[:, 0]

        #theta1(p) = -atan2(d1(2),d1(1)) #% save the coefs
        theta1[p - 1] = - math.atan2(d1[1], d1[0]) #% save the coefs

        #rho1(p) = [-d1(2),d1(1)]*mean(pts(:,inlier1),2)
        rho1[p - 1] = [-d1(2),d1(1)] * pts[:, inlier1].mean(2)

    #% 3. choose the coef with the most inliers
    #[~,idx] = max(inlrNum)
    idx = argmax(inlrNum)

    theta = theta1[idx]
    rho = rho1[idx]

    alpha = -sin(theta) / cos(theta)
    beta  = rho / cos(theta)

    return alpha, beta
"""


'''
About costs:
   - Evangelidis' causal() uses a
      mean over Vote space and weighted sum w.r.t. the scales
   - my causal() uses a simple summation
   -
   - Evangelidis' dp3() starts from
      sum after scale of Vote space
     but also uses in the update phase of the memoization table the
          weights over the different scales.
'''


def compute_cost(crossref, v, file_name="crossref.txt"):
    # v[r][q] = votes of ref frame r for query frame q

    print("compute_cost(): v.shape = %s" % str(v.shape))
    print("compute_cost(): crossref.shape = %s" % str(crossref.shape))

    # TODO: print also a synchronization error (look at TPAMI 2013 Evangelidis)

    num_back = 0
    total_step = 0
    penalty_cost = 0
    my_min = crossref[0][1]
    my_max = crossref[0][1]
    for i in range(1, crossref.shape[0]):
        if my_min > crossref[i][1]:
            my_min = crossref[i][1]
        if my_max < crossref[i][1]:
            my_max = crossref[i][1]

        total_step += abs(crossref[i][1] - crossref[i - 1][1])
        # TODO: check also if we stay too long in the same ref frame and
        #  penalize if more than 10-20 same value in a row
        penalty_cost += abs(crossref[i][1] - crossref[i - 1][1])
        if crossref[i][1] < crossref[i - 1][1]:
            num_back += 1
    abs_avg_step = total_step / (crossref.shape[0] - 1)
    avg_step = (crossref[crossref.shape[0] - 1][1] - crossref[0][1]) / (
        crossref.shape[0] - 1)

    cost = 0.0
    my_text2 = "compute_cost(): crossref and v =\n"
    for q in range(crossref.shape[0]):
        assert crossref[q][0] == q
        try:
            cost += v[crossref[q][1]][q]
            my_text2 += "[%d %d] %.7f " % \
                        (q, crossref[q][1] + config.initFrame[1],
                         v[crossref[q][1]][q])
            for r in range(int(crossref[q][1]) - 5, int(crossref[q][1]) + 5):
                if r < 0:
                    continue
                if r >= v.shape[0]:
                    break
                my_text2 += "%.7f " % v[r, q]
        except:
            common.DebugPrintErrorTrace()

        """
        We print the first to nth order statistics - e.g., the first 5 biggest
          vote values.
        I got inspired from
          https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
         (see also
          https://stackoverflow.com/questions/10337533/a-fast-way-to-find-the-largest-n-elements-in-an-numpy-array)
        """
        my_arr = v[:, q].copy()
        my_arr_indices = my_arr.argsort()[-5:][::-1]
        my_text2 += " max ind = %s" % str(my_arr_indices + config.initFrame[1])
        my_text2 += " max vals = %s" % str(my_arr[my_arr_indices])
        my_text2 += "\n"

    my_text2 += "\n\ncost computed is %.7f\n" % cost
    my_text2 += "penalty is %.7f\n" % penalty_cost
    my_text2 += "reference frames are in the interval [%d, %d]\n" % \
                (my_min + config.initFrame[1], my_max + config.initFrame[1])
    my_text2 += "absolute avg step computed is %.7f\n" % abs_avg_step
    my_text2 += "  avg step computed is %.7f\n" % avg_step
    my_text2 += "Number of times going back (num_back) is %d" % num_back

    # TODO: print also a synchronization error (look at TPAMI 2013 Evangelidis)

    f_output = open(file_name, "wt")
    f_output.write(my_text2)
    f_output.close()


def causal_alex(v_space, num_frames_q, num_frames_r):
    nos = v_space.shape[2]
    # We transform nan in 0 in v_space
    for i in range(1, nos + 1):
        v_temp = v_space[:, :, i - 1]
        v_temp[np.isnan(v_temp)] = 0
        v_space[:, :, i - 1] = v_temp

    # TODO: we should use a weighted sum w.r.t. the scales,
    #  just like in causal()
    v = v_space.sum(2)

    crossref = np.zeros((num_frames_q, 2))

    for iFor in range(num_frames_q):
        crossref[iFor, 0] = iFor
        b = v[:, iFor].argmax()
        crossref[iFor, 1] = b

    print("causal_alex(): crossref = %s" % str(crossref))

    compute_cost(crossref, v, "crossref_causal_Alex.txt")
    print("causal_alex(): END")

    # TODO: write crossref_causal_Alex.txt
    return crossref


def causal(v_space, H, num_frames_q, num_frames_r, bov_flag, crop_flag,
           const_type,
           nop=0):
    # causal() is the local/greedy optimization solution
    # nargin>7 means that we do the local smoothing with RANSAC (see the paper)

    print("causal(): At entrance \n"
          "    num_frames_q=%s, num_frames_r=%s, bov_flag=%d, crop_flag=%d, "
          "const_type=%d, nop=%d" %
          (num_frames_q, num_frames_r, bov_flag, crop_flag,
           const_type, nop))

    # Normally bov_flag=0, crop_flag=0, const_type=1,nop=0

    if common.MY_DEBUG_STDOUT:
        print("causal(): v_space.shape = %s" % str(v_space.shape))
        print("causal(): H.shape = %s" % str(H.shape))

        for i in range(v_space.shape[2]):
            common.DebugPrint("causal(): v_space[:, :, %d] = \n%s" % (
                i, str(v_space[:, :, i])))

        for i in range(H.shape[2]):
            common.DebugPrint(
                "causal(): H[:, :, %d] = \n%s" % (i, str(H[:, :, i])))

    # 3D matrix
    nos = v_space.shape[2]

    # same with the multi_scale_harris function
    sigma_0 = 1.2
    n = range(0, nos)

    # NOT GOOD in Python: sigma_d = math.sqrt(1.8)**n * sigma_0
    sq18 = np.ones((1, nos)) * math.sqrt(1.8)
    sigma_d = sq18 ** n * sigma_0
    w = sigma_d[0]
    w = w / w.sum()

    # Alex: normally NOT executed
    if bov_flag == 1:
        w = w[:, ::-1]  # We flip on the vertical (left becomes right)

    print("causal(): w = %s" % str(w))

    # Alex: normally NOT executed
    # TODO: Remove assert False or delete code block.
    """
    if (const_type == 1) and (bov_flag == 1):
        assert False  # Normally this code does NOT get executed
        vv = np.zeros((v_space.shape[0], v_space.shape[1]))

        for j in range(1, nos + 1):
            vv = vv + v_space[:, :, j - 1]

        X, Y = sort(vv, "descend")

        # enable top-N list
        N = 300

        for s in range(1, nos + 1):
            for i in range(1, v_space.shape[1] + 1):
                y = Y[:, i - 1]
                votes = v_space[:, i - 1, s - 1]
                h = H[:, i - 1, s - 1]
                votes[y[N:]] = 0
                h[y[N:]] = 0
                v_space[:, i - 1, s - 1] = votes
                H[:, i - 1, s - 1] = h
    """

    vv = None

    crossref = np.zeros((num_frames_q, 2))

    # We transform nan in 0 in v_space
    """
    We substitute i - 1 with i, since array indexing starts from 1 in Matlab
        and 0 in Python.
    """
    for i in range(nos):
        v_temp = v_space[:, :, i]
        v_temp[np.isnan(v_temp)] = 0
        v_space[:, :, i] = v_temp

    # Alex: I personally find this idea of using filter2() VERY BAD -
    # the results in v_space should already be VERY good

    if not CAUSAL_DO_NOT_SMOOTH:
        # this filtering of votes favors smoother results
        b = Matlab.hamming(11)

        """
        We substitute i - 1 with i, since array indexing starts from 1 in Matlab
            and 0 in Python.
        """
        for i in range(nos):
            """
            From the Matlab help:
              Y = filter2(h,X) filters
              the data in X with the two-dimensional FIR filter
              in the matrix h. It computes the result, Y,
              using two-dimensional correlation, and returns the central part of
              the correlation that is the same size as X.
            """
            v_space[:, :, i] = Matlab.filter2(b, v_space[:, :, i])
            # TODO: do the optimization Evangelidis says here

    """
    From Matlab help:
      M = mean(A,dim) returns
      the mean values for elements along the dimension of A specified
      by scalar dim. For matrices, mean(A,2) is
      a column vector containing the mean value of each row.
    """
    # this might help more instead of starting from zero votes
    v = v_space.mean(2)

    """
    We substitute i - 1 with i, since array indexing starts from 1 in Matlab
        and 0 in Python.
    """
    for i in range(nos):
        if crop_flag == 0:
            if (const_type == 1) and (bov_flag == 1):
                # TODO: think well *
                v += w[i] * H[:, :, i] + v_space[:, :, i]
            else:
                # Alex: we are normally in this case, since crop_flag == 0,
                # const_type == 1, bov_flag == 0
                # TODO: think well *   Exception "ValueError: operands could
                #  not be broadcast together with shapes (5) (23,8)"
                v += w[i] * v_space[:, :, i]
        else:
            v += w[i] * v_space[:, :, i] + H[:, :, i]  # TODO: think well *

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("causal(): v.shape = %s" % str(v.shape))
        common.DebugPrint("causal(): v (the matrix used to choose the"
                          "max-voting reference frame) = %s" % str(v))

    """
    We substitute iFor -1 with iFor since arrays start with 0 in Python,
        not with 1 like in Matlab
    """
    for iFor in range(num_frames_q):
        crossref[iFor, 0] = iFor  # TODO - think well

        b = v[:, iFor].argmax()
        a = v[:, iFor][b]

        crossref[iFor, 1] = b  # TODO - think well

    # We normally do NOT execute the following code, since nop == 0
    # TODO: This code is broken, both mod and ransac_line methods either don't
    #   exist or are not finished.
    """
    if nop != 0:
        xx = crossref[:, 0].T
        yy = crossref[:, 1].T
        yy_new = yy

        if mod(nop, 2) == 0:
            quit()

        # miso is used only as array index and in
        # range expressions --> it can be an integer
        miso = (nop - 1) / 2

        if nop < 7:
            pass

        for iFor in range(miso + 1, len(xx) - miso + 1):
            xx = xx[iFor - miso - 1 : iFor + miso]
            yy = yy[iFor - miso - 1 : iFor + miso]

            if nop < 9:
                # iter_num is used only in range expressions and
                #       array dimensions --> it can be an integer.
                iter_num = nop * (nop - 1) / 2
            else:
                iter_num = 10

            th_dist = 2
            th_inlr_ratio = 0.6

            # From Matlab help:
            #   RANSAC Use RANdom SAmple Consensus to fit a line
            #     RESCOEF = RANSAC(PTS,ITERNUM,THDIST,THINLRRATIO) PTS is 2*n
            #     matrix including n points, ITERNUM is the number of iteration,
            #     THDIST is the inlier distance threshold and
            #     ROUND(THINLRRATIO*SIZE(PTS,2)) is the inlier number threshold.
            #     The final fitted line is y = alpha*x+beta.
            #     Yan Ke @ THUEE, xjed09@gmail.com
            alpha, beta = ransac_line( np.r_[xx, yy], iter_num,
                                      th_dist, th_inlr_ratio)

            if alpha != 0:
                yy_new = alpha * xx + beta  # TODO: think well *
                yy_new[iFor - 1] = yy_new[miso+1 - 1]

        crossref[:, 2] = yy_new
    """

    crossref = crossref.astype(int)

    print("causal(): crossref = %s" % str(crossref))

    compute_cost(crossref, v, "crossref_causal.txt")

    return crossref


def dp3(vspace, num_frames_r, num_frames_q, bov_flag):
    # TODO: There are two vars in this fn, one 'D" and the other 'd'. We should
    #   find a different name for one of them.
    # Dynamic programming for a maximum-vote path in vote-space
    # 2010, Georgios Evangelidis <georgios.evangelidis@iais.fraunhofer.de>

    print("Entered dp3(): Running dynamic programming...")
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("dp3(): v_space = %s" % str(vspace))

    crossref = np.zeros((num_frames_q, 2))
    sigma_0 = 1.2

    r, c, d = vspace.shape
    n = np.array(range(1, d + 1))
    sigma_i = math.sqrt(1.8)**n * sigma_0
    w = sigma_i
    w = w / float(w.sum())

    if bov_flag == 1:
        w = w[:, ::-1]

    # Initialization
    D = np.zeros((r + 1, c + 1))
    D[0, :] = np.nan
    D[:, 0] = np.nan
    D[0, 0] = 0

    """
    We substitute i - 1 with i since arrays start with 0 in Python,
        not with 1 like in Matlab.
    """
    for i in range(d):
        v_temp = vspace[:, :, i]
        v_temp[np.isnan(v_temp)] = 0  # TODO: check OK
        vspace[:, :, i] = v_temp

    vv = np.zeros((r, c))
    """
    We substitute j - 1 with j since arrays start with 0 in Python,
        not with 1 like in Matlab.
    """
    for j in range(d):
        vv = vv + vspace[:, :, j]

    D[1:, 1:] = vv

    new_dp3_alex = False

    if new_dp3_alex:
        # Alex: added cost
        cost = np.zeros((r + 1, c + 1))

    tback = np.zeros((r + 1, c + 1))

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("dp3(): printing locally optimum solutions:")
        # Alex: trying out to find a better solution than dp3() !!!!TODO : more
        # This solution is basically the one returned by causal() IF we do NOT
        # apply Matlab.filter2() on v_space
        for j in range(1, c + 1):
            max_col = 0.0
            max_pos = -1
            for i in range(1, r + 1):
                assert D[i, j] >= 0.0
                if max_col < D[i, j]:
                    max_col = D[i, j]
                    # So for query frame j we have a candidate matching
                    # ref frame i
                    max_pos = i
                    common.DebugPrint("dp3(): for query frame %d - "
                                      "candidate frame %d" % (j - 1, max_pos))
            common.DebugPrint("dp3(): for query frame %d we found matching "
                              "ref frame %d" % (j - 1, max_pos))

    # TODO: make i =0.., j=0.. and substitute i-1 with i, i-2 with i-1, etc
    for i in range(1, r + 1):
        for j in range(1, c + 1):
            if (i > 1) and (j > 1):
                dd1 = w[0] * vspace[i - 2, max(0, j - 3), 0]
                dd2 = w[0] * vspace[max(0, i - 3), j - 2, 0]
                dd3 = w[0] * vspace[i - 2, j - 2, 0]
                dd4 = w[0] * vspace[i - 2, j - 1, 0]
                dd5 = w[0] * vspace[i - 1, j - 2, 0]

                if d > 1:
                    for sc in range(2, d + 1):
                        dd1 = max(dd1, w[sc - 1] *
                                  vspace[i - 2, max(0, j - 3), sc - 1])
                        dd2 = max(dd2, w[sc - 1] * \
                                  vspace[max(0, i - 3), j - 2, sc - 1])
                        dd3 = max(dd3, w[sc - 1] * vspace[i - 2, j - 2, sc - 1])
                        dd4 = max(dd4, w[sc - 1] * vspace[i - 2, j - 1, sc - 1])
                        dd5 = max(dd5, w[sc - 1] * vspace[i - 1, j - 2, sc - 1])

                D[i - 1, j - 2] += dd1
                D[i - 2, j - 1] += dd2
                D[i - 1, j - 1] += dd3
                D[i - 1, j] += dd4
                D[i, j - 1] += dd5

                dmax, tb = Matlab.max(np.array([
                    D[i - 1, j - 1] + 1.0 / math.sqrt(2.0),
                    D[i - 2, j - 1] + 1.0 / math.sqrt(5.0),
                    D[i - 1, j - 2] + 1.0 / math.sqrt(5.0),
                    D[i - 1, j] + 1,
                    D[i, j - 1] + 1]))
            else:
                dmax, tb = Matlab.max(
                    np.array([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("dp3(): dmax = %s" % str(dmax))
                    common.DebugPrint("dp3(): tb = %s" % str(tb))

            if new_dp3_alex:
                cost[i, j] = 0  # TODO: think more
            else:
                D[i, j] += dmax  # TODO: for me it's weird he adds dmax here...

            tback[i, j] = tb

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("dp3(): D.shape = %s" % str(D.shape))
        common.DebugPrint("dp3(): D = %s" % str(D))
        common.DebugPrint("dp3(): tback.shape = %s" % str(tback.shape))
        common.DebugPrint("dp3(): tback = %s" % str(tback))

    # Traceback
    i = r + 1
    j = c + 1
    y = i - 1
    x = j - 1

    while (i > 2) and (j > 2):
        tb = tback[i - 1, j - 1] + 1  # In Matlab, max returns indices from 1..
        if tb == 1:
            i -= 1
            j -= 1
        elif tb == 2:
            i -= 2
            j -= 1
        elif tb == 3:
            i -= 1
            j -= 2
        elif tb == 4:
            i -= 1
            j = j
        elif tb == 5:
            j -= 1
            i = i
        else:
            assert False

        y = np.hstack([i - 1, y])
        x = np.hstack([j - 1, x])

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("dp3(): before D.shape = %s" % str(D.shape))
        common.DebugPrint("dp3(): before D = %s" % str(D))

    # Strip off the edges of the D matrix before returning
    D = D[1: (r + 1), 1: (c + 1)]

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("dp3(): D.shape = %s" % str(D.shape))
        common.DebugPrint("dp3(): D = %s" % str(D))

    rd_start = 1

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("dp3(): v_space.shape = %s" % str(vspace.shape))

    # TODO: understand well what is x,y and why computes p
    for i in range(0, vspace.shape[1]):
        crossref[i, 0] = i  # TODO: think if OK

        p = np.nonzero(x == i)
        p = p[0]

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("dp3(): x.shape = %s" % str(x.shape))
            common.DebugPrint("dp3(): x = %s" % str(x))
            common.DebugPrint("dp3(): y.shape = %s" % str(y.shape))
            common.DebugPrint("dp3(): y = %s" % str(y))
            common.DebugPrint("dp3(): i = %s" % str(i))
            common.DebugPrint("dp3(): p = %s" % str(p))

        if p.size == 0:
            # Alex: Vali Codreanu said to change from temp=0 to temp=3
            temp = 0

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("dp3(): temp = %s" % str(temp))

            crossref[i, 1] = 0 + rd_start - 1
        else:
            temp = y[p]

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("dp3(): temp = %s" % str(temp))

            if temp.size == 1:
                # If temp has only 1 element:
                crossref[i, 1] = temp + rd_start - 1
            else:
                crossref[i, 1] = temp[-1] + rd_start - 1

    common.DebugPrint("dp3(): crossref = %s" % str(crossref))
    compute_cost(crossref, vv, "crossref_dp3.txt")
    return y, x, D, tback, crossref


def dp_alex(v_space, num_frames_r, num_frames_q, bov_flag, prev_ref=5,
            next_ref=0):
    """
    v_space is a matrix with shape (num_frames_r, num_frames_q).
    See multiscale_quad_retrieval.py for definition:
        Votes_space = np.zeros( (len(RD), len(QD)) )
    """
    t1 = float(cv2.getTickCount())

    common.DebugPrint("Entered dp_alex(): Running dynamic programming...")

    r, c, d = v_space.shape

    # We substitute all NaN's of v_space
    for i in range(d):
        v_temp = v_space[:, :, i]
        v_temp[np.isnan(v_temp)] = 0
        v_space[:, :, i] = v_temp

    vv = np.zeros((r, c))
    for j in range(d):
        vv += v_space[:, :, j]

    # Checking that vv has positive elements
    assert np.nonzero(vv < 0.0)[0].size == 0

    print_matrices = True

    if common.MY_DEBUG_STDOUT and print_matrices:
        print("dp_alex(): r = %d, c = %d" % (r, c))
        print("dp_alex(): vv = \n%s" % str(vv))
        sys.stdout.flush()

    D = np.zeros((r, c))
    tback = np.zeros((r, c))

    for ref in range(r):
        D[ref, 0] = vv[ref, 0]
        tback[ref, 0] = -1

    for qry in range(1, c):
        for ref in range(r):
            # We enumerate a few reference frames to find the one with
            # highest votes
            lb = ref - prev_ref
            ub = ref + next_ref

            if lb < 0:
                lb = 0
            if lb >= r:
                lb = r - 1

            if ub < 0:
                ub = 1
            if ub >= r:
                ub = r - 1

            max_pos = lb
            for i in range(lb + 1, ub + 1):
                """
                We use <= --> we break ties by going forward in the
                    reference video (incrementing the reference frame for
                    the next query frame).
                """
                if D[max_pos, qry - 1] <= D[i, qry - 1]:
                    max_pos = i
            # max_pos is the maximum vote reference frame for query frame qry

            D[ref, qry] += D[max_pos, qry - 1] + vv[ref, qry]
            tback[ref, qry] = max_pos

    if common.MY_DEBUG_STDOUT and print_matrices:
        print("D = \n%s" % str(D))
        print("tback = \n%s" % str(tback))

    crossref = np.zeros((num_frames_q, 2))

    # Find max-cost path (the critical path) for the last query frame:
    max_pos = 0
    for ref in range(1, r):
        """
        We use <= --> we break ties by going forward in the
            reference video (incrementing the reference frame for
            the next query frame) - debatable if this is a good idea!!!!TODO.
        """
        if D[max_pos, c - 1] <= D[ref, c - 1]:
            max_pos = ref
    print("max_pos = %d" % max_pos)

    print("dp_alex(): cost critical path = %s" % str(D[max_pos, c - 1]))

    pos_ref = max_pos
    for qry in range(c - 1, 0-1, -1):
        crossref[qry][0] = qry
        crossref[qry][1] = pos_ref

        common.DebugPrint("qry=%d, pos_ref=%d" % (qry, pos_ref))
        pos_ref = tback[pos_ref, qry]

    # time took
    common.DebugPrint("dp_alex(): crossref = %s" % str(crossref))
    compute_cost(crossref, vv, "crossref_dp_Alex.txt")
    # TODO: assert cost computed is = D[max_pos,...]

    t2 = float(cv2.getTickCount())
    my_time = (t2 - t1) / cv2.getTickFrequency()
    print("dp_alex() took %.6f [sec]" % my_time)

    y = None
    x = None
    return y, x, D, tback, crossref


dp3Orig = dp3
dp3 = dp_alex


# TODO: Move tests into separate files.

import unittest


class TestSuite(unittest.TestCase):
    def testCausal(self):
        # This is a test case from the videos from Evangelidis, with
        # frameStep = 200
        vspace = np.zeros( (12, 4, 5) )
        vspace[:, :, 0] = np.array([ [0,        0,        0,        0],
                                     [0,        0,   1.1646,        0],
                                     [0,   1.1646,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0]])
        # The rest of v_space is completely 0 :)

        H = np.ones((12, 4, 5))

        num_frames_q = 4
        num_frames_r = 12
        bov_flag = 0
        crop_flag = 0
        const_type = 1

        res = causal(vspace, H, num_frames_q, num_frames_r, bov_flag,
                     crop_flag, const_type)
        print("testCausal(): res from causal() = %s" % str(res))

        resGood = np.array([[ 0, 2],
                            [ 1, 2],
                            [ 2, 1],
                            [ 3, 1]])

        aZero = res - resGood

        self.assertTrue((aZero == 0).all())

        res = dp3(vspace, num_frames_r, num_frames_q, bov_flag)
        # TODO: test result of dp3()


if __name__ == '__main__':
    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    np.set_printoptions(threshold=1000000, linewidth=5000)

    unittest.main()
