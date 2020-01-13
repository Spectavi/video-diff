import cv2

import common_cv

# From lk_homography.py

lk_params = dict(winSize=(19, 19),
                 maxLevel=2,
                 criteria=(
                 cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Alex: added useHarrisDetector for multi-scale harris detector inspired from
# Evangelidis, (otherwise it was using cornerMinEigenVal detector)
feature_params = dict( maxCorners=1000,
                       qualityLevel=0.01,
                       minDistance=8,
                       blockSize=19,
                       useHarrisDetector=True)


def checked_trace(img0, img1, p0, back_threshold = 1.0):
    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold

    return p1, status


img1Prev = None
LKp0 = None
LKp1 = None
LKp2 = None

"""
    TODO: do a REAL Lucas-Kanade implementation:
    think how to take out else: branch (img1Prev == None), etc -
    BASICALLY THIS LK assumes only 1 video and does ~matching w.r.t. the
    1st frame of the video
"""


# Computing homography with Lucas-Kanade's algo (taken from lk_homography.py)
def lucas_kanade_homography(img1):
    global img1Prev, LKp0, LKp1, LKp2

    h = None
    status = None

    # TODO: Figure out why it displays black instead of green
    # Is it because the image is gray already?
    green = (0, 255, 0)

    if LKp0 and img1Prev:
        LKp2, trace_status = checked_trace(img1Prev, img1, LKp1)

        LKp1 = LKp2[trace_status].copy()
        LKp0 = LKp0[trace_status].copy()
        img1Prev = img1

        if len(LKp0) < 4:
            LKp0 = None
        else:
            h, status = cv2.findHomography(LKp0, LKp1, 0, 10.0)
    else:
        img1Prev = img1

        """
        From opencv2refman.pdf, page 327, Section 3.7,
          "The function finds the most prominent corners in the image or in
            the specified image region, as described in [Shi94]"
        """
        print("feature_params (a dictionary) = %s" % str(feature_params))
        p = cv2.goodFeaturesToTrack(img1, **feature_params)

        if True:
            print "Alex: len(p)=%d, p = %s" % (len(p), str(p))

        if p is not None:
            """
            We display this information only at the first call of this function,
                BUT not in the next calls.
            """
            for x, y in p[:,0]:
                cv2.circle(img1, (x, y), 5, green, -1)
            common_cv.draw_str(img1, (20, 20),
                               "feature count (from goodFeaturesToTrack): %d" %
                               len(p))

    """
    Python: 
    cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, 
                            minDistance[, corners[, mask[, blockSize[, 
                            useHarrisDetector[, k ]]]]]) -> corners

    useHarrisDetector - Parameter indicating whether to use a Harris detector 
        (see cornerHarris()) or cornerMinEigenVal()
    k - Free parameter of the Harris detector (see page 324 for details on the 
        dst(x, y) formula using k)
    """
    LKp0 = cv2.goodFeaturesToTrack(img1, **feature_params)

    if LKp0 is not None:
        LKp1 = LKp0
        img1Prev = img1

    # TODO: think if this is a good idea
    return h, status
