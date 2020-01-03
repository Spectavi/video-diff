import numpy as np

import common
import config
import cv2
from ecc_homo_spacetime import MyImageRead
import synchro_script
import SpatialAlignment


def AlignVideos(captureQ, captureR):
    print("Entered VideoAlignmentEvangelidis.AlignVideos().")

    crossref = synchro_script.TemporalAlignment(captureQ, captureR)

    if config.PREPROCESS_REFERENCE_VIDEO_ONLY == True:
        return

    print(
        "VideoAlignmentEvangelidis.AlignVideos(): crossref = %s" % str(crossref))

    if config.SKIP_SPATIAL_ALIGNMENT:
        OutputCrossrefImages(crossref, captureQ, captureR)
    else:
        SpatialAlignment.SpatialAlignmentEvangelidis(crossref, captureQ, captureR)


def OutputCrossrefImages(crossref, captureQ, captureR):
    rframe = None
    gray_rframe = None
    last_r_idx = None
    for q_idx, r_idx in crossref:
        q_idx = int(q_idx)
        r_idx = int(r_idx)

        if common.MY_DEBUG_STDOUT:
            t1 = float(cv2.getTickCount())

        qframe = MyImageRead(captureQ, q_idx, grayscale=False)
        gray_qframe = common.ConvertImgToGrayscale(qframe)
        if r_idx != last_r_idx:
            rframe = MyImageRead(captureR, r_idx, grayscale=False)
            gray_rframe = common.ConvertImgToGrayscale(rframe)

        last_r_idx = r_idx

        if common.MY_DEBUG_STDOUT:
            t2 = float(cv2.getTickCount())
            common.DebugPrint(
                "ecc_homo_spacetime(): grabbing frames took %.6f [sec]" %
                ((t2 - t1) / cv2.getTickFrequency()))

        # Default diff is a G-channel swap.
        gdiff = np.zeros(qframe.shape)
        gdiff[:, :, 0] = gray_qframe
        gdiff[:, :, 1] = gray_rframe  # G-Channel Swap
        gdiff[:, :, 2] = gray_qframe

        ############################# DIFF FRAMES ##############################
        if config.VISUAL_DIFF_FRAMES:
            # Compute difference between frames.
            diff_mask = cv2.absdiff(qframe, rframe)

            for i in range(diff_mask.shape[2]):
                diff_mask[:, :, i] = cv2.GaussianBlur(src=diff_mask[:, :, i],
                                                ksize=(7, 7),
                                                sigmaX=10)

            xxxSum = diff_mask.sum(axis=2)
            rDiff, cDiff = np.nonzero(
                xxxSum >= config.MEANINGFUL_DIFF_THRESHOLD * diff_mask.shape[2])
            meaningfulIndices = zip(rDiff, cDiff)
            # Create all black image so we can set the diff pixels to white.
            diff_mask = np.zeros((diff_mask.shape[0], diff_mask.shape[1]),
                                 dtype=np.uint8)
            diff_mask[(rDiff, cDiff)] = 255

            assert rDiff.size == cDiff.size
            assert rDiff.size == len(meaningfulIndices)

            """
            See http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours
            cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
            """
            res = cv2.findContours(
                image=diff_mask,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE)  # Gives error: <<ValueError: too many values to unpack>>

            contours = res[1]

            colorC = 255
            meaningfulContours = 0
            diff_mask = np.zeros((diff_mask.shape[0], diff_mask.shape[1]),
                                 dtype=np.uint8)
            for indexC, contour in enumerate(contours):
                if len(contour) < 15:
                    continue
                else:
                    cv2.drawContours(
                        image=diff_mask,
                        contours=[contour],
                        contourIdx=0,
                        color=colorC,
                        thickness=-1)
                    meaningfulContours += 1

            cv2.imwrite(config.FRAME_PAIRS_MATCHES_FOLDER +
                        ("%.6d_diff_mask" % q_idx) + config.imformat,
                        diff_mask.astype(int))

            if config.SHOW_MASKED_DIFF:
                qframe_diff = cv2.bitwise_and(qframe, qframe, mask=diff_mask)
                rframe_diff = cv2.bitwise_and(rframe, rframe, mask=diff_mask)
                # Save the masked diffs.
                cv2.imwrite(config.FRAME_PAIRS_MATCHES_FOLDER +
                            ("%.6d_query_masked_diff" % q_idx) + config.imformat,
                            qframe_diff.astype(int))
                cv2.imwrite(config.FRAME_PAIRS_MATCHES_FOLDER +
                            ("%.6d_ref_masked_diff" % q_idx) + config.imformat,
                            rframe_diff.astype(int))

    # We use q_idx because the ref video is aligned against the query video.
    # These frames represent the aligned output.
    cv2.imwrite(config.FRAME_PAIRS_MATCHES_FOLDER +
                ("%.6d_query" % q_idx) + config.imformat,
                qframe.astype(int))
    cv2.imwrite(config.FRAME_PAIRS_MATCHES_FOLDER +
                ("%.6d_ref" % q_idx) + config.imformat,
                rframe.astype(int))

    print("Saving crossref frames took %.6f [sec]" % (
        (float(cv2.getTickCount()) - t1) / cv2.getTickFrequency()))
