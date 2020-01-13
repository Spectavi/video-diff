import os

import cv2
import numpy as np

import common
import config
from ecc_homo_spacetime import MyImageRead
import SpatialAlignment
import synchro_script


def align_videos(capture_q, capture_r):
    print("Entered VideoAlignmentEvangelidis.AlignVideos().")

    crossref = synchro_script.TemporalAlignment(capture_q, capture_r)

    if config.PREPROCESS_REFERENCE_VIDEO_ONLY:
        return

    print("VideoAlignmentEvangelidis.AlignVideos(): crossref = %s" % str(
          crossref))

    if config.SKIP_SPATIAL_ALIGNMENT:
        output_crossref_images(crossref, capture_q, capture_r)
    else:
        SpatialAlignment.spatial_alignment_evangelidis(crossref, capture_q,
                                                       capture_r)


def output_crossref_images(crossref, capture_q, capture_r):
    if not os.path.exists(config.FRAME_PAIRS_MATCHES_FOLDER):
        os.makedirs(config.FRAME_PAIRS_MATCHES_FOLDER)

    rframe = None
    gray_rframe = None
    last_r_idx = None
    t0 = float(cv2.getTickCount())
    for q_idx, r_idx in crossref:
        q_idx = int(q_idx)
        r_idx = int(r_idx)

        if common.MY_DEBUG_STDOUT:
            t1 = float(cv2.getTickCount())

        qframe = MyImageRead(capture_q, q_idx, grayscale=False)
        gray_qframe = common.ConvertImgToGrayscale(qframe)
        if r_idx != last_r_idx:
            rframe = MyImageRead(capture_r, r_idx, grayscale=False)
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

        if config.VISUAL_DIFF_FRAMES:
            # Compute difference between frames.
            diff_mask = cv2.absdiff(qframe, rframe)

            for i in range(diff_mask.shape[2]):
                diff_mask[:, :, i] = cv2.GaussianBlur(src=diff_mask[:, :, i],
                                                      ksize=(7, 7),
                                                      sigmaX=10)

            diff_mask_sum = diff_mask.sum(axis=2)
            r_diff, c_diff = np.nonzero(
                diff_mask_sum >= config.MEANINGFUL_DIFF_THRESHOLD *
                diff_mask.shape[2])
            meaningful_indices = zip(r_diff, c_diff)
            # Create all black image so we can set the diff pixels to white.
            diff_mask = np.zeros((diff_mask.shape[0], diff_mask.shape[1]),
                                 dtype=np.uint8)
            diff_mask[(r_diff, c_diff)] = 255

            assert r_diff.size == c_diff.size
            assert r_diff.size == len(meaningful_indices)

            res = cv2.findContours(
                image=diff_mask,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE)

            contours = res[1]

            color_c = 255
            meaningful_contours = 0
            diff_mask = np.zeros((diff_mask.shape[0], diff_mask.shape[1]),
                                 dtype=np.uint8)
            for index_c, contour in enumerate(contours):
                if len(contour) < 15:
                    continue
                else:
                    cv2.drawContours(
                        image=diff_mask,
                        contours=[contour],
                        contourIdx=0,
                        color=color_c,
                        thickness=-1)
                    meaningful_contours += 1

            cv2.imwrite(config.FRAME_PAIRS_MATCHES_FOLDER +
                        ("%.6d_diff_mask" % q_idx) + config.imformat,
                        diff_mask.astype(int))

            if config.SHOW_MASKED_DIFF:
                qframe_diff = cv2.bitwise_and(qframe, qframe, mask=diff_mask)
                rframe_diff = cv2.bitwise_and(rframe, rframe, mask=diff_mask)
                # Save the masked diffs.
                cv2.imwrite(config.FRAME_PAIRS_MATCHES_FOLDER +
                            ("%.6d_query_diff" % q_idx) + config.imformat,
                            qframe_diff.astype(int))
                cv2.imwrite(config.FRAME_PAIRS_MATCHES_FOLDER +
                            ("%.6d_ref_diff" % q_idx) + config.imformat,
                            rframe_diff.astype(int))

        if config.SAVE_FRAMES:
            # We use q_idx because ref video is aligned against the query video.
            # These frames represent the aligned output.
            cv2.imwrite(config.FRAME_PAIRS_MATCHES_FOLDER +
                        ("%.6d_query" % q_idx) + config.imformat,
                        qframe.astype(int))
            cv2.imwrite(config.FRAME_PAIRS_MATCHES_FOLDER +
                        ("%.6d_ref" % q_idx) + config.imformat,
                        rframe.astype(int))

        cv2.imwrite(config.FRAME_PAIRS_MATCHES_FOLDER +
                    ("%.6d_gchan_diff" % q_idx) + config.imformat,
                    gdiff.astype(int))

    print("Saving crossref frames took %.6f [sec]" % (
        (float(cv2.getTickCount()) - t0) / cv2.getTickFrequency()))
