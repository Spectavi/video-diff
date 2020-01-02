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

    # We allocate space for xxx
    xxx = np.zeros(qframe.shape)
    xxx[:, :, 0] = gray_qframe
    xxx[:, :, 2] = gray_qframe

    if config.VISUAL_DIFF_FRAMES:
      xxx[:, :, 1] = gray_qframe
    else:
      xxx[:, :, 1] = gray_rframe  # This is the original behavior

    ############################# DIFF FRAMES ##############################
    if config.VISUAL_DIFF_OUTPUT:
      xxx = qframe.copy()

      # HStack the diff and original query and ref frames.
      merged = np.hstack((qframe, rframe))

      # Compute difference between frames inputFrame and refFrame:
      xxx = cv2.absdiff(qframe, rframe)

      for i in range(xxx.shape[2]):
        xxx[:, :, i] = cv2.GaussianBlur(src=xxx[:, :, i],
                                        ksize=(7, 7),
                                        sigmaX=10)

      xxxSum = xxx.sum(axis=2)
      rDiff, cDiff = np.nonzero(
        xxxSum >= config.MEANINGFUL_DIFF_THRESHOLD * xxx.shape[2])
      meaningfulIndices = zip(rDiff, cDiff)
      # Create all black image so we can set the diff pixels to white.
      xxx = np.zeros((xxx.shape[0], xxx.shape[1]), dtype=np.uint8)
      xxx[(rDiff, cDiff)] = 255

      assert rDiff.size == cDiff.size
      assert rDiff.size == len(meaningfulIndices)

      """
      See http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours
      cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
      """
      res = cv2.findContours(
          image=xxx,
          mode=cv2.RETR_TREE,
          method=cv2.CHAIN_APPROX_SIMPLE)  # Gives error: <<ValueError: too many values to unpack>>

      contours = res[1]

      colorC = 255
      meaningfulContours = 0
      xxx = np.zeros((xxx.shape[0], xxx.shape[1]), dtype=np.uint8)
      for indexC, contour in enumerate(contours):
        if len(contour) < 15:
          continue
        else:
          """
          From http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#drawcontours
              <<thickness - Thickness of lines the contours are drawn with. If it is negative (for example, thickness=CV_FILLED ), the contour interiors are drawn.>>
          """
          cv2.drawContours(
              image=xxx,
              contours=[contour],
              contourIdx=0,
              color=colorC,
              thickness=-1)
          meaningfulContours += 1
      ############################# END DIFF FRAMES ##########################

    if config.VISUAL_DIFF_OUTPUT:
      masked_qframe = cv2.bitwise_and(qframe, qframe, mask=xxx)
      masked_rframe = cv2.bitwise_and(rframe, rframe, mask=xxx)

      xxx = np.hstack((merged, masked_qframe, masked_rframe))
      fileNameOImage = "%.6d_good_diff" % q_idx
    else:
      fileNameOImage = "%.6d_good" % q_idx

    cv2.imwrite(o_path + fileNameOImage + config.imformat,
                xxx.astype(int))
    print("Saving crossref frames took %.6f [sec]" % (
          (float(cv2.getTickCount()) - t1) / cv2.getTickFrequency()))
