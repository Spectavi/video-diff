from absl import app
from absl import flags
import sys

import cv2
import numpy as np
import scipy

import common
import config
import ReadVideo

FLAGS = flags.FLAGS

flags.DEFINE_boolean("preprocess_ref", False,
                     "Preprocess only the reference video.")
flags.DEFINE_boolean("process_query_and_align_videos", False,
                     "Preprocess both videos and perform alignment.")


def ask_first():
    print("To speed up the video alignment on future runs, we save intermediate"
          " results for later reuse:\n" 
          "   - Harris features and\n"
          "   - matrices computed in the decision step of temporal alignment.\n"
          "In case you do NOT want to use them we invite you to "
          "delete this data from the local folder(s) yourself, otherwise we can"
          " obtain WRONG results, "
          "if the saved data is not corresponding to the videos analyze.\n"
          "Are you OK to continue and use any of these intermediate results, if"
          " any?\n")

    choice = raw_input("Press Y to continue or N to quit:").lower()
    if choice == "n":
        quit()
    elif choice == "y":
        return
    return

def main(argv):
    assert len(sys.argv) >= 3
    if FLAGS.preprocess_ref:
        config.PREPROCESS_REFERENCE_VIDEO_ONLY = True
    elif FLAGS.process_query_and_align_videos:
        config.PREPROCESS_REFERENCE_VIDEO_ONLY = False
    else:
        config.PREPROCESS_REFERENCE_VIDEO_ONLY = False

    print("config.PREPROCESS_REFERENCE_VIDEO_ONLY = %s" % str(
        config.PREPROCESS_REFERENCE_VIDEO_ONLY))

    ask_first()

    # Inspired from https://stackoverflow.com/questions/1520234/how-to-check-which-version-of-numpy-im-using
    print("numpy.version.version = %s" % str(np.version.version))
    print("scipy.version.version = %s" % str(scipy.version.version))
    np.show_config()
    scipy.show_config()

    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    # We use 7 digits precision and suppress using scientific notation.
    np.set_printoptions(precision=7, suppress=True,
                        threshold=70000, linewidth=4000)

    # Inspired from \OpenCV2-Python-Tutorials-master\source\py_tutorials\py_core\py_optimization
    # normally returns True - relates to using the SIMD extensions of x86:
    # SSX, AVX
    common.DebugPrint("cv2.useOptimized() is %s" % str(cv2.useOptimized()))

    """
    From http://docs.opencv.org/modules/core/doc/utility_and_system_functions_and_macros.html#checkhardwaresupport
        CV_CPU_MMX - MMX
        CV_CPU_SSE - SSE
        CV_CPU_SSE2 - SSE 2
        CV_CPU_SSE3 - SSE 3
        CV_CPU_SSSE3 - SSSE 3
        CV_CPU_SSE4_1 - SSE 4.1
        CV_CPU_SSE4_2 - SSE 4.2
        CV_CPU_POPCNT - POPCOUNT
        CV_CPU_AVX - AVX
    """
    # TODO: Figure out the correct way to reference these in OpenCV 3.x
    """
    # Need to setUseOptimized before calling checkHardwareSupport
    cv2.setUseOptimized(True)
    if config.OCV_OLD_PY_BINDINGS == False:
        featDict = {cv2.CpuFeatures.CV_CPU_AVX: "AVX",
                cv2.CPU_MMX: "MMX",
                cv2.CPU_NEON: "NEON",
                cv2.CPU_POPCNT: "POPCNT",
                cv2.CPU_SSE: "SSE",
                cv2.CPU_SSE2: "SSE2",
                cv2.CPU_SSE3: "SSE3",
                cv2.CPU_SSE4_1: "SSE4.1",
                cv2.CPU_SSE4_2: "SSE4.2",
                cv2.CPU_SSSE3: "SSSE3"}

        for feat in featDict:
            res = cv2.checkHardwareSupport(feat)
            print("%s = %d" % (featDict[feat], res))
    """

    # "Returns the number of logical CPUs available for the process."
    common.DebugPrint("cv2.getNumberOfCPUs() (#logical CPUs) is %s" % str(
        cv2.getNumberOfCPUs()))
    common.DebugPrint(
        "cv2.getTickFrequency() is %s" % str(cv2.getTickFrequency()))

    video_file_q = sys.argv[1]  # input/current video
    video_file_r = sys.argv[2]  # reference video

    # TODO: use getopt() to run Evangelidis' or "Alex's" algorithm, etc

    ReadVideo.Main(video_file_q, video_file_r)


if __name__ == '__main__':
    app.run(main)
