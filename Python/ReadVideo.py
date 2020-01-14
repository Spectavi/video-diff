from __future__ import print_function

import os

import cv2
import sys

# Find max number of matched features with Simulated annealing
import common
import config

if config.OCV_OLD_PY_BINDINGS:
    import cv


if not config.USE_EVANGELIDIS_ALGO:
    import MatchFrames
    import SimAnneal


captureQ = None
frameCountQ = None
captureR = None
frameCountR = None

resVideoQ = (-1, -1)
resVideoR = (-1, -1)


# Do performance benchmarking
def benchmark():
    global captureQ, frameCountQ
    global captureR, frameCountR

    # Doing benchmarking
    while True:
        if config.OCV_OLD_PY_BINDINGS:
            frame1 = captureQ.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        else:
            frame1 = captureQ.get(cv2.CAP_PROP_POS_FRAMES)
        common.DebugPrint("Alex: frame1 = %d" % frame1)

        MatchFrames.counter_q = int(frame1)

        common.DebugPrint(
            "Alex: MatchFrames.counter_q = %d" % MatchFrames.counter_q)

        ret_q, img_q = captureQ.read()
        if not ret_q:
            break

        # TODO: This method below doesn't seem to exist?
        # MatchFrames.Main_img1(img_q, MatchFrames.counter_q)

    # 36.2 secs (38.5 secs with Convert to RGB)
    common.DebugPrint("Alex: time after Feature Extraction of all frames of "
                      "video 1 = %s" %
                      common.GetCurrentDateTimeStringWithMilliseconds())

    while True:
        if config.OCV_OLD_PY_BINDINGS:
            frame_r = captureR.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        else:
            frame_r = captureR.get(cv2.CAP_PROP_POS_FRAMES)

        common.DebugPrint("Alex: frame_r = %d" % frame_r)

        MatchFrames.counter_r = int(frame_r)

        common.DebugPrint("Alex: counter_r = %d" % MatchFrames.counter_r)

        ret_r, img_r = captureR.read()
        if not ret_r:
            break

        # TODO: This method below doesn't seem to exist?
        # MatchFrames.Main_img2(img_r, MatchFrames.counter_r)

    # Note: 47.2 secs (56.7 secs with Convert to RGB)
    common.DebugPrint("Alex: time after Feature Extraction of all frames of "
                      "video 2 and (FLANN?) matching once for each frame = %s" %
                      common.GetCurrentDateTimeStringWithMilliseconds())

    quit()


# videoType = 0 --> query (input), 1 --> reference
# OpenCV can read AVIs (if no ffmpeg support installed it can't read MP4,
# nor 3GP, nor FLVs with MPEG compression)
# From http://answers.opencv.org/question/6/how-to-readwrite-video-with-opencv-in-python/
def open_video_capture(video_file_name, video_type):

    """
    Unfortunately, normally cv2.VideoCapture() continues even if it does not
        find videoPathFileName
    """
    assert os.path.isfile(video_file_name)

    # From http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-videocapture

    capture = cv2.VideoCapture(video_file_name)
    # Inspired from https://stackoverflow.com/questions/16703345/how-can-i-use-opencv-python-to-read-a-video-file-without-looping-mac-os

    if config.OCV_OLD_PY_BINDINGS:
        frame_count = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    else:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if config.OCV_OLD_PY_BINDINGS:
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, config.initFrame[video_type])
    else:
        capture.set(cv2.CAP_PROP_POS_FRAMES, config.initFrame[video_type])

    if config.OCV_OLD_PY_BINDINGS:
        width = capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
        codec = capture.get(cv2.cv.CV_CAP_PROP_FOURCC)
    else:
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        codec = capture.get(cv2.CAP_PROP_FPS)

    assert width < 32767  # we use np.int16
    assert height < 32767  # we use np.int16

    duration = frame_count / fps
    print("Video '%s' has resolution %dx%d, %.2f fps and "
          "%d frames, duration %.2f secs, codec=%s" %
          (video_file_name, width, height, fps,
           frame_count, duration, codec))

    steps = [config.counterQStep, config.counterRStep]
    # TODO: take into account also initFrame
    used_frame_count = frame_count / steps[video_type]
    assert not((video_type == 0) and (used_frame_count <= 10))
    common.DebugPrint(
        "We use video '%s', with %d frames, from which we use ONLY %d\n" %
        (video_file_name, frame_count, used_frame_count))

    resolution = (width, height)

    return capture, frame_count, resolution


# Note: the extension of the videoPathFileName needs to be the same as the
# fourcc, otherwise, gstreamer, etc can return rather criptic error messages.
def write_video_capture(video_file_name, folder_name):
    # OpenCV can read only AVIs - not 3GP, nor FLVs with MPEG compression
    # From http://answers.opencv.org/question/6/how-to-readwrite-video-with-opencv-in-python/

    video_writer = None

    folder_content = os.listdir(folder_name)
    sorted_folder_content = sorted(folder_content)

    for fileName in sorted_folder_content:
        path_file_name = folder_name + "/" + fileName
        if os.path.isfile(path_file_name) and \
           fileName.lower().endswith("_good.png"):

            common.DebugPrint("ComputeHarlocs(): Loading %s" % path_file_name)
            img = cv2.imread(path_file_name)
            assert img is not None

            if video_writer:
                common.DebugPrint("img.shape = %s" % str(img.shape))
                # From http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#saving-a-video
                # See also http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
                # WRITES 0 BYTES IN THE VIDEO: vid_fourcc = cv2.VideoWriter_fourcc('M','J','P','G');

                # See also http://www.fourcc.org/codecs.php
                vid_fourcc = cv2.VideoWriter_fourcc(*'XVID')

                video_writer = cv2.VideoWriter(filename=video_file_name,
                                               fourcc=vid_fourcc, fps=10,
                                               frameSize=(
                                                   img.shape[1], img.shape[0]))
            else:
                common.DebugPrint("Error in creating video writer")
                sys.exit(1)

            video_writer.write(img)

    video_writer.release()
    common.DebugPrint("Finished writing the video")
    return


def main(video_file_q, video_file_r):
    global captureQ, frameCountQ
    global captureR, frameCountR
    global resVideoQ, resVideoR

    if not config.USE_EVANGELIDIS_ALGO:
        if not os.path.exists(config.IMAGES_FOLDER):
            os.makedirs(config.IMAGES_FOLDER)
        if not os.path.exists(config.FRAME_PAIRS_FOLDER):
            os.makedirs(config.FRAME_PAIRS_FOLDER)
        if not os.path.exists(config.FRAME_PAIRS_MATCHES_FOLDER):
            os.makedirs(config.FRAME_PAIRS_MATCHES_FOLDER)

    total_t1 = float(cv2.getTickCount())

    if config.OCV_OLD_PY_BINDINGS:
        common.DebugPrint("dir(cv) = %s" % str(dir(cv)))
    common.DebugPrint("dir(cv2) = %s" % str(dir(cv2)))

    if not config.USE_EVANGELIDIS_ALGO:
        f_output = open("output.txt", "w")
        print(
            "Best match for frames from (input/current) video A w.r.t. "
            "reference video B:",
            file=f_output)

    captureQ, frameCountQ, resVideoQ = open_video_capture(video_file_q, 0)
    captureR, frameCountR, resVideoR = open_video_capture(video_file_r, 1)

    common.DebugPrint("Main(): frameCountQ = %d" % frameCountQ)
    common.DebugPrint("Main(): frameCountR = %d" % frameCountR)

    """
    In case the videos have different resolutions an error will actually take
        place much longer when using Evangelidis' algorithm, in spatial
        alignment, more exactly in Matlab.interp2():

      File "/home/asusu/drone-diff/Backup/2014_03_25/Matlab.py", line 313, in interp2
        V4 = np.c_[V[1:, 1:] * xM[:-1, :-1] * yM[:-1, :-1], nanCol1];
        ValueError: operands could not be broadcast together with shapes (719,1279) (239,319)
    """
    assert resVideoQ == resVideoR

    if not config.USE_EVANGELIDIS_ALGO:
        SimAnneal.LIMIT = frameCountR
        SimAnneal.captureR = captureR

    print("ReadVideo.Main(): time before pre_main() = %s" %
          common.GetCurrentDateTimeStringWithMilliseconds())

    if not config.USE_EVANGELIDIS_ALGO:
        MatchFrames.pre_main(n_frames_q=frameCountQ, n_frames_r=frameCountR)

    common.DebugPrint("Main(): time after pre_main() = %s" %
                      common.GetCurrentDateTimeStringWithMilliseconds())

    # Here we start the (main part of the) algorithm

    # Distinguish between Alex's alignment algo and Evangelidis's algo
    if config.USE_EVANGELIDIS_ALGO:
        import VideoAlignmentEvangelidis
        VideoAlignmentEvangelidis.align_videos(captureQ, captureR)
    else:
        MatchFrames.process_input_frames(captureQ, captureR, f_output)

    if config.USE_GUI:
        cv2.destroyAllWindows()

    if not config.USE_EVANGELIDIS_ALGO:
        f_output.close()

    captureQ.release()
    captureR.release()

    total_t2 = float(cv2.getTickCount())
    my_time = (total_t2 - total_t1) / cv2.getTickFrequency()

    print("ReadVideo.Main() took %.6f [sec]" % my_time)
