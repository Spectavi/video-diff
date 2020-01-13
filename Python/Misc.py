import cv

# TODO: This class doesn't seem to be used and is based on old OpenCV bindings.
# Either finish the class or remove it.


def convert_np_to_cvmat(img_np):
    """
    This gives a: AttributeError: 'numpy.ndarray' object has no attribute
    'from_array'
    ImageAlignment.template_image = ImageAlignment.template_image.from_array()
    """
    # Inspired from https://stackoverflow.com/questions/5575108/how-to-convert-a-numpy-array-view-to-opencv-matrix :
    h_np, w_np = img_np.shape[:2]
    tmp_cv = cv.CreateMat(h_np, w_np, cv.CV_8UC3)
    cv.SetData(tmp_cv, img_np.data, img_np.strides[0])
    return tmp_cv


def convert_np_to_ipl_image(img_np):
    # Inspired from https://stackoverflow.com/questions/11528009/opencv-converting-from-numpy-to-iplimage-in-python
    # img_np is numpy array
    num_colors = 1

    bitmap = cv.CreateImageHeader((img_np.shape[1], img_np.shape[0]),
                                  cv.IPL_DEPTH_8U, num_colors)
    cv.SetData(bitmap, img_np.tostring(),
               img_np.dtype.itemsize * num_colors * img_np.shape[1])
    return bitmap
