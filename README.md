## Traffic Sign Classifier using Convolutional Neural Networks
This project consists of classifying (German) traffic signs using a convolutional neural network (CNN). Below is a sampling of the dataset with each row pertaining to a unique sign or class (stop, slippery road, etc). The write-up is also available at [www.lrgonzales.com/traffic-lane-recognition](http://www.lrgonzales.com/traffic-lane-recognition).

<p align="center">
  <img src="https://user-images.githubusercontent.com/4633154/38846095-7fe42122-41c8-11e8-9755-01c13528094b.jpg" width="480px" height="270px"/>
</p>

### Introduction
Classifying street signs is a challenging and important real-world problem, particularly with the promise of self-driving cars. The environment in which classification takes place is relatively constrained in that street signs are typically standardized for a given geographical region and the camera/s used to "see" the traffic signs is/are assumed to be stationary with respect to an observant vehicle and positioned upright. However, varied lighting and weather conditions — and even blur due to velocity — are expected.

### Dataset
The algorithm has three steps, summarized below.


### Usage
The program can be performed on a `jpg` or `mp4` file from the `input` directory by executing `python lane_recog.py input/<file>` where `<file>` is the desired input.

`lane_recog.py` does not make use of `src/edge_det.py` or `src/hough.py`. The `cv2.HoughLinesP` usage in `lane_recog.py` can be replaced with the method in `src/hough.py`, but usage of `src/edge_det.py` is not recommended in this particular application because it does not perform hysteresis thresholding.

### Improvements
Throughout the design, it became apparent that the methods used rely on a fairly smooth road, free of obstacles and relatively free of debris. For this reason, the input was preprocessed with a Gaussian blur. However, a Gaussian blur may not be enough for some unusual road surfaces or when obstacles or large debris are present. The danger is that these features could be mistaken for a lane in the algorithm. More advanced techniques, such as convolutional neural networks, could be more robust to such non-idealities.

The `src/hough.py` implementation of the Hough transform could benefit from an accumulator implemented with more efficient storage than a `list`. Doing so would likely result in on-par performance to the OpenCV implementation.

### Dependencies
`lane_recog.py` makes use of `numpy`, `matplotlib`, `cv2`, and `moviepy` (for `.mp4` input files).
