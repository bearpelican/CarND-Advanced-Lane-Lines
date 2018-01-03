**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/user_added/undistorted_chessboard_1.png "Undistorted"
[image2]: ./examples/user_added/undistorted_chessboard_11.png "Undistorted 11"
[image3]: ./examples/user_added/undistorted_lane_2.png "Road Transformed"
[image4]: ./examples/user_added/threshold_compare.png "Threshold comparison"
[image5]: ./examples/user_added/perspective_warp.png "Perspective Warped"
[image6]: ./examples/user_added/centroids.png "Find Centroids"
[image7]: ./examples/user_added/poly_fit.png "Fit lane line"
[image8]: ./examples/user_added/draw_lanes.png "Lane Identification"
[image9]: ./examples/user_added/color_threshold_imgs.png "Color Threshold"
[image10]: ./examples/user_added/gradient_threshold_imgs.png "Gradient Threshold"


## Writeup / README

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in [Camera-Calibration.ipynb](./Camera-Calibration.ipynb).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `get_objpoints()` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `Threshold.ipynb`).

For color threshold, I used the HLS, LAB, and LUV colorspaces to filter for white and yellow lane lines. Here are the values:
* LS(both): MIN[0,100,90] MAX[40,255,255]
* LAB(yellow): MIN[0,0,150] MAX[255,255,255]
* LUV(white): MIN[210,0,0] MAX[255,255,255]

You can find this in [Threshold.ipynb](./Camera-Calibration.ipynb) Code cell 10

![alt text][image9]

For gradient threshold, I used a combination of sobel absolute, magnitude and direction filters.  
[Threshold.ipynb](./Camera-Calibration.ipynb) Code cell 8  
* Absolute_X:[20,100]
* Absolute_Y:[20,100]
* Magnitude:[30,100]
* Direction:[.7,1.3]

![alt text][image10] 

Here's an example of the combined gradient and color images.  
(Color thresholding in blue, gradient threshold in green)

![alt text][image4]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in 10th code cell in the IPython notebook [LaneDrawing.ipynb](./LaneDrawing.ipynb).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

LaneDrawing.ipynb - Cell 10
```python
def get_src(img):
    h, w = img.shape[:2]
    mid1, mid2 = [(w*.408), (h*.7)], [(w*.6), (h*.7)]
    bot1, bot2 = [(w*.16), h], [(w*.865), h]
    vertices = [mid1, mid2, bot2, bot1]
    return vertices

def get_dst(img):
    h, w = img.shape[:2]
    offsetx=100
    offsety=-10
    mid1, mid2 = [offsetx, offsety], [w-offsetx, offsety]
    bot1, bot2 = [offsetx, h], [w-offsetx, h]
    dst = [mid1, mid2, bot2, bot1]
    return dst
```


| Source        | Destination   |  
|:-------------:|:-------------:|  
| 522, 504      | 100, -10      |  
| 205, 720      | 100, 720      |  
| 768, 504      | 1180, -10     |  
| 1107, 720     | 1180, 720     |  

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used a sliding window search on the threshold + perspective transformed image.  
[LaneDrawing.ipynb](./LaneDrawing.ipynb) Code cell 11

This method convolves a filter over an image region to find the area with the maximun number of ativated pixels. In the case of our images, the areas that are most activated AKA "window centroid" should be the lane lines.  
The image is separated into left and right regions to find their respective lane lines. We then move the filter horizontally on each region (starting from the bottom) to find the first window centroid. The next centroid is found by convolving the filter in the horizontal segment right above where the previous centroid was found.

Here are the parameters I used:  
* Window size - 70x90
* Margin - 70
* Filter - Gaussian
* Starting Left Region Search Area - (512, 480), (512, 720), (0, 720), (0, 480)
* Starting Right Region Search Area - (768, 480), (768, 720), (0, 720), (0, 480)


I used a gaussian filter over a ones-matrix so that the window location would be centered. With a ones filter, the window location will be slightly off if the window is too big or too small - multiple maximum locations.  
The window centroid is discarded if it falls between a certain threshold (2500). The x-coordinate for the previous found centroid was used instead


![alt text][image6]

A 2nd order polynomial was used to fit the window centroids.  
[LaneDrawing.ipynb](./LaneDrawing.ipynb) Code cell 12

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated in [LaneDrawing.ipynb](./LaneDrawing.ipynb) in the function `curvature_meters()` - Code cell 15


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step on `Cell 17` in [LaneDrawing.ipynb](./LaneDrawing.ipynb) in the function `overlay_img()`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/project_video_extended.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found that this pipeline could be fine tuned pretty well for a specific set of roads and conditions. However, it's much harder to find a set of parameters that work for the majority of the cases.  

The pipeline fares poorly in the challenge video due to lots of changing conditions. Here are a few obvious problems:
1. Shadows - guard rail shadows and bridge shadows
    * Overhead bridge shadow: I tried using histogram equalization for better contrast. However, further work needed to be done to determine whether an image needed this equalization or not.
    * Gaurd rail shadow: I added a lot more weight towards the color thresholds, so that yellow and white lines would have more weight than black lines. However, window searching still failed as the lane got farther out.

2. Error handling -
    * Window centroid
        * sometimes there are outlier centroids when searching on dotted white lines. It could help to try and detect these outliers.
        * if a centroid cannot be found, we use the location of the previous one. It could be worth trying to extrapolate where the next one should be, in this case


### Challenge

Here's a [link to my video result](./output_video/challenge_video_1.mp4)