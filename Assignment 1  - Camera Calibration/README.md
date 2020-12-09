# Assignment 1 (Camera Calibration) [![Maintenance](https://img.shields.io/badge/Python-3.7.x-brightgreen.svg)](https://www.python.org)

This project contains the self-designed python framework to perform calibration of any camera (demonstrated using an input image provided with the project). The report explains the methodology and the fundamentals of camera calibration. The code is well commented and contains the following functions:

  - **normalise**: Function that performs homogenization and normarlisation and return T and U transformation matrices which will be used later for denormalisation process
  - **denormalise**: Function to perform denormalisation on projection matrix. It takes T and U as input 
  - **DLT**: Function to perform Direct Linear Transform (Using Singular Value Decomposition) on normalised coordinates to return the normalised Projection matrix
  - **intrinsic**: Function to calculate intrinsic parameters of the camera such as camera optic center, focal length, skew etc
  - **extrinsic**: Function to calculate extrinsic parameters of the camera such as the rotation and translation parameters that are used to transform from the world coordinate frame to the camera coordinate frame
  - **ResizeWithAspectRatio**: Function used to resize the image window
  
# Installation

Our project use two python libraries:
- **numpy**: For performing matrix caluculations
- **open cv**: For displaying image and taking input from user
```sh
$ pip install opencv-python-headless
$ pip install numpy
```

To run the code

```sh
$ python3 assignment.py
```

# References
[1] https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
[2]  https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/

# Contributers
- Varun Gupta (2017EE30551)
- Sarthak Garg (2017EE30546)
