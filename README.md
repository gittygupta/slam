# slam
Simplest implementation of Simultaneous Localisation and Mapping using-
1.  OpenCV-Python for 2D display and feature extraction,
2.  Pangolin for 3D mapping 
3.  g2o (python) for optimisation of mapping [Done. To be improved further]

![output on a test video](https://github.com/gittygupta/slam/blob/master/output.png)

Will keep updating. Check commits for history.

Might encounter Cholesky errors in certain iterations but it fixes itself. Currently usably fast.

### TODO:
1. Add search by projections
2. Remove Essential matrix once tracked
3. Add kinemtic model, becomes shitty later on. Fine initially though.
4. Add g2o optimization only to latest pose and not all the previous ones, which probably makes it slower and worse in the future.
5. Add optimizer for F (different for different videos)
6. check is points are in the FOV of the camera
7. Improve to not need 'REVERSE' check in os env

Will add environment variables to switch between backward and forward SLAM. And, to add custom videos too, and adjustable focal length of camera.


