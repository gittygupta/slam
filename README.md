# slam
Simplest implementation of Simultaneous Localisation and Mapping using-
1.  OpenCV-Python for 2D display and feature extraction,
2.  [Pangolin](https://github.com/uoip/pangolin) for 3D mapping 
3.  [g2o (python)](https://github.com/uoip/pangolin) for optimisation of mapping [Done. To be improved further]

![output on a test video](https://github.com/gittygupta/slam/blob/master/output.png)

The focal length needs to be passed as the parameter F in bash shell along with the user's video directory. Moreover, a 'yes' or 'no' parameter has to be passed depending on the user's requirement of using a reverse SLAM.

example command : 
```
python3 slam.py F=500 PATH=videos/reversed.mp4 REVERSE=n
```
OR,

```
python3 slam.py 500 videos/reversed.mp4 n
```

To exit the program press 'q' on the 'frames' window.

To install [Pangolin](https://github.com/uoip/pangolin) and [g2o](https://github.com/uoip/g2opy) follow the tutorial on their github repository. Make sure you have the dependencies that are listed, installed on your machine.

### TODOS:
1. Add kinemtic model, becomes shitty later on. Fine initially though.
2. Add g2o optimization only to latest pose and not all the previous ones, which probably makes it slower and worse in the future.
3. Add optimizer for Focal length (different for different camera lens)

Might encounter Cholesky errors in certain iterations but it fixes itself. Currently usably fast.

Will keep updating. Check commits for history.

