# Added colors to points

import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
from frame import Frame, denormalise, match_frames
from mapp import Map, Point
 
# Colors
magenta = (255, 0, 255)
cyan = (255, 255, 0)

# Camera insintrics
W = 854
H = 480
 
# Focal Length
F = 500     # varies for videos, make it a parameter

# Global Variables
K = np.array(([F, 0, W // 2], [0, F, H // 2], [0, 0, 1]))
Kinv = np.linalg.inv(K) 

# Main classes    
mapp = Map()
mapp.create_viewer()
 
# Utility
def display(frame):
    cv2.imshow('frames', frame)

def triangulate(pose1, pose2, pts1, pts2):
    # linear triangulation method
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    
    return ret

    #return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T
 
def process(image):
    image = cv2.resize(image, (W, H))
    frame = Frame(mapp, image, K)
    if frame.id == 0:       # i.e just the start frame
        return
    
    # Match
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]
    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)

    for i, idx in enumerate(idx2):
        if f2.pts[idx] is not None:
            f2.pts[idx].add_observation(f1, idx1[i])

    # Homogeneous 3D coords
    pts4d = triangulate(f1.pose, f2.pose,
            f1.kps[idx1], f2.kps[idx2])
    pts4d /= pts4d[:, 3:]
    
    # Reject some points without enough parallax
    unmatched_points = np.array([f1.pts[i] is None for i in idx1])
    print('Adding ' + str(np.sum(unmatched_points)) + ' points')
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points
    
    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        
        # Adding color
        u, v = int(f1.kpus[idx1[i], [0]]), int(f1.kpus[idx1[i], [1]])
        color = image[v][u]
        
        pt = Point(mapp, p, color)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])
    
    
    num_matches = 0
    for pt1, pt2 in zip(f1.kps[idx1], f2.kps[idx2]):
        num_matches += 1
        u1, v1 = denormalise(K, pt1)
        u2, v2 = denormalise(K, pt2)
        cv2.circle(image, (u1, v1), color=magenta, radius=3)
        cv2.line(image, (u1, v1), (u2, v2), color=cyan, thickness=1)
    print('Matches : ', num_matches)
    
    # 2D display
    display(image)
    
    # optimize 3d display
    if frame.id >= 4: 
        error_units = mapp.optimize()
        print('Units of error : %d' % error_units)

    # 3D display
    mapp.display_map()



print('Vary F if required ')
 
cap = cv2.VideoCapture('videos/test_vid1.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        cv2.waitKey(3000)
        break
    process(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
 
 

