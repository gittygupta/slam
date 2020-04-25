import cv2
import sys
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
from frame import Frame, denormalise, match_frames
from mapp import Map, Point

args = sys.argv
 
# Colors
magenta = (255, 0, 255)
cyan = (255, 255, 0)

# Camera insintrics (settable)
W = 854
H = 480
 
# Environment variables
F = int(args[1])
PATH = args[2]
REVERSE = args[3]

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
    if frame.id == 0:
        return
    
    # Match
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)

    for i, idx in enumerate(idx2):
        if f2.pts[idx] is not None:
            f2.pts[idx].add_observation(f1, idx1[i])

    good_pts4d = np.array([f1.pts[i] is None for i in idx1])
    
    # points locally in front of the camera
    # Reject some points without enough parallax
    pts_tri_local = triangulate(Rt, np.eye(4), f1.kps[idx1], f2.kps[idx2])
    good_pts4d &= np.abs(pts_tri_local[:, 3]) > 0.005

    # Homogeneous 3D coordinates
    # reject points behind the camera
    pts_tri_local /= pts_tri_local[:, 3:]
    good_pts4d &= pts_tri_local[:, 2] > 0

    # project into world
    pts4d = np.dot(np.linalg.inv(f1.pose), pts_tri_local.T).T

    print('Adding ' + str(np.sum(good_pts4d)) + ' points')

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


if REVERSE == 'y' or REVERSE == 'Y':
    # play backward
    cap = cv2.VideoCapture(PATH)
    frame_idx = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
    print(frame_idx)
    while(cap.isOpened() and frame_idx >= 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        frame_idx -= 1
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(3000)
            break
        process(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

else:
    # play forward
    cap = cv2.VideoCapture(PATH)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(3000)
            break
        process(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
