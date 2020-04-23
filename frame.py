import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
 
# Colors
magenta = (255, 0, 255)
cyan = (255, 255, 0)

# Global Variables
 
# Utility functions
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
 
def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret

# pose
def extractRt(F):
    # E -> Essential Matrix
    # F -> Fundamental Matrix
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(F)
    '''assert np.linalg.det(U) > 0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0'''
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    return poseRt(R, t)
 
def normalise(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]
 
def denormalise(K, pt):
    ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
    ret /= ret[2]
    return int(ret[0]), int(ret[1])
 
def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)
    
    # Lowe's Ratio Test
    ret = []
    idx1, idx2 = [], []
    
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            '''
            # keep indices
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)
            
            p1 = f1.features[m.queryIdx]
            p2 = f2.features[m.trainIdx]

            ret.append((p1, p2))
            '''
            p1 = f1.kps[m.queryIdx]
            p2 = f2.kps[m.trainIdx]
            
            # travel less than 10% of diagonal and be within orb distance 32
            if np.linalg.norm((p1 - p2)) < 0.1 * np.linalg.norm([f1.w, f1.h]) and m.distance < 32:
                # keep indices
                # TODO
                if m.queryIdx not in idx1 and m.trainIdx not in idx2 : 
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                
                    ret.append((p1, p2))

    # no duplicates (duplicates cause problems with optimization)
    assert(len(set(idx1))) == len(idx1)
    assert(len(set(idx2))) == len(idx2)


    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    
    # Fit matrix
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            FundamentalMatrixTransform,
                            #EssentialMatrixTransform,
                            min_samples=8,
                            #residual_threshold=0.01,
                            residual_threshold=0.001,
                            max_trials=100)
    Rt = extractRt(model.params)
    
    return idx1[inliers], idx2[inliers], Rt

def extract(frame):
    orb = cv2.ORB_create()
    
    # Detection
    features = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8),
            maxCorners=1000, qualityLevel=0.01, minDistance=7)
    
    # Extraction
    kp = [cv2.KeyPoint(x = f[0][0], y = f[0][1], _size=20) for f in features]
    kp, des = orb.compute(frame, kp)
    
    # return points of interest
    return np.array([(k.pt[0], k.pt[1]) for k in kp]).reshape(-1, 2), des
    
class Frame(object):
    def __init__(self, mapp, image, K):
        self.pose = np.eye(4)
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.scale = 2

        self.h, self.w = image.shape[0:2]
    
        self.kpus, self.des = extract(image)
        self.kps = normalise(self.Kinv, self.kpus)
        self.pts = [None] * len(self.kps)
    
        self.id = len(mapp.frames)
        mapp.frames.append(self)
 
 

