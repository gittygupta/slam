import numpy as np
import OpenGL.GL as gl
import pangolin
import g2o
from multiprocessing import Process, Queue
 
class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = None

    ## ** optimizer ** ##

    def optimize(self):

        # create g2o optimiser
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)

        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))   # mse

        # add frames to graph
        for f in self.frames:
            '''
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(f.id)
            se3 = g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3])
            v_se3.set_estimate(se3)
            v_se3.set_fixed(f.id == 0)
            opt.add_vertex(v_se3)'''

            sbacam = g2o.SBACam(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3]))
            sbacam.set_cam(f.K[0][0], f.K[1][1], f.K[2][0], f.K[2][1], 1.0)

            v_se3 = g2o.VertexCam()
            v_se3.set_id(f.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(f.id == 0)
            opt.add_vertex(v_se3)

        # add points to frame
        for p in self.points:
            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(p.id + 0x10000)
            pt.set_estimate(p.location[0:3])
            pt.set_marginalized(True)
            opt.add_vertex(pt)

            for f in p.frames:
                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_vertex(0, pt)
                edge.set_vertex(1, opt.vertex(f.id))
                kp = f.kps[f.pts.index(p)]
                edge.set_measurement(kp)
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)
        
        # init g2o optimizer
        opt.initialize_optimization()
        #opt.set_verbose(True)

        opt.optimize(20)

    ## ** viewer ** ##        

    def create_viewer(self):
        self.q = Queue()
        # p -> vp
        self.p = Process(target=self.viewer_thread, args=(self.q,))
        self.p.daemon = True
        self.p.start()
    
    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while True:
            self.viewer_loop(q)
    
    # refer HelloPangolin.py for this function
    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('SLAM', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        self.scam = pangolin.OpenGlRenderState(
                pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
                pangolin.ModelViewLookAt(0, -10, -5,   # position behind car
                                         0, 0, 0,       # look at (0, 0, 0)
                                         0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create view window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
        self.dcam.SetHandler(self.handler)
    
        self.darr = None
    
    def viewer_loop(self, q):      # viewer_refresh()
        if self.state is None or not q.empty():
            self.state = q.get()
    
        spts = np.array(self.state[1])
    
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)     # background
        self.dcam.Activate(self.scam)

        # draw poses of car
        gl.glColor3f(0.0, 1.0, 0.0) 
        pangolin.DrawCameras(self.state[0])

        # draw keypoints
        gl.glPointSize(2)
        gl.glColor3f(1.0, 1.0, 1.0)     # color
        pangolin.DrawPoints(self.state[1])
        
        pangolin.FinishFrame()
    
    
    def display_map(self):
        if self.q is None:
            return
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.location)
        #self.state = poses, pts
        #self.viewer_loop()
        self.q.put((np.array(poses), np.array(pts)))


class Point(object):
    # A point is a 3D point in the world
    # Each point appears on multiple frames
    def __init__(self, mapp, loc):
        self.frames = []
        self.location = loc     #pt
        self.idxs = []
    
        self.id = len(mapp.points)
        mapp.points.append(self)
    
    def add_observation(self, frame, idx):
        # frame -> the whole frame
        # idx -> index of which descriptor the point is in the frame
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)
    