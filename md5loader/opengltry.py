# -*- coding: cp1252 -*-
import wx
import sys
import time
import math
import numpy as np
from OpenGL.arrays import vbo
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from md5loader import MD5Model

class CustomPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        box = wx.BoxSizer(wx.VERTICAL)
        canvas = AvatarCanvas(self)
        canvas.SetMinSize((200, 200))
        box.Add(canvas, 1, wx.EXPAND|wx.ALIGN_CENTER|wx.ALL, 0)
        self.SetAutoLayout(True)
        self.SetSizer(box)
class MyCanvasBase(glcanvas.GLCanvas):
    md5=0
    def __init__(self, parent):
        glcanvas.GLCanvas.__init__(self, parent, -1)
        self.init = False
        self.context = glcanvas.GLContext(self)
       # initial mouse position
        self.lastx = self.x = 30
        self.lasty = self.y = 30
        self.size = None
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.OnOpen(0)
    def OnOpen(self, event):
        "Open an Obj file, set title if successful"
        # Create a file-open dialog in the current directory
        filters = 'md5 files (*.md5mesh)|*.md5mesh'
        dlg = wx.FileDialog(self, message="Open an Image...", defaultDir=os.getcwd()+"\objects", 
                            defaultFile="", wildcard=filters, style=wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()
            self.load_md5(filename)
        dlg.Destroy() # we don't need the dialog any more so we ask it to clean-up
    def load_md5(self,filename):  
        self.md5 = MD5Model()
        self.md5.LoadModel(filename)
        self.md5.setMaxSize(10)#(2.5)
        self.Refresh(True)
    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.
    def OnSize(self, event):
        wx.CallAfter(self.DoSetViewport)
        event.Skip()

    def DoSetViewport(self):
        size = self.size = self.GetClientSize()
        self.SetCurrent(self.context)
        glViewport(0, 0, size.width, size.height)
        
    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw()
def q_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return [x, y, z, w]
def toOldquad(quad):
    x,y,z,w=quad
    return [w,x,y,z]
def toNewquad(quad):
    w,x,y,z=quad
    return [x,y,z,w]
def dot(v1,v2):
    r=0
    for i in range(len(v1)):
        r+=v1[i]*v2[i]
    return r
def sumV(v1,v2):
    r=[]
    for i in range(len(v1)):
        r.append(v1[i]+v2[i])
    return r
def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return c
def mulS(v,s):
    return [x*s for x in v]
def restaV(v1,v2):
    r=[]
    for i in range(len(v1)):
        r.append(v1[i]-v2[i])
    return r
def vectorQ(v,q):
    u=[q[0], q[1], q[2]]
    s = q[3]
    return sumV(sumV(mulS(u,dot(u, v)*2.0),mulS(v,s*s-dot(u,u))),mulS(cross(u,v),2*s))
#asume un newQuad (dentro lo transforma)
##def toEuler(quad):
##    oldq=toOldquad(quad)
####    oldq=quad
##    q0,q1,q2,q3=oldq
##    roll=math.atan2(2*((q0*q1)+(q2*q3)),1-(2*((q1*q1)+(q2*q2))))
##    pitch=math.asin(2*((q0*q2)-(q3*q1)))
##    yaw=math.atan2(2*((q0*q3)+(q1*q2)),1-(2*((q2*q2)+(q3*q3))))
##    return [roll,pitch,yaw]
###asume un newquad (creo)
##def toQuad(v):
##    roll,pitch,yaw=v
##    qx=[math.cos(pitch/2),math.sin(pitch/2), 0, 0]
##    qy=[math.cos(yaw/2),0,math.sin(yaw/2),0]
##    qz=[math.cos(roll/2),0,0,math.sin(roll/2)]
##    qt=qx
##    qt=q_mult(qt,qy)
##    qt=q_mult(qt,qz)
##    return qt
##def addX(quad,d):
##        x,y,z,w=quad
##        d=min(d,0.9-x)
##        d=max(d,-(0.9-x))
##        print (x+d)*math.pi/180
##        k=math.sqrt(1.0-((2*x*d+(d*d))/((w*w)+(y*y)+(z*z))))
##        return [x+d,k*y,k*z,k*w]
##def addY(quad,d):
##        x,y,z,w=quad
##        d=min(d,0.9-y)
##        d=max(d,-(0.9-y))
##        print (y+d)*math.pi/180
##        k=math.sqrt(1.0-((2*y*d+(d*d))/((x*x)+(w*w)+(z*z))))
##        return [k*x,y+d,k*z,k*w]
##def addZ(quad,d):
##        x,y,z,w=quad
##        d=min(d,0.9-z)
##        d=max(d,-(0.9-z))
##        print (z+d)*math.pi/180
##        k=math.sqrt(1.0-((2*z*d+(d*d))/((x*x)+(y*y)+(w*w))))
##        return [k*x,k*y,z+d,k*w]
##def addW(quad,d):
##        x,y,z,w=quad
##        d=min(d,0.9-w)
##        d=max(d,-(0.9-w))
##        print (w+d)*math.pi/180
##        k=math.sqrt(1.0-((2*w*d+(d*d))/((x*x)+(y*y)+(z*z))))
##        return [k*x,k*y,k*z,w+d]
##


class Quaternion:
    """Quaternions for 3D rotations"""
    def __init__(self, x):
##        array=toOldquad(x)
        array=x
        self.x = np.asarray(array, dtype=float)
        
    @classmethod
    def from_v_theta(cls, v, theta):
        """
        Construct quaternion from unit vector v and rotation angle theta
        """
        theta = np.asarray(theta)
        v = np.asarray(v)
        
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)
        vnrm = np.sqrt(np.sum(v * v))

        q = np.concatenate([[c], s * v / vnrm])
        return cls(q)

    def __repr__(self):
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other):
        # multiplication of two quaternions.
        prod = self.x[:, None] * other.x

        return self.__class__([(prod[0, 0] - prod[1, 1]
                                 - prod[2, 2] - prod[3, 3]),
                                (prod[0, 1] + prod[1, 0]
                                 + prod[2, 3] - prod[3, 2]),
                                (prod[0, 2] - prod[1, 3]
                                 + prod[2, 0] + prod[3, 1]),
                                (prod[0, 3] + prod[1, 2]
                                 - prod[2, 1] + prod[3, 0])])
    def get(self):
        return self.x
##        return toNewquad(self.x)
    def as_v_theta(self):
        """Return the v, theta equivalent of the (normalized) quaternion"""
        # compute theta
        norm = np.sqrt((self.x ** 2).sum(0))
        theta = 2 * np.arccos(self.x[0] / norm)

        # compute the unit vector
        v = np.array(self.x[1:], order='F', copy=True)
        v /= np.sqrt(np.sum(v ** 2, 0))

        return v, theta

    def as_rotation_matrix(self):
        """Return the rotation matrix of the (normalized) quaternion"""
        v, theta = self.as_v_theta()
        c = np.cos(theta)
        s = np.sin(theta)

        return np.array([[v[0] * v[0] * (1. - c) + c,
                          v[0] * v[1] * (1. - c) - v[2] * s,
                          v[0] * v[2] * (1. - c) + v[1] * s],
                         [v[1] * v[0] * (1. - c) + v[2] * s,
                          v[1] * v[1] * (1. - c) + c,
                          v[1] * v[2] * (1. - c) - v[0] * s],
                         [v[2] * v[0] * (1. - c) - v[1] * s,
                          v[2] * v[1] * (1. - c) + v[0] * s,
                          v[2] * v[2] * (1. - c) + c]])
def q_conjugate(q):
    w,x,y,z = q
    return [w,-x,-y,-z]
def module(v):
    r=0
    for x in v:
        r+=x**2
    return math.sqrt(r)
def inverse(quad):
    m=module(quad)**2
    c=q_conjugate(quad)
    return [c[0],c[1],c[2],c[3]]
def ComputeQuatW(quat):
    t = 1.0-(quat[0]*quat[0])-(quat[1]*quat[1])-(quat[2]*quat[2])
    if t < 0.0:
        quat[3]=0.0
    else:
        quat[3]=-math.sqrt(t)
    return quat
class AvatarCanvas(MyCanvasBase):
    start=0.0
    swapYZ=False
    cam=[-0.13,0.1,-15]
    Z=15
    theta=-1.5
    h=0.1
    index=-1
    running=True
    playing=True
    displayskel=0
    def rotateX(self,i,angle,follow=True):
        self.rotateJoint(i,[1,0,0],angle,follow)
    def rotateY(self,i,angle,follow=True):
        self.rotateJoint(i,[0,1,0],angle,follow)
    def rotateZ(self,i,angle,follow=True):
        self.rotateJoint(i,[0,0,1],angle,follow)
    def rotateJoint(self,i,axis,angle,follow=True):
        animatedJoint=self.displayskel.m_Joints[i]
        assert(animatedJoint.m_ParentID>=0)
        oldrot=animatedJoint.m_Orient[:]
        invrot=inverse(oldrot[:])
        oldpos=animatedJoint.m_Pos[:]
        parentJoint = self.displayskel.m_Joints[animatedJoint.m_ParentID]
        parentpoint=parentJoint.m_Pos
        finalpoint=animatedJoint.m_Pos
        quad=Quaternion(animatedJoint.m_Orient)#5 era el spine
        rot = Quaternion.from_v_theta(axis,angle)
        rotmin=Quaternion.from_v_theta(axis,-angle)
        deltapoint=restaV(finalpoint,parentpoint)
        deltapoint=vectorQ(deltapoint,toNewquad(rot.get()))
        animatedJoint.m_Pos=sumV(parentpoint,deltapoint)
        animatedJoint.m_Orient=(rotmin*quad).get()
        if follow:
            self.followPosition(i,oldpos)
##            for j in range(len(self.displayskel.m_Joints)):
##                if self.displayskel.m_Joints[j].m_ParentID==i:
##                    self.rotateJoint(j,axis,angle)
##                    self.displayskel.m_Joints[j].m_Orient=ComputeQuatW(self.displayskel.m_Joints[j].m_Orient)
##            self.followOrient(i,oldrot)
    def followOrient(self,i,oldrot):
        for j in range(len(self.displayskel.m_Joints)):
            joint=self.displayskel.m_Joints[j]
            if joint.m_ParentID==i:
                newold=joint.m_Orient[:]
                deltarot=Quaternion(oldrot)*Quaternion(joint.m_Orient)
                joint.m_Orient=(deltarot*Quaternion(self.displayskel.m_Joints[i].m_Orient)).get()
                self.followOrient(j,newold)
    def followPosition(self,i,oldpos):
        for j in range(len(self.displayskel.m_Joints)):
            joint=self.displayskel.m_Joints[j]
            if joint.m_ParentID==i:
                newold=joint.m_Pos[:]
                olddelta=restaV(joint.m_Pos,oldpos)
                joint.m_Pos=sumV(self.displayskel.m_Joints[i].m_Pos,olddelta)
                self.followPosition(j,newold)
    def onKeyPress(self, event):
        keycode = event.GetKeyCode()
        inc=0.1
        print keycode
        if keycode == 81:#q
            self.Z+=inc
        if keycode == 69:#e
            self.Z-=inc
        if keycode == 87:#up
            self.h+=inc*2
        if keycode == 83:#down
            self.h-=inc*2
        if keycode == 68:
            self.theta-=inc/2#left
        if keycode == 65:
            self.theta+=inc/2#right
##credit    https://jakevdp.github.io/blog/2012/11/24/simple-3d-visualization-in-matplotlib/
        if keycode == 88:
            self.rotateX(7,math.pi/5)#9,7
            self.md5.m_Animation.setSkeleton(self.displayskel)
        if keycode == 89:
            self.rotateY(7,math.pi/5)
            self.md5.m_Animation.setSkeleton(self.displayskel)
        if keycode == 90:
            self.rotateZ(7,math.pi/5)
            self.md5.m_Animation.setSkeleton(self.displayskel)
        if keycode == wx.WXK_SPACE:
##            self.running=not self.running
            print "blank skeleton"
            self.displayskel=self.md5.m_Animation.getBlankSkeleton()
            print self.displayskel
            self.md5.m_Animation.setSkeleton(self.displayskel)
##            self.index+=1
##            self.index=self.index%len(self.md5.m_Animation.m_Skeletons)
##            print self.index
##            self.md5.m_Animation.setSkeleton(self.md5.m_Animation.m_Skeletons[self.index])
        self.refreshCam()
    def refreshCam(self):
        self.Z=max(self.Z,0)
        self.h=max(-(self.Z-0.0001),self.h)
        self.h=min(self.Z-0.0001,self.h)
        c=math.sqrt(max((self.Z**2)-(self.h**2),0))
        x=c*math.cos(self.theta)
        z=c*math.sin(self.theta)
##        if(self.swapYZ):
##            self.cam=[x,z,self.h]
##        else:
        self.cam=[x,self.h,z]
##        print str(self.Z)+","+str(self.h)+","+str(self.theta)
##        print self.cam
        self.Refresh(True)
    def OnMouseDown(self, evt):
##        self.CaptureMouse()
        self.x, self.y = self.lastx, self.lasty = evt.GetPosition()
    def OnMouseUp(self, evt):
        pass
##        self.ReleaseMouse()
    def onMouseWheel(self,evt):
        inc=1.0
        if evt.GetWheelRotation()>0:
            self.Z-=inc
        else:
            self.Z+=inc
        self.refreshCam()
    def OnMouseMotion(self, evt):
        if evt.Dragging() and evt.LeftIsDown():
            self.lastx, self.lasty = self.x, self.y
            self.x, self.y = evt.GetPosition()
            xScale = 0.01
            yScale = 0.1
            self.theta-=(self.x - self.lastx) * xScale
            self.h+=(self.y - self.lasty)* yScale
            self.refreshCam()
##    def IdleGL(self):
##        elapsedTime=1.0/30
##        fDeltaTime = self.getDeltaTime()
##        if self.md5!=0 and self.playing:
##            self.md5.Update(fDeltaTime)
##        self.Refresh(True) 
    def InitGL(self):
        self.swapYZ=True
        self.Bind(wx.EVT_KEY_DOWN, self.onKeyPress)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        self.Bind(wx.EVT_MOUSEWHEEL,self.onMouseWheel)
        glMatrixMode(GL_PROJECTION)
        glFrustum(-0.5, 0.5, -0.5, 0.5, 1.0, 20.0)
        glMatrixMode(GL_MODELVIEW)
        self.adj_amb_light(10.0)#(1.0)
##        self.adj_light_pos(20)
##        self.adj_dif_light(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
##        self.startTimer()
##    def startTimer(self):
##        self.start=time.time()
##        self.running=True
##        self.getDeltaTime()
##        self.Refresh()
##    def getDeltaTime(self):
##        end=time.time()
##        dt=end-self.start
##        self.start=end
##        if self.running:
##            wx.CallLater(1.000/24,self.IdleGL)
##        return dt
    def OnDraw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1,1,1,1)
        glLoadIdentity()
        if self.swapYZ:
            gluLookAt(self.cam[0],self.cam[2],self.cam[1],0,0,0,0,0,1.0)
        else:
            gluLookAt(self.cam[0],self.cam[1],self.cam[2],0,0,0,0,1.0,0)
        if self.md5!=0:
            self.md5.Render()
        if self.size is None:
            self.size = self.GetClientSize()
        self.SwapBuffers()
        
    def adj_amb_light(self,v):
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [v,v,v])
        self.Refresh(True)    
    def adj_light_pos(self,v):
##        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
##        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, (3, 3,-v))
        self.Refresh(True)
    def adj_dif_light(self,v):
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [v,v,v])
        self.Refresh(True)
def module(v):
    r=0
    for x in v:
        r+=x**2
    return math.sqrt(r)
def normalize(v):
    r=[]
    m=module(v)
    for x in v:
        r.append(x/m)
    return r
class RunDemoApp(wx.App):
    def __init__(self):
        wx.App.__init__(self, redirect=False)
    def OnInit(self):
        frame = wx.Frame(None, -1, "RunDemo: ", pos=(0,0),style=wx.DEFAULT_FRAME_STYLE, name="run a sample")
        frame.Show(True)
        win = CustomPanel(frame)
        frame.SetSize((400,400))
        return True
    
app = RunDemoApp()
app.MainLoop()
