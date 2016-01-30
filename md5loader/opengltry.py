# -*- coding: cp1252 -*-
import wx
import sys
import time
import numpy as np
from OpenGL.arrays import vbo
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLUT import *
from md5loader import MD5Model
global avatarc
avatarc=0
class CustomPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        box = wx.BoxSizer(wx.VERTICAL)
        global avatarc
        c = AvatarCanvas(self)
        avatarc=c
        c.SetMinSize((200, 200))
        box.Add(c, 1, wx.EXPAND|wx.ALIGN_CENTER|wx.ALL, 0)

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
    def load_obj(self, obj_file):  
        self.obj = OBJ(obj_file, swapyz=True)
        self.Refresh(True)
    def load_md5(self,filename):  
        self.md5 = MD5Model()
        self.md5.LoadModel(filename)
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


class AvatarCanvas(MyCanvasBase):
    
    obj=0
    start=0.0
    playing=False
    def onKeyPress(self, event):
        keycode = event.GetKeyCode()
        print keycode
        inc=1.0
        if keycode == 81:
            glTranslatef(0.0,0.0,-inc)
            return
        if keycode == 69:
            glTranslatef(0.0,0.0,inc)
            return
        if keycode == 87:
            glTranslatef(0.0,inc,0.0)
            return
        if keycode == 83:
            glTranslatef(0.0,-inc,0.0)
            return
        if keycode == 68:
            glTranslatef(inc,0.0,0.0)
            return
        if keycode == 65:
            glTranslatef(-inc,0.0,0.0)
            return
        if keycode == wx.WXK_SPACE:
            if self.playing:
                self.playing=False
            else:
                self.startTimer()
            return
    
    def OnMouseDown(self, evt):
        self.CaptureMouse()
        self.x, self.y = self.lastx, self.lasty = evt.GetPosition()

    def OnMouseUp(self, evt):
        self.ReleaseMouse()

    def OnMouseMotion(self, evt):
        if evt.Dragging() and evt.LeftIsDown():
            self.lastx, self.lasty = self.x, self.y
            self.x, self.y = evt.GetPosition()
            w, h = self.size
            w = max(w, 1.0)
            h = max(h, 1.0)
            xScale = 180.0 / w
            yScale = 180.0 / h
            glRotatef((self.y - self.lasty) * yScale, 1.0, 0.0, 0.0)
            glRotatef((self.x - self.lastx) * xScale, 0.0, 1.0, 0.0)
            
##            self.Refresh(False)
    def InitGL(self):
        self.Bind(wx.EVT_KEY_DOWN, self.onKeyPress)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        # Set viewing projection
        glMatrixMode(GL_PROJECTION)
        glFrustum(-0.5, 0.5, -0.5, 0.5, 1.0, 20.0)#3.0
        # Position viewer
        glMatrixMode(GL_MODELVIEW)
        glScalef(0.5,0.5,0.5)
        glTranslatef(0.0,0.0,-20.0)
##        self.adj_amb_light(1.0)
##        self.adj_light_pos(1.0)
##        self.adj_dif_light(1.0)
        # Position object
        glRotatef(self.y, 1.0, 0.0, 0.0)
        glRotatef(self.x, 0.0, 1.0, 0.0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        self.startTimer()
    def startTimer(self):
        self.start=time.time()
        self.playing=True
        self.getDeltaTime()
        self.Refresh()
    def getDeltaTime(self):
        end=time.time()
        dt=end-self.start
        self.start=end
        return dt
##    def updateTime(self):
##        end=time.time()
##        dt=end-self.start
##        self.advanceTime(dt)
##        if self.playing:
##            wx.CallLater(1000.0/self.animframe.fps,self.updateTime)
##        self.start=end
##    def advanceTime(self,dt):
##        if not self.playing:
##            return
##        self.globaltime+=dt
##        n=int(self.globaltime/(1/self.animframe.fps))
##        self.Refresh()
    def loadDefault(self):
        self.md5=MD5Model()
##        filename="C:\\Users\\Toshiba\\Documents\\dmcr\\StickAnimator\\md5loader\\bob_lamp_update\\bob_lamp_update_export.md5mesh"
        filename="C:\\Users\\Toshiba\\Documents\\dmcr\\StickAnimator\\md5loader\\bob_lamp_update\\bob_lamp_update.md5mesh"
##        filename="C:\\Users\\Toshiba\\Documents\\dmcr\\StickAnimator\\md5loader\\noman.md5mesh"
        self.md5.LoadModel(filename)
        
    def OnDraw(self):
##        global avatarc
##        print self.GetContext()
        # Clear color buffer and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.md5!=0:
            self.md5.Render(self.getDeltaTime())
        if self.size is None:
            self.size = self.GetClientSize()
        self.SwapBuffers()
        if self.playing:
            wx.CallLater(1000.0/24,self.Refresh)
    def adj_amb_light(self,v):
        glLightfv(GL_LIGHT0, GL_AMBIENT,  (v,v,v))
        self.Refresh(True)    
    def adj_light_pos(self,v):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, (3, 3,-v))
        self.Refresh(True)
    def adj_dif_light(self,v):
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  (v,v,v))
        self.Refresh(True)
def drawPyGame(indexPositions,vertexPositions):
##    glUseProgram(shader)
    indexPositions.bind()
    
    vertexPositions.bind()
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)
    #glDrawArrays(GL_TRIANGLES, 0, 3) #This line still works
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, None) #This line does work too!

    
##        def OnDraw(self):
##        # clear color and depth buffers
##        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
##        glLoadIdentity()
##        
##        if self.size is None:
##            self.size = self.GetClientSize()
##        w, h = self.size
##        w = max(w, 1.0)
##        h = max(h, 1.0)
##        xScale = 180.0 / w
##        yScale = 180.0 / h
##        
##        glTranslate(float(self.tx*xScale)/20.0, float(self.ty*yScale)/20.0, self.zpos)
##        glRotatef(float(self.rx*xScale), 0.0, 1.0, 0.0);
##        glRotatef(float(self.ry*yScale), 1.0, 0.0, 0.0);
##        
##        glCallList(self.obj.gl_list)
##        self.SwapBuffers()


class RunDemoApp(wx.App):
    def __init__(self):
        wx.App.__init__(self, redirect=False)
    def OnInit(self):
        frame = wx.Frame(None, -1, "RunDemo: ", pos=(0,0),style=wx.DEFAULT_FRAME_STYLE, name="run a sample")
        frame.Show(True)
        win = CustomPanel(frame)
        frame.SetSize((400,400))
##        global avatarc
##        print avatarc.GetContext()
##        avatarc.loadDefault()
        return True
    
app = RunDemoApp()
app.MainLoop()
