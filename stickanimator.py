import sys,math
from numpy import fft,arange, sin, pi
from scipy import signal as sign
import numpy as np 
import time
from Tkinter import *
import ntpath
import tkMessageBox
import tkFont
from PIL import Image, ImageDraw, ImageTk
import scipy.interpolate as si
import signal
from tkFileDialog import askopenfilename, asksaveasfilename
import cv2

class CsvNormalizer(object):
    corners=[]#[i,j] where we change previous node of i so it becomes j
    #first element doesnt have previous so its fine whatever value of angle it has
    corners.append([4,2])
    corners.append([8,2])
    corners.append([12,0])
    corners.append([16,0])
    longitudes=[0,0.07,0.34,0.17,0.2,0.17,0.2,0.1,0.2,0.17,0.2,0.1,0.1,0.35,0.36,0.08,0.1,0.35,0.36,0.08]
    longitud=140##esta es la longitud "normal" de un "bone"
    ## un "bone" es la distancia entre un nodo y otro
    for i in range(len(longitudes)):
        longitudes[i]=longitudes[i]*longitud
    def getLengths(self,frame):
        lens=[]
        for i in range(1,len(frame)):
            ind=self.cornersIndex(i)
            if(ind==-1):
                ind=i-1
            v=0
            v+=(frame[i][0]-frame[ind][0])**2
            v+=(frame[i][1]-frame[ind][1])**2
            v+=(frame[i][2]-frame[ind][2])**2
            v=math.sqrt(v)
            v=float("{0:.2f}".format(v))
            lens.append(v)
        return lens
        
    def cornersIndex(self,i):
        for c in self.corners:
            if i == c[0]:
                return c[1]
        return -1
    def __init__(self):
        pass
    def normalizeAll(self,allframes):
        for i in range(len(allframes)):
            allframes[i]=self.normalize(allframes[i])
    def normalize(self,frame):
        angles=self.getAngles(frame)
        newframe=[]
        for i in range(len(frame)):
            if(i==0):
                newframe.append([0,0,0])
            else:
                p=self.longitudes[i]
                ind=self.cornersIndex(i)
                if(ind==-1):
                    ind=i-1
                x=newframe[ind][0]+(p*math.sin(angles[i][0])*math.cos(angles[i][1]))
                y=newframe[ind][1]+(p*math.sin(angles[i][0])*math.sin(angles[i][1]))
                z=newframe[ind][2]+(p*math.cos(angles[i][0]))
                newframe.append([x,y,z])
        return newframe
    def getAngles(self,frame):
        angles=[]
        for i in range(len(frame)):
            if(i==0):
                angles.append([0,0])
            else:
                ind=self.cornersIndex(i)
                if(ind==-1):
                    ind=i-1
                x=frame[i][0]-frame[ind][0]
                y=frame[i][1]-frame[ind][1]
                z=frame[i][2]-frame[ind][2]
                theta=math.acos(z/math.sqrt((x*x)+(y*y)+(z*z)))
                if(theta>math.pi):
                    theta-=math.pi
                else:
                    if(theta<0):
                        theta+=math.pi
                phi=math.atan2(y,x)
                if(phi>2*math.pi):
                    phi-=2*math.pi
                else:
                    if(phi<0):
                        phi+=2*math.pi
                angles.append([theta,phi])
        return angles
def new_mainloop():
    global tasks
    while(True):
        master.update()
        if(len(tasks)!=0):
            for t in tasks:
                t()
            tasks=[]
    master.destroy()
def drawLines(pt,color):
    global primera
    if primera==0:
        primera=pt
    else:
        w.create_line(primera[0],primera[1],pt[0],pt[1],fill =color ,width=3)
        primera=pt
        
def drawLinesSlide(pt,color):
    global primera
    if primera==0:
        primera=pt
    else:
        wslide.create_line(primera[0],primera[1],pt[0],pt[1],fill =color ,width=3)
        primera=pt

def drawOvalsPlayer(pt,color):
    w.create_oval(pt[0]-5,pt[1]-5,pt[0]+5,pt[1]+5,fill=color)
def drawOvals(pt,color):
    wslide.create_oval(pt[0]-5,pt[1]-5,pt[0]+5,pt[1]+5,fill=color)
def drawEmptyOval(pt,rad,color):
    w.create_oval(pt[0]-rad,pt[1]-rad,pt[0]+rad,pt[1]+rad,outline=color,width=3)
  
def bspline2D(d,K,N):
    x=bsplineAnt(d[:,0],K,N)
    y=bsplineAnt(d[:,1],K,N)
    nuevo=[]
    for j in range(0,len(x)):
        nuevo.append([x[j],y[j]])
    return nuevo
def bspline3D(d,K,N):
    x=bsplineAnt(d[:,0],K,N)
    y=bsplineAnt(d[:,1],K,N)
    z=bsplineAnt(d[:,2],K,N)
    nuevo=[]
    for j in range(0,len(x)):
        nuevo.append([x[j],y[j],z[j]])
    return nuevo
def bsplineAnt(d, K=3, N=100):
    K=min(len(d)-1,K)
    t = range(len(d))
    ipl_t = np.linspace(0.0, len(d) - 1, N)
    x_tup = si.splrep(t, d, k=K)
    x_list = list(x_tup)
    xl = d.tolist()
    x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]
    x_i = si.splev(ipl_t, x_list)
    return x_i
#http://devosaurus.blogspot.com/2013/10/exploring-b-splines-in-python.html  
def make_knot_vector(n, m):
    total_knots = m+n+2                           
    outer_knots = n+1                            
    inner_knots = total_knots - 2*(outer_knots)
    knots  = [0]*(outer_knots)
    knots += [i for i in range(1, inner_knots)]
    knots += [inner_knots]*(outer_knots)
    return tuple(knots) 
def C_factory(P, V=None, n=2):
    m = len(P)    
    D = len(P[0]) 
    b_n = basis_factory(n)
    def S(t, d):
        out=0.
        for i in range(m):
            out += P[i][d]*b_n(t, i, V)
        return out
    def C(t):
        out = [0.]*D           
        for d in range(D):     
            out[d] = S(t,d)
        return out   
    C.V = V                   
    C.spline = S              
    C.basis = b_n             
    C.min = V[0]              
    C.max = V[-1]-0.0001             
    C.endpoint = True#C.max!=V[-1]
    return C
def basis_factory(degree):
    if degree == 0:
        def basis_function(t, i, knots):
            t_this = knots[i]
            t_next = knots[i+1]
            out = 1. if (t>=t_this and t< t_next) else 0.         
            return out
    else:
        def basis_function(t, i, knots):
            out = 0.
            t_this = knots[i]
            t_next = knots[i+1]
            t_precog  = knots[i+degree]
            t_horizon = knots[i+degree+1]            
            top = (t-t_this)
            bottom = (t_precog-t_this)
            if bottom != 0:
                out  = top/bottom * basis_factory(degree-1)(t, i, knots)
                
            top = (t_horizon-t)
            bottom = (t_horizon-t_next)
            if bottom != 0:
                out += top/bottom * basis_factory(degree-1)(t, i+1, knots)
            return out       
    basis_function.lower = None if degree==0 else basis_factory(degree-1)
    basis_function.degree = degree
    return basis_function
def BezierSpline(d,K,N):
    if(len(d)<=1):
        return d
    n = K
    V = make_knot_vector(n, len(d))
    C = C_factory(d, V, n)
    sampling = [t for t in np.linspace(C.min, C.max, N,endpoint=C.endpoint)]
    curvepts = [ C(s) for s in sampling ]
    return curvepts
def play():
    if len(animframe.animations)==0:
        return
    if(animframe.getAnim().isAnimationFinished()):
        progress["value"] = 0
        animframe.getAnim().currenti=0
    global playing
    playing=True
    play_pause.config(image=pausegif,width="20",height="20",command = pause)
def pause():
    if len(animframe.animations)==0:
        return
    global playing
    playing=False
    play_pause.config(image=playgif,width="20",height="20",command = play)
def setFrame(event):
    if len(animframe.animations)==0:
        return
    framei=int(len(animframe.getAnim().frames)*(float(event.x)/progress.winfo_width()))
    framei=max(min(framei,len(animframe.getAnim().frames)-1),0)
    animframe.getAnim().currenti=framei
    progress["value"]=animframe.getAnim().currenti
global master
master = Tk()
master.columnconfigure(0, weight=1)
master.columnconfigure(1, weight=1)
master.columnconfigure(2, weight=1)
master.rowconfigure(0, weight=1)
master.rowconfigure(1, weight=1)

class CustomCanvas(Canvas):
    widthback=0
    heightback=0
    back=0
    pin=0
    topaint=0
    def __init__(self,master, *args, **kwargs):
        Canvas.__init__(self,master,bd=2,background="gray",*args, **kwargs)
        self.back = PhotoImage(file="cuadros.gif")
        path = "pin.png"
        image = Image.open(path)
        image = image.resize((40,40), Image.ANTIALIAS)
        self.pin = ImageTk.PhotoImage(image)
        self.bind("<Configure>",self.repaint)
        self.topaint=[]
        self.widthback=self.back.width()
        self.heightback=self.back.height()
        self.repaint(0)
    def repaint(self,evt):
        self.delete("all")
        if self["scrollregion"]:
            col=int(math.ceil(float(float(self["scrollregion"].split(" ")[2]))/self.widthback)) 
        else:
            col=int(math.ceil(float(self.winfo_width())/self.widthback))
        row=int(math.ceil(float(self.winfo_height())/self.heightback))
        for i in range(col):
            for j in range(row):
                self.create_image((i+0.5)*self.widthback,(j+0.5)*self.heightback, image=self.back)
        self.create_image(self.winfo_width()/2,20, image=self.pin)
        for f in self.topaint:
            f()
class CustomFrame(Frame):
    expandbutton=0
    toolbar=0
    name=0
    customFont=0
    expgif=0
    shrgif=0
    prop=0
    def __init__(self,master,name):
        Frame.__init__(self,master,bd=3,background='white')
        self.name=name
        self.expgif=PhotoImage(file="expand.gif")
        self.shrgif=PhotoImage(file="shrink.gif")
        self.customFont = tkFont.Font(family='Buxton Sketch', size=14)
        self.toolbar=Frame(self,background="white",bd=2,cursor="arrow")
        self.toolbar.pack(side=TOP, fill="x")
        self.expandbutton = Button(self.toolbar,command=self.expand,image=self.expgif,width="20",height="20")
        self.expandbutton.pack(side="right")
        title=Label(self.toolbar,text=self.name,font=self.customFont,background="white",fg="black")
        title.pack(side="left")
    def saveProp(self):
        self.prop=[int(self.grid_info()["column"]),int(self.grid_info()["row"]),float(self.canvas.winfo_width()),float(self.canvas.winfo_height()),int(self.grid_info()["columnspan"]),int(self.grid_info()["rowspan"])]
    def loadProp(self):
        self.grid(column=self.prop[0],row=self.prop[1],columnspan=self.prop[4],rowspan=self.prop[5])
        self.canvas.config(width=float(math.floor(self.prop[2])),height=float(math.floor(self.prop[3])))
    def expand(self):
        global customframes
        for frame in customframes:
            frame.saveProp()
            frame.grid_forget()
            
        self.grid(column=0,row=0)
        global master
        self.canvas.config(width=master.winfo_width(),height=master.winfo_height())
        self.expandbutton.config(image=self.shrgif,command=self.shrink)
    def shrink(self):
        global customframes
        for frame in customframes:
            frame.grid_forget()
            frame.loadProp()
        self.expandbutton.config(image=self.expgif,command=self.expand)
    
        
class AnimationFrame(CustomFrame):
    canvas=0
    h_off=40
    pointer=0
    timepos=0
    hbar=0
    botframe=0
    maxText=0
    fps=30.0
    stack=0
    interpoled=0
    render=0
    rad=25
    poseS=-1
    poseE=-1
    clipboard=[]
    redolist=[]
    undolist=[]
    maxdolist=10
    def __init__(self,master):
        CustomFrame.__init__(self,master,"área de animación")
        self.grid(column=0,row=0,columnspan=3,sticky=W+E)
        self.canvas=CustomCanvas(self)
        self.canvas.pack(fill=X)
        self.spc=5.0/1000#segundos por cada pixel de canvas
        self.maxTime=10.0#en segundos

        self.stack=[]
        self.interpoled=[]
        self.render=[]
        
        self.canvas.topaint.append(self.redraw)
        self.canvas.bind("<Button-1>",self.clickedTime)
        self.canvas.bind("<Motion>",self.moveTime)
        self.canvas.bind("<ButtonRelease-1>",self.releaseTime)
        self.canvas.bind("<Double-Button-1>",self.doubleTime)
        self.canvas.bind('<Control-c>', self.copiar)
        self.canvas.bind('<Control-x>', self.cortar)
        self.canvas.bind('<Control-v>', self.pegar)
        self.canvas.bind('<Control-z>', self.deshacer)
        self.canvas.bind('<Control-r>', self.rehacer)
        self.canvas.bind("<Delete>",self.deletePose)
        self.botframe=Frame(self)
        self.botframe.pack(side=BOTTOM,fill=X)
        self.hbar=Scrollbar(self.botframe,orient=HORIZONTAL)
        self.hbar.pack(side=LEFT,expand=1,fill=X)
        self.hbar.config(command=self.canvas.xview)
        self.canvas.config(xscrollcommand=self.hbar.set)
        self.maxText = Text(self.botframe,height=1,width=6,font=self.customFont)
        self.maxText.bind("<Return>", self.changeMax)
        self.maxText.bind("<Button-1>", self.enableMax)
        self.canvas.config(scrollregion=(0,0,self.maxTime/self.spc,0))
        self.maxText.insert(INSERT,str(self.maxTime)+"s")
        
        self.maxText.pack(side=RIGHT)
        self.timepos=0
        self.pointer=0
    def deshacer(self,evt=None):
        if(len(self.undolist)!=0):
            self.redolist.append(self.stack[:])
            if(len(self.redolist)>self.maxdolist):
                self.redolist=self.redolist[1:]
            self.stack=self.undolist.pop()
        self.interpolateStack()
    def rehacer(self,evt=None):
        if(len(self.redolist)!=0):
            self.undolist.append(self.stack[:])
            if(len(self.undolist)>self.maxdolist):
                self.undolist=self.undolist[1:]
            self.stack=self.redolist.pop()
        self.interpolateStack()
    def do(self):
        self.redolist=[]
        self.undolist.append(self.stack[:])
        if(len(self.undolist)>self.maxdolist):
            self.undolist=self.undolist[1:]
        
    def copiar(self, evt=None):
        self.clipboard=self.interpoled[self.pointer][:]
    def cortar(self, evt):
        self.clipboard=self.interpoled[self.pointer][:]
        self.stack[self.pointer]=[]
        self.interpolateStack()
        self.do()
    def pegar(self, evt):
        self.stack[self.pointer]=self.clipboard[:]
        self.interpolateStack()
        self.do()
    def unirPose(self,pose):
        newpose=[]
        for s in pose:
            for p in s:
                newpose.append(p)
        return newpose
    def separarPose(self,pose):
        newpose=[]
        newpose.append([pose[0]])
        newpose.append([pose[1]])
        stroke=[]
        for i in range(2,len(pose)):
            stroke.append(pose[i])
            if(len(stroke)==5):
                newpose.append(stroke)
                stroke=[]
        return newpose
    def vectorizePose(self,pose):
        newpose=[]
        for p in pose:
            for v in p:
                newpose.append(v)
        return newpose
    def unvectorizePose(self,pose):
        newpose=[]
        point=[]
        for v in pose:
            point.append(v)
            if(len(point)==2):
                newpose.append(point)
                point=[]
        return newpose
    def fitStack(self):
        n=0
        for i in range(1,len(self.stack)):
           if(len(self.stack[-i])==0):
               n+=1
           else:
               break
        if(n==0):
            return
        self.stack=self.stack[:-n]
    def getInterpolation(self,init=0):
        if(init==-1 or init==len(self.stack)-1):
            return []
        last=-1
        for i in range(init+1,len(self.stack)):
            if(len(self.stack[i])!=0):
                last=i
                break
        n=last-init+1
        v1=self.vectorizePose(self.unirPose(self.stack[init]))
        v2=self.vectorizePose(self.unirPose(self.stack[last]))
        inter=[np.linspace(i,j,n) for i,j in zip(v1,v2)]
        inter=np.array(inter).T
        if init!=0:
            inter=inter[1:]
        return list(inter)+self.getInterpolation(last)
    def interpolateStack(self):
        interp=self.getInterpolation()
        self.interpoled=[]
        for v in interp:
            self.interpoled.append(self.separarPose(self.unvectorizePose(v)))
        self.canvas.repaint(0)
    def curveStack(stack):
        pass
    def setMax(self,maxt):
        self.maxTime=maxt
        self.canvas.config(scrollregion=(0,0,self.maxTime/self.spc,0))
        self.maxText.delete(1.0,END)
        self.maxText.insert(INSERT,str(self.maxTime)+"s")
        self.maxText.config(state = DISABLED)
        self.canvas.repaint(0)
        
    def changeMax(self,evt):
        s=self.maxText.get("%d.%d"%(1,0),END)
        s=''.join(c for c in s if c.isdigit() or c=='.')
        self.setMax(float(s))
        return 'break'
    def enableMax(self,evt):
        self.maxText.config(state = NORMAL)
    def toSeconds(self,x):
        i=self.hbar.get()[0]
        recorrido=i*self.maxTime/self.spc
        return (recorrido+x)*self.spc
    def recalculateCanvas(self,x):
        i=self.hbar.get()[0]
        recorrido=i*self.maxTime/self.spc
        return recorrido+x
    def drawStack(self):
        for i in range(len(self.stack)):
            if i==self.pointer:
                self.dibujarPose(i,self.stack[i],"blue")
            else:
                self.dibujarPose(i,self.stack[i])
    def drawInterpoled(self):
        i=self.pointer
        if i>=0 and i<len(self.interpoled):
            self.dibujarPose(i,self.interpoled[i],"gray70")
    def redraw(self):
        self.drawTimeline()
        self.drawAllTimeMarks()
        self.drawInterpoled()
        self.drawStack()
        if self.poseS!=-1:
            self.dibujarPose(self.pointer,self.stack[self.poseS],"red")
        self.drawPointer()
        self.drawTimePos()
    def savePose(self,num):
        archivo = open("pose"+str(num)+".txt", 'w')
        for p in self.frames[num]:
            archivo.write(str(p[0])+"/"+str(p[1])+"/"+str(p[2])+"@")
        archivo.close()
    def toString(self):
        s=""
        s+=str(self.maxTime)
        s+="@"
        for pose in self.stack:
            s+=self.poseToString(pose)
            s+="#"
        s=s[:-1]
        return s
    def loadAnimation(self,string):
        loaded=string.split("@")
        maxt=float(loaded[0])
        posesstring=loaded[1]
        self.stack=[]
        for s in posesstring.split("#"):
            self.stack.append(self.stringToPose(s))
        self.setMax(maxt)
        self.interpolateStack()
    def poseToString(self,pose):
        string=""
        for s in pose:
            for p in s:
                for v in p:
                    string+=str(v)
                    string+=":"
                string=string[:-1]
                string+=";"
            string=string[:-1]    
            string+=","
        string=string[:-1]
        return string
    def stringToPose(self,string):
        pose=[]
        if len(string)==0:
            return []
        for stroke in string.split(","):
            s=[]
            for point in stroke.split(";"):
                p=[]
                for value in point.split(":"):
                    p.append(float(value))
                s.append(p)
            pose.append(s)
        return pose
    def drawTimePos(self):
        x=self.getStackCanvas(self.timepos)
        self.canvas.create_line(x,0,x,self.canvas.winfo_height(),fill="black",width=2)
        self.canvas.create_text(x,self.canvas.winfo_height()-20,anchor="nw", text="{0:.2f}".format(self.getStackSeconds(self.timepos))+"s")            
    def drawPointer(self):
        x=self.getStackCanvas(self.pointer)
        self.canvas.create_line(x,0,x,self.canvas.winfo_height(),fill="gray",width=3)
        self.canvas.create_text(x,self.canvas.winfo_height()-20,anchor="nw",fill="gray40", text="{0:.2f}".format(self.getStackSeconds(self.pointer))+"s")            
    def drawAllTimeMarks(self):
        for i in range(int(self.maxTime)+1):
            self.drawTimeMark(float(i))
    def drawTimeMark(self,t):
        x=t/self.spc
        he=self.canvas.winfo_height()
        self.canvas.create_line(x,he-self.h_off,x,he-self.h_off/2,fill="orange",width=3)
        self.canvas.create_text(x,he-self.h_off/2,anchor="nw", text="{0:.2f}".format(t)+"s")        
    def drawTimeline(self):
        wid=self.maxTime/self.spc
        he=self.canvas.winfo_height()
        self.canvas.create_line(0,he-self.h_off,wid,he-self.h_off,fill = "yellow",width=6)
        self.canvas.create_line(wid-20,he-self.h_off,wid,he-self.h_off,fill = "red",width=6)
    def getStackPos(self,x):
        return int(x/((1/self.fps)/self.spc))
    def getStackCanvas(self,i):
        return i*((1/self.fps)/self.spc)
    def getStackSeconds(self,i):
        return float(i)/self.fps
    def clickedTime(self,evt):
        self.poseS=-1
        self.canvas.focus_set()
        if(len(self.stack)==0):
            return
        i=self.getStackPos(self.recalculateCanvas(evt.x))
        self.timepos=i
        if(i<len(self.stack) and i!=0):
            self.poseS=i
        self.canvas.repaint(0)
    def doubleTime(self,evt):
        i=self.getStackPos(self.recalculateCanvas(evt.x))
        if i!=0 and i<len(self.stack) and len(self.stack[i])!=0:
            global draw
            draw.initEditPose(i)
            self.poseE=i
    def moveTime(self,evt):
        i=self.getStackPos(self.recalculateCanvas(evt.x))
        self.pointer=i
        self.canvas.repaint(0)
    def releaseTime(self,evt):
        if self.poseS==-1:
            return
        if self.poseS==self.poseE:
            self.poseE=self.pointer
        if self.pointer>=len(self.stack):
            n=self.pointer-len(self.stack)
            self.stack=self.stack+([[]]*n)
            self.stack.append(self.stack[self.poseS])
            self.stack[self.poseS]=[]
        else:
            temp=self.stack[self.poseS][:]
            self.stack[self.poseS]=self.stack[self.pointer]
            self.stack[self.pointer]=temp
            self.fitStack()
        
        self.poseS=-1
        self.do()
        self.interpolateStack()
    def deletePose(self,evt):
        if(self.timepos!=0 and self.timepos<len(self.stack) and len(self.stack[self.timepos])!=0):
            if(tkMessageBox.askyesno("Eliminar pose","Seguro que quieres eliminar esta pose?")):
                self.stack[self.timepos]=[]
        self.do()
        self.interpolateStack()
    def agregarPose(self,pose):
        i=self.timepos
        act=len(self.stack)
        if i>=act:
            n=i-act
            self.stack=self.stack+([[]]*n)
            self.stack.append(self.scalePose(pose))
        else:
            self.stack[i]=self.scalePose(pose)
        self.do()
        self.interpolateStack()
    def refreshPose(self,pose):
        self.stack[self.poseE]=self.scalePose(pose)
        self.interpolateStack()
        self.canvas.repaint(0)
    def endEditPose(self):
        self.poseE=-1
    def scalePose(self,pose):
        escalado=[]
        escalado.append(pose[0])
        for i in range(1,len(pose)):
            escalado.append([])
            for p in pose[i]:
                escalado[-1].append([p[0]*self.rad,p[1]*self.rad])
        return escalado
    def dibujarPose(self,index,pose,color="black"):
        if len(pose)==0:
            return
        todraw=[]
        ojos=pose[0]
        for i in range(1,len(pose)):
            todraw+=pose[i]
        c=Centroid(todraw)
##        [minX, minY, maxX - minX, maxY - minY]
        p=[self.getStackCanvas(index),(self.canvas.winfo_height())/2]
        todraw=TranslateTo(todraw,[p[0],p[1]])
        head=getCircle(self.rad,250)
        head=TranslateTo(head,todraw[0])
        self.drawOjos(todraw[0],ojos[0],color)
        separados=[]
        stroke=[]
        for i in range(1,len(todraw)):
            stroke.append(todraw[i])
            if(len(stroke)==5):
                separados.append(stroke)
                stroke=[]
        self.drawAllNormal(separados,color)
        self.drawPoints(head,color)
    def drawAllNormal(self,All,color):
        for a in All:
            self.drawPoints(toSpline(a),color)
    def drawPoints(self,todraw,color):
        global primera 
        primera=[]
        for a in todraw:
            drawPluma(self.canvas,a,color)
    
    def drawOjos(self,head,ojos,color):
        diametro=self.rad*2
        radOjos=diametro/10
        fullsep=diametro/3
        c=head
        sep=fullsep*(1-abs(ojos[0]))
        x=c[0]+ojos[0]*self.rad
        y=c[1]+ojos[1]*self.rad
        r=radOjos
        self.canvas.create_oval(x+sep/2-r, y-r, x+sep/2+r, y+r,outline=color, width=3)
        self.canvas.create_oval(x-sep/2-r, y-r, x-sep/2+r, y+r,outline=color, width=3)   
    
        
class controlPoint(object):
    rad=10
    locations=[]
    draw=0
    color="red"
    def __init__(self,draw,locs=[]):
        self.locations=locs
        self.draw=draw
    def isSelected(self,x,y):
        return Distance([x,y],self.draw.strokes[self.locations[0][0]][self.locations[0][1]])<=self.rad
    def moveTo(self,x,y):
        for l in self.locations:
            i=l[0]
            j=l[1]
            offset=[[x-self.draw.strokes[i][j][0]],[y-self.draw.strokes[i][j][1]]]
            self.draw.strokes[i][j][0]=x
            self.draw.strokes[i][j][1]=y
##           self.draw.limitPoint(i,j)
            self.moveBackwards(i,j)
            self.dragForward(i,j,offset)
        self.draw.limitAllNF()  
            
##            self.moveChain(i,j)
    def moveChain(self,i,j):
        ind=[0,2,4]
        if not(j in ind):
            ind.append(j)
            ind.sort()
        newstroke=[]
        for index in ind:
            newstroke.append(self.draw.strokes[i][index])
        self.draw.strokes[i]=toStick(newstroke)
    def dragForward(self,i,j,offset):
        for m in range(j+1,len(self.draw.strokes[i])):
            self.draw.strokes[i][m][0]+=offset[0]
            self.draw.strokes[i][m][1]+=offset[1]
    def moveForward(self,i,j):
        if(j+2>=len(self.draw.strokes[i])):
            return
        p0=self.draw.strokes[i][j+2]
        pf=self.draw.strokes[i][j]
        pm=[(p0[0]+pf[0])/2,(p0[1]+pf[1])/2]
        self.draw.strokes[i][j+1][0]=pm[0]
        self.draw.strokes[i][j+1][1]=pm[1]
        

    def moveBackwards(self,i,j):
        if(j-2<0):
            return
        p0=self.draw.strokes[i][j-2]
        pf=self.draw.strokes[i][j]
        pm=[(p0[0]+pf[0])/2,(p0[1]+pf[1])/2]
        self.draw.strokes[i][j-1][0]=pm[0]
        self.draw.strokes[i][j-1][1]=pm[1]
    def drawCP(self):
        p=self.draw.strokes[self.locations[0][0]][self.locations[0][1]]
        self.draw.canvas.create_oval(p[0]-self.rad/2,p[1]-self.rad/2,p[0]+self.rad/2,p[1]+self.rad/2,fill=self.color)
class controlHead(controlPoint):
    def getPosition(self):
        return self.draw.head[int(len(self.draw.head)*0.75)]
    def isSelected(self,x,y):
        return Distance([x,y],self.getPosition())<=self.rad
    def moveTo(self,x,y):
        p=self.getPosition()
        c=Centroid(self.draw.head)
        self.draw.head=TranslateTo(self.draw.head,[x-(p[0]-c[0]),y-(p[1]-c[1])])
    def drawCP(self):
        p=self.getPosition()
        self.draw.canvas.create_oval(p[0]-self.rad/2,p[1]-self.rad/2,p[0]+self.rad/2,p[1]+self.rad/2,fill=self.color)
class controlEyes(controlPoint):
    def getPosition(self):
        c=Centroid(self.draw.head)
        diametro=Distance(self.draw.head[0],self.draw.head[int(len(self.draw.head)/2)])
        r=diametro/2
        return [c[0]+self.locations[0]*r,c[1]+self.locations[1]*r]
    def isSelected(self,x,y):
        return Distance([x,y],self.getPosition())<=self.rad
    def moveTo(self,x,y):
        diametro=Distance(self.draw.head[0],self.draw.head[int(len(self.draw.head)/2)])
        rad=diametro/2
        if(Distance([x,y],Centroid(self.draw.head))>rad):
            return
        c=Centroid(self.draw.head)
        self.locations[0]=(x-c[0])/rad
        self.locations[1]=(y-c[1])/rad
    def drawCP(self):
        p=self.getPosition()
        self.draw.canvas.create_oval(p[0]-self.rad/2,p[1]-self.rad/2,p[0]+self.rad/2,p[1]+self.rad/2,fill=self.color)  
class DrawFrame(CustomFrame):
    canvas=0
    strokes=0
    currentpoints=0
    bottomButton=0
    body=0
    head=0
    lastsize=[]
    finished=False
    controlPoints=0
    selectedCP=0
    proportions=0
    lbar=0
    tbar=0
    ant=[]

    newb=0
    insertb=0
    doneb=0
    newgif=0
    insertgif=0
    donegif=0
    def __init__(self,master):
        CustomFrame.__init__(self,master,"área de dibujo")
        self.grid(column=0,row=1,sticky=W+E+N+S)
        self.canvas=CustomCanvas(self)
        self.canvas.pack(expand=1,fill=BOTH)
        self.currentpoints=[]
        self.strokes=[]
        self.head=getCircle(9*0.01,250)
        self.proportions=[0.58,0.67,0.67,0.89,0.89]
        self.head=TranslateTo(self.head,[self.canvas.winfo_width()*0.5,self.canvas.winfo_height()*0.20])
        self.lastsize=[float(self.canvas.winfo_width()),float(self.canvas.winfo_height())]
        self.canvas.topaint.append(self.redraw)
        self.canvas.bind( "<B1-Motion>", paintDraw )
        self.canvas.bind( "<Button-1>", clickedDraw )
        self.canvas.bind( "<ButtonRelease-1>", releaseDraw )
        self.canvas.bind( "<Button-3>", clicked3Draw )
        
        self.lbar=Frame(self.canvas)
        self.tbar=Frame(self.lbar)
        
        self.newgif=PhotoImage(file="new.gif")
        self.insertgif=PhotoImage(file="insert.gif")
        self.donegif=PhotoImage(file="done.gif")
        self.newb = Button(self.tbar,command=self.newPose,image=self.newgif,width="20",height="20")
        self.insertb = Button(self.tbar,command=self.insertPose,image=self.insertgif,width="20",height="20")
        self.doneb = Button(self.tbar,command=self.donePose,image=self.donegif,width="20",height="20")
        
        self.newb.pack(side=LEFT)
        self.insertb.pack(side=LEFT)
        
        self.config(cursor="pencil")
        self.controlPoints=[controlHead(self)]
    def centrarTodo(self):
        minp=[9999999,999999]
        maxp=[0,0]
        for p in self.head:
            minp[0]=min(p[0],minp[0])
            minp[1]=min(p[1],minp[1])
            maxp[0]=max(p[0],maxp[0])
            maxp[1]=max(p[1],maxp[1])
        for s in self.strokes:
            for p in s:
                minp[0]=min(p[0],minp[0])
                minp[1]=min(p[1],minp[1])
                maxp[0]=max(p[0],maxp[0])
                maxp[1]=max(p[1],maxp[1])
        cm=[self.canvas.winfo_width()/2,self.canvas.winfo_height()/2]
        pm=[(minp[0]+maxp[0])/2,(minp[1]+maxp[1])/2]
        c=Centroid(self.head)
        offset=[c[0]-pm[0],c[1]-pm[1]]
        newp=[cm[0]+offset[0],cm[1]+offset[1]]
        self.head=TranslateTo(self.head,newp)
        for i in range(len(self.strokes)):
            c=Centroid(self.strokes[i])
            offset=[c[0]-pm[0],c[1]-pm[1]]
            newp=[cm[0]+offset[0],cm[1]+offset[1]]
            self.strokes[i]=TranslateTo(self.strokes[i],newp)
            
            
    def initEditPose(self,i):
        self.ant=[self.head[:],self.strokes[:],self.lastsize[:]]
        global animator

        ojos=animator.stack[i][0][0]
        headp=animator.stack[i][1][0]

        diametro=Distance(self.head[0],self.head[int(len(self.head)/2)])
        r=diametro/2
        
        self.head=TranslateTo(self.head,headp)
        self.strokes=animator.stack[i][2:][:]
        self.centrarTodo()
        self.hechoAction()
        self.controlPoints[1].locations[0]=ojos[0]
        self.controlPoints[1].locations[1]=ojos[1]
        self.canvas.repaint(0)
    def donePose(self):
        self.newPose()
        self.head=self.ant[0][:]
        self.strokes=self.ant[1][:]
        self.lastsize=self.ant[2][:]
        self.ant=[]
        global animator
        animator.endEditPose()
        self.canvas.config(highlightbackground="SystemButtonFace")
        if len(self.strokes)==5:
            self.hechoAction()
    def insertPose(self):
        global animator
        p=self.standarPose()
        animator.agregarPose(p)
    def newPose(self):
##        self.head=getCircle(9*0.01,250)
        self.head=TranslateTo(self.head,[self.canvas.winfo_width()*0.5,self.canvas.winfo_height()*0.20])
        self.lastsize=[float(self.canvas.winfo_width()),float(self.canvas.winfo_height())]
        self.canvas.bind( "<B1-Motion>", paintDraw )
        self.canvas.bind( "<Button-1>", clickedDraw )
        self.canvas.bind( "<ButtonRelease-1>", releaseDraw )
        self.canvas.bind( "<Button-3>", clicked3Draw )
        self.currentpoints=[]
        self.strokes=[]
        self.config(cursor="pencil")
        self.controlPoints=[controlHead(self)]
        self.finished=False
        self.canvas.repaint(0)
        self.tbar.pack_forget()
        self.lbar.pack_forget()
    def standarPose(self):
        diametro=Distance(self.head[0],self.head[int(len(self.head)/2)])
        r=diametro/2
        pose=self.getPose()
        pose=self.unir(pose)
        pose=TranslateTo(pose,[0,0])
        for i in range(len(pose)):
            pose[i][0]=pose[i][0]/r
            pose[i][1]=pose[i][1]/r
        return [np.array([self.controlPoints[1].locations])]+self.separar(pose)
    def getPose(self):
        pose=[]
        pose.append(np.array([Centroid(self.head)]))
##        pose.append(self.controlPoints[1].locations)
        for s in self.strokes:
            pose.append(s[:])
        return pose
    def unir(self,strokes):
        unidos=[]
        for s in strokes:
            unidos+=list(s)
        return unidos
    def separar(self,pose):
        separados=[]
        separados.append([pose[0]])
##        separados.append(pose[1])
        stroke=[]
        for i in range(1,len(pose)):
            stroke.append(pose[i])
            if(len(stroke)==5):
                separados.append(stroke)
                stroke=[]
        return separados      
    def limitAll(self):
        for i in range(len(self.strokes)):
            for j in range(len(self.strokes[i])):
                self.limitPoint(i,j)
    def limitAllNF(self):
        for i in range(len(self.strokes)):
            for j in range(len(self.strokes[i])):
                self.limitPointNF(i,j)
    
    def limitPointNF(self,i,j):
        if j==0:
            return
        maxl=self.getMaxLength(i)
        maxl=maxl/(len(self.strokes[i])-1)
        pi=self.strokes[i][j-1]
        pf=self.strokes[i][j]
        d=Distance(pi,pf)
        if d<=maxl:
            return
        pd=[pf[0]-pi[0],pf[1]-pi[1]]
        m=maxl/d
        end=[pi[0]+pd[0]*m,pi[1]+pd[1]*m]
        offset=[end[0]-self.strokes[i][j][0],end[1]-self.strokes[i][j][1]]
        self.strokes[i][j]=end
    def limitPoint(self,i,j):
        if j==0:
            return
        maxl=self.getMaxLength(i)
        maxl=maxl/(len(self.strokes[i])-1)
        pi=self.strokes[i][j-1]
        pf=self.strokes[i][j]
        d=Distance(pi,pf)
        if d<=maxl:
            return
        pd=[pf[0]-pi[0],pf[1]-pi[1]]
        m=maxl/d
        end=[pi[0]+pd[0]*m,pi[1]+pd[1]*m]
        offset=[end[0]-self.strokes[i][j][0],end[1]-self.strokes[i][j][1]]
        for n in range(j,len(self.strokes[i])):
            self.strokes[i][n][0]+=offset[0]
            self.strokes[i][n][1]+=offset[1]
        
    def limitStroke(self,i):
        maxl=self.getMaxLength(i)
        actl=0
        for j in range(1,len(self.strokes[i])):
           actl+=Distance(self.strokes[i][j],self.strokes[i][j-1])
        if actl<=maxl:
            return
        m=maxl/actl
        c=Centroid(self.strokes[i])
        newstrokes=self.strokes[i]
        newstrokes=TranslateTo(newstrokes,[0,0])
        for j in range(len(newstrokes)):
            newstrokes[j]*=m
        newstrokes=TranslateTo(newstrokes,c)
        self.strokes[i]=newstrokes
    def getMaxLength(self,i):
        diametro=Distance(self.head[0],self.head[int(len(self.head)/2)])
        length=diametro*2.4
        return self.proportions[i]*length
    def dibujarOjos(self):
        diametro=Distance(self.head[0],self.head[int(len(self.head)/2)])
        rad=diametro/2
        radOjos=diametro/10
        fullsep=diametro/3
        c=Centroid(self.head)
        sep=fullsep*(1-abs(self.controlPoints[1].locations[0]))
        x=c[0]+self.controlPoints[1].locations[0]*rad
        y=c[1]+self.controlPoints[1].locations[1]*rad
        r=radOjos
        self.canvas.create_oval(x+sep/2-r, y-r, x+sep/2+r, y+r,outline="black", width=3)
        self.canvas.create_oval(x-sep/2-r, y-r, x-sep/2+r, y+r,outline="black", width=3)
    def normalizePositionAll(self):
        if self.finished:
            self.normalizePositionAllFinished()
            return
        for i in range(len(self.strokes)):
            c=Centroid(self.strokes[i])
            if(i==0):
                cuello=self.getCuello(self.strokes[i])
                delta=[cuello[0]-self.strokes[i][0][0],cuello[1]-self.strokes[i][0][1]]
            else:
                jointArms=self.strokes[0][1]
                jointLegs=self.strokes[0][-1]
                distancearms=min(Distance(jointArms,self.strokes[i][0]),Distance(jointArms,self.strokes[i][-1]))
                distancelegs=min(Distance(jointLegs,self.strokes[i][0]),Distance(jointLegs,self.strokes[i][-1]))
                if(distancearms<distancelegs):
                    delta=[jointArms[0]-self.strokes[i][0][0],jointArms[1]-self.strokes[i][0][1]]
                else:
                    delta=[jointLegs[0]-self.strokes[i][0][0],jointLegs[1]-self.strokes[i][0][1]]
            self.strokes[i]=TranslateTo(self.strokes[i],[c[0]+delta[0],c[1]+delta[1]])
    def normalizePositionAllFinished(self):
        for i in range(len(self.strokes)):
            c=Centroid(self.strokes[i])
            if(i==0):
                cuello=self.getCuello(self.strokes[i])
                delta=[cuello[0]-self.strokes[i][0][0],cuello[1]-self.strokes[i][0][1]]
            else:
                if(i==1 or i==2):
                    jointArms=self.strokes[0][1]
                    delta=[jointArms[0]-self.strokes[i][0][0],jointArms[1]-self.strokes[i][0][1]]
                else:
                    jointLegs=self.strokes[0][-1]
                    delta=[jointLegs[0]-self.strokes[i][0][0],jointLegs[1]-self.strokes[i][0][1]]
            self.strokes[i]=TranslateTo(self.strokes[i],[c[0]+delta[0],c[1]+delta[1]])
    def normalize(self,points):
        c=Centroid(points)
        newpoints=TranslateTo(points,[0,0])
        maxy=0
        miny=99999999999
        for p in newpoints:
            maxy=max(maxy,p[1])
            miny=min(miny,p[1])
        newpoints=ScaleToY(newpoints,(maxy-miny)*self.canvas.winfo_height()/self.lastsize[1])
        newpoints=TranslateTo(newpoints,[c[0]*self.canvas.winfo_width()/self.lastsize[0],c[1]*self.canvas.winfo_height()/self.lastsize[1]])
        return newpoints
    def numBrazos(self):
        num=0
        jointArms=self.strokes[0][1]
        jointLegs=self.strokes[0][-1]
        for i in range(1,len(self.strokes)):
            distancearms=min(Distance(jointArms,self.strokes[i][0]),Distance(jointArms,self.strokes[i][-1]))
            distancelegs=min(Distance(jointLegs,self.strokes[i][0]),Distance(jointLegs,self.strokes[i][-1]))
            if(distancearms<distancelegs):
                num+=1
        return num
    def numPiernas(self):
        return len(self.strokes)-self.numBrazos()-1
    def completePoints(self,points):
        newpoints=points[:]
        if len(self.strokes)==0:
            ch=Centroid(self.head)
            if(Distance(ch,newpoints[-1])<=Distance(ch,newpoints[0])):
                newpoints.reverse()
            newpoints=self.arreglarCuello(newpoints)
            return newpoints
        jointArms=self.strokes[0][1]
        jointLegs=self.strokes[0][-1]
        distancearms=min(Distance(jointArms,newpoints[0]),Distance(jointArms,newpoints[-1]))
        distancelegs=min(Distance(jointLegs,newpoints[0]),Distance(jointLegs,newpoints[-1]))
        if(self.numPiernas()>=2 or (distancearms<distancelegs and self.numBrazos()<2)):
            if(Distance(jointArms,newpoints[0])>Distance(jointArms,newpoints[-1])):
                newpoints.reverse()
            newpoints.insert(0,jointArms)
        else:
            if(Distance(jointLegs,newpoints[0])>Distance(jointLegs,newpoints[-1])):
                newpoints.reverse()
            newpoints.insert(0,jointLegs)
        return newpoints
    def redraw(self):
        self.head=self.normalize(self.head)
        for i in range(len(self.strokes)):
            self.strokes[i]=self.normalize(self.strokes[i])
        self.normalizePositionAll()
        self.drawAllNormal(self.strokes,"black")
        self.drawPoints(self.head,"black")
        self.lastsize=[float(self.canvas.winfo_width()),float(self.canvas.winfo_height())]
        for cp in self.controlPoints:
                cp.drawCP()
        if self.finished:
            self.dibujarOjos()
    def isHuman(self):
        return len(self.strokes)==6
    def hechoAction(self):
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.unbind("<Button-3>")
        
        self.canvas.bind( "<B1-Motion>", self.selectMove )
        self.canvas.bind( "<Button-1>", self.selectClick)
        self.canvas.bind( "<ButtonRelease-1>", self.selectRelease )
        
        self.config(cursor="arrow")
        self.ordenarBody()
        self.finished=True
        self.controlPoints+=[controlEyes(self,[0.0,0.0]),
                            controlPoint(self,[[0,1],[1,0],[2,0]]),
                            controlPoint(self,[[0,4],[3,0],[4,0]]),
                            controlPoint(self,[[1,2]]),
                            controlPoint(self,[[1,4]]),
                            controlPoint(self,[[2,2]]),
                            controlPoint(self,[[2,4]]),
                            controlPoint(self,[[3,2]]),
                            controlPoint(self,[[3,4]]),
                            controlPoint(self,[[4,2]]),
                            controlPoint(self,[[4,4]]),
                            controlPoint(self,[[0,2]])
                            ]
        self.canvas.pack_propagate(False)
        self.tbar.pack(side=LEFT)
        self.lbar.pack(side=TOP,fill=X)
        self.doneb.pack_forget()
        if len(self.ant)!=0:
            self.doneb.pack(side=LEFT)
            self.canvas.config(highlightbackground="red")
        
    def estaDentroCabeza(self,p,c,rad):
        return Distance(p,c)<=rad
    def getCuello(self,points):
        if(len(points)==0):
            return points
        c=Centroid(self.head)
        rad=Distance(self.head[0],self.head[len(self.head)/2])/2
        start=points[0]
        end=c
        delta=[end[0]-start[0],end[1]-start[1]]
        p=1-(float(rad)/Distance(start,end))
        end=[start[0]+p*delta[0],start[1]+p*delta[1]]
        return end
    def arreglarCuello(self,points):
        newpoints=points[:]
        c=Centroid(self.head)
        rad=Distance(self.head[0],self.head[len(self.head)/2])/2
        while(len(newpoints)>0 and self.estaDentroCabeza(newpoints[0],c,rad)):
            newpoints = newpoints[1:]
        newpoints.insert(0,self.getCuello(newpoints))
        return newpoints
    def drawAllNormal(self,All,color):
        for a in All:
            self.drawPoints(toSpline(a),color)
    def selectClick(self,evt):
        for cp in self.controlPoints:
            if(cp.isSelected(evt.x,evt.y)):
                self.selectedCP=cp
                break
    def selectMove(self,evt):
        if self.selectedCP==0:
            return
        self.selectedCP.moveTo(evt.x,evt.y)
        if(len(self.ant)!=0):
            global animator
            p=self.standarPose()
            animator.refreshPose(p)
        self.canvas.repaint(0)
    def selectRelease(self,evt):
        self.selectedCP=0    
    def drawAllSmooth(self,All,color):
        for a in All:
            self.drawPoints(a,color)
        self.drawPoints(self.head,color)
    def drawAllRaw(self,All):
        colors=["blue","green","red","black","yellow"]
        for i in range(len(All)):
            self.drawPointsStick(All[i],colors[i])
        self.drawPoints(self.head,"magenta")
    def drawPoints(self,todraw,color):
        global primera 
        primera=[]
        for a in todraw:
            drawPluma(self.canvas,a,color)
    def drawPointsStick(self,todraw,color):
        primera=0
        for a in todraw:
            if primera==0:
                primera=a
            else:
                self.canvas.create_line(primera[0],primera[1],a[0],a[1],fill = color,width=2)
                primera=a
            self.canvas.create_oval(a[0]-2.5,a[1]-2.5,a[0]+2.5,a[1]+2.5,fill="red")
        self.canvas.create_oval(todraw[0][0]-2.5,todraw[0][1]-2.5,todraw[0][0]+2.5,todraw[0][1]+2.5,fill="blue")
        self.canvas.create_oval(todraw[len(todraw)-1][0]-2.5,todraw[len(todraw)-1][1]-2.5,todraw[len(todraw)-1][0]+2.5,todraw[len(todraw)-1][1]+2.5,fill="orange")
    def ordenarBody(self):
        brazos=[]
        newstrokes=[self.strokes[0]]
        jointArms=self.strokes[0][1]
        jointLegs=self.strokes[0][-1]
        for i in range(1,len(self.strokes)):
            distancearms=min(Distance(jointArms,self.strokes[i][0]),Distance(jointArms,self.strokes[i][len(self.strokes[i])-1]))
            distancelegs=min(Distance(jointLegs,self.strokes[i][0]),Distance(jointLegs,self.strokes[i][len(self.strokes[i])-1]))
            if(distancearms<distancelegs):
                brazos.append(i)    
            if(len(brazos)==2):
                break
        if(Centroid(self.strokes[brazos[0]])[0]>Centroid(self.strokes[brazos[1]])[0]):
            brazos.reverse()
        newstrokes.append(self.strokes[brazos[0]])
        newstrokes.append(self.strokes[brazos[1]])
        indexes=[1,2,3,4]
        indexes.remove(brazos[0])
        indexes.remove(brazos[1])
        if(Centroid(self.strokes[indexes[0]])[0]>Centroid(self.strokes[indexes[1]])[0]):
            newstrokes.append(self.strokes[indexes[1]])
            newstrokes.append(self.strokes[indexes[0]])
        else:
            newstrokes.append(self.strokes[indexes[0]])
            newstrokes.append(self.strokes[indexes[1]])
        self.strokes=newstrokes
def cornersIndex(corners,i):
        for c in corners:
            if i == c[0]:
                return c[1]
        return -1
def TranslateTo(pts,p):
   c=Centroid(pts)
   return np.subtract(np.add(pts,[p]*len(pts)),[c]*len(pts))
def getInclinacion(stroke):
    c=Centroid(stroke)
    v=0
    for p in stroke:
       v+=(p[0]-c[0])**2
    v/=len(stroke)
    return v
def getTorso(body):
    maxy=0
    eliminar=-1
    ignorar=[]
    for i in range(len(body)):
        y=Centroid(body[i])[1]
        if y> maxy:
            maxy=y
            eliminar=i
    ignorar.append(eliminar)
    mind=999999999
    eliminar=-1
    for i in range(len(body)):
        y=Centroid(body[i])[1]
        if y>maxy and not(i in ignorar):
            maxy=y
            eliminar=i
    ignorar.append(eliminar)
    res=-1
    for i in range(len(body)):
        inc=getInclinacion(body[i])
        if inc< mind and not(i in ignorar):
            mind=inc
            res=i
    return res
def PathLength(pts):
   d = 0
   primera=[]
   for a in pts:
          if len(primera)==0:
             primera=a
          else:
             d+=Distance(primera,a)
             primera=a
   return d
def Vectorize(points,oSensitive=True):
    centroid=Centroid(points)
    points=TranslateTo(points,centroid)
    indicativeAngle=math.atan2(points[0][1],points[0][0])
    if oSensitive:
        baseOrientation=(math.pi/4)*math.floor((indicativeAngle+math.pi/8)/(math.pi/4))
        delta=baseOrientation-indicativeAngle
    else:
        delta=-indicativeAngle
    total=0
    vector=[]
    for p in points:
        newX=p[0]*math.cos(delta)-p[1]*math.sin(delta)
        newY=p[1]*math.cos(delta)+p[0]*math.sin(delta)
        vector.append(newX)
        vector.append(newY)
        total+=newX**2+newY**2
    magnitude=math.sqrt(total)
    for i in range(len(vector)):
        vector[i]=vector[i]/magnitude
    return vector
def optimalCosineDistance(A,B):
    a=0
    b=0
    for i in range(0,len(A),2):
        a+=A[i]*B[i]+A[i+1]*B[i+1]
        b+=A[i]*B[i+1]-A[i+1]*B[i]
    angle=math.atan(b/a)
    return math.acos(a*math.cos(angle)+b*math.sin(angle))
def distanceAtAngle(A,B,angle):
    newPoints=RotateBy(A,angle)
    d=compareStrokes(newPoints,B)
    return d
def distanceAtBestAngle(A,B):
    phi=(math.sqrt(5)-1)/2
    theta=math.pi/2
    delta=2*math.pi/180
    x1=phi*-theta+(1-phi)*theta
    f1=distanceAtAngle(A,B,x1)
    x2=(1-phi)*-theta+phi*theta
    f2=distanceAtAngle(A,B,x2)
    ta=-theta
    tb=theta
    while abs(tb-(ta))>delta:
        if f1<f2:
            tb=x2
            x2=x1
            f2=f1
            x1=phi*ta+(1-phi)*tb
            f1=distanceAtAngle(A,B,x1)
        else:
            ta=x1
            x1=x2
            f1=f2
            x2=(1-phi)*ta+phi*tb
            f2=distanceAtAngle(A,B,x2)
    return min(f1,f2)
def ordenarStroke(stroke):
    if(stroke[0][1]>stroke[len(stroke)-1][1]):
        return stroke[:]
    else:
        return list(reversed(stroke))
def toSpline(points):
    num=20
    return BezierSpline(points,2,num)#bspline2D(np.array(points),2,100)
def toShownStick(points):
    return BezierSpline(points,2,3)
def toStick(points):
    return BezierSpline(points,2,5)#3,4
def paintDraw(event):
    global draw
    if draw.selectedCP!=0:
        draw.selectedCP.moveTo(event.x,event.y)
        draw.canvas.repaint(0)
        return
    drawLapiz(draw.canvas,draw.currentpoints[len(draw.currentpoints)-1])
    draw.currentpoints.append([event.x,event.y])
def clickedDraw(event):
    global draw
    global primera
    primera=[]
    draw.currentpoints=[]
    for cp in draw.controlPoints:
            if(cp.isSelected(event.x,event.y)):
                draw.selectedCP=cp
                return
    draw.currentpoints.append([event.x,event.y])
def offset(points,p):
    for i in range(len(points)):
        for j in range(len(points[i])):
            points[i][j]+=p[j]
def Distance(p1, p2):
   dx = p2[0] - p1[0]
   dy = p2[1] - p1[1]
   return math.sqrt(dx * dx + dy * dy)
def Centroid(pts):
   x = 0
   y = 0
   for a in pts:
      x += a[0]
      y += a[1]
   x /= len(pts)
   y /= len(pts)
   return [x,y]
def releaseDraw(event):
    global draw
    if draw.selectedCP!=0:
        draw.selectedCP=0
        return   
    draw.currentpoints.append([event.x,event.y])
    drawLapiz(draw.canvas,draw.currentpoints[len(draw.currentpoints)-1])
    draw.strokes.append(toStick(draw.completePoints(draw.currentpoints)))
    if(len(draw.currentpoints)<10 or len(draw.strokes[len(draw.strokes)-1])<=1):
        clicked3Draw(0)
    draw.currentpoints=[]
    if(len(draw.strokes)==5):
        draw.hechoAction()
    draw.limitAll()
    draw.canvas.repaint(0)
def clicked3Draw( event ):
    global draw
    if len(draw.strokes)>0:
        popped=draw.strokes.pop()
        draw.canvas.repaint(0)
def Resample(pts,n):
   points=pts[:]
   I=PathLength(points)/(n-1)
   D=0
   newpoints=[list(points[0])]
   i=1
   while i<len(points):
      d=Distance(points[i-1],points[i])
      if (D+d)>=I:
         qx=points[i-1][0]+((I-D)/d)*(points[i][0]-points[i-1][0])
         qy=points[i-1][1]+((I-D)/d)*(points[i][1]-points[i-1][1])
         q=[qx,qy]
         points.insert(i,q) 
         newpoints.append(q)
         D=0
      else:
         D+=d
      i+=1
   if(len(newpoints)<n):
      newpoints.append(pts[len(pts)-1])
   return newpoints
def getRoundest(pts):
    minv=99999999
    result=-1
    for i in range(len(pts)):
        error=RecognizeCircle(pts[i])
        if(error<minv):
            minv=error
            result=i
    return result
def getRadius(pts):
    data=[]
    center=Centroid(pts)
    mean=0
    for p in pts:
        mean+=Distance(p,center)
    mean=mean/len(pts)
    return mean
def BoundingBox(pts):
   minX=99999999.0
   maxX=-minX
   minY=99999999.0
   maxY=-minY
   for a in pts:
      minX=min(minX,float(a[0]))
      minY=min(minY,float(a[1]))
      maxX=max(maxX,float(a[0]))
      maxY=max(maxY,float(a[1]))
   return [minX, minY, maxX - minX, maxY - minY]
def ScaleTo(pts, size):
    B=BoundingBox(pts)
    newpoints=[]
    return pts*[size/B[2],size/B[3]]
def ScaleToMax(pts, size):
    B=BoundingBox(pts)
    newpoints=[]
    return pts*[max(size/B[2],size/B[3]),max(size/B[2],size/B[3])]
def ScaleToY(pts, size):
    B=BoundingBox(pts)
    newpoints=[]
    return pts*[size/B[3],size/B[3]]
def IndicativeAngle(pts):
   c = Centroid(pts)
   return math.atan2(c[1] - pts[0][1], c[0] - pts[0][0]);
def RotateBy(pts, radians):
   c = Centroid(pts)
   cos = math.cos(radians)
   sin = math.sin(radians)
   newpoints =[]
   for a in pts:
      qx=(a[0]-c[0])*cos-(a[1]-c[1])*sin+c[0]
      qy=(a[0]-c[0])*sin+(a[1]-c[1])*cos+c[1]
      newpoints.append([qx,qy])
   return newpoints

def RecognizeCircle(pts):
    num=250
    size=100
    r=size/2
    c=Centroid(pts)
    a=IndicativeAngle(pts)
    newpts=RotateBy(pts,math.pi-a)
    newpts=Resample(newpts,num)
    newpts=TranslateTo(newpts,[0,0])
    newpts=ScaleTo(newpts, 100)
    Rcircle=getCircle(r,num)
    Lcircle=Rcircle[:]
    Lcircle.reverse()
    errorR=compareStrokes(Rcircle,newpts)
    errorL=compareStrokes(Lcircle,newpts)
    error=min(errorR,errorL)
    return error
def getCircle(r,n):
    newpoints=[]
    for i in range(0,n):
        newpoints.append([r*math.cos(2*math.pi*i/n),r*math.sin(2*math.pi*i/n)])
    return newpoints
def linealizarNew(datos,iteraciones):
    arrays=[datos]
    while(iteraciones>0):
        newarrays=[]
        subscore=getSublineScore(arrays)
        toremove=arrays[subscore[0]]

        for i in range(len(arrays)):
            if i==subscore[0]:
                newarrays.append(toremove[:subscore[1]])
                newarrays.append(toremove[subscore[1]:])
            else:
                newarrays.append(arrays[i])
        arrays=newarrays
        iteraciones-=1
    
    res=[]
    res.append(arrays[0][0])
    for a in arrays:
        res.append(a[len(a)-1])
    return res
    
def getLineScore(datos):
    Pi=datos[0]
    Pf=datos[len(datos)-1]
    Pd=[Pf[0]-Pi[0],Pf[1]-Pi[1]]
    maxD=0
    curri=-1
    Pideal=[0,0]
    for i in range(len(datos)):
        Pideal[0]=Pi[0]+((float(i)/len(datos))*Pd[0])
        Pideal[1]=Pi[1]+((float(i)/len(datos))*Pd[1])
        d=((Pideal[0]-datos[i][0])**2)+((Pideal[1]-datos[i][1])**2)
        if(d>maxD):
            maxD=d
            curri=i
    return [maxD,curri]
def getSublineScore(array):
    maxScore=[0,0]
    linei=-1
    for i in range(len(array)):
        score=getLineScore(array[i])
        if(score[0]>maxScore[0]):
            maxScore=[score[0],score[1]]
            linei=i
    return [linei,maxScore[1]]
class ReproductorFrame(CustomFrame):
    canvas=0    
    def __init__(self,master):
        CustomFrame.__init__(self,master,"área de reproducción")
        self.grid(column=1,row=1,sticky=W+E+N+S)
        self.canvas=CustomCanvas(self)
        self.canvas.pack(expand=1,fill=BOTH)
        
class AvatarFrame(CustomFrame):
    canvas=0
    
    def __init__(self,master):
        CustomFrame.__init__(self,master,"área del avatar")
        self.grid(column=2,row=1,sticky=W+E+N+S)
        self.canvas=CustomCanvas(self)
        self.canvas.pack(expand=1,fill=BOTH)

global customframes       
global animator
global draw
global reproductor
global avatar
global primera
primera=[]
animator=0
draw=0
w=0
def drawLapiz(canvas,pt):
    global primera
    if len(primera)==0:
        primera=pt
    else:
        canvas.create_line(primera[0],primera[1],pt[0],pt[1],fill ="gray",width=3)
        primera=pt
def drawPluma(canvas,pt,color):
    global primera
    if len(primera)==0:
        primera=pt
    else:
        canvas.create_line(primera[0],primera[1],pt[0],pt[1],fill =color,width=2)
        primera=pt
def drawLines(pt,color):
    global primera
    if primera==0:
        primera=pt
    else:
        vision.canvas.create_line(primera[0],primera[1],pt[0],pt[1],fill =color ,width=3)
        primera=pt
def drawLinesSlide(pt,color):
    global primera
    if primera==0:
        primera=pt
    else:
        vision.canvas.create_line(primera[0],primera[1],pt[0],pt[1],fill =color ,width=3)
        primera=pt
def drawEmptyOval(pt,rad,color):
    vision.canvas.create_oval(pt[0]-rad,pt[1]-rad,pt[0]+rad,pt[1]+rad,outline=color,width=3)
def drawOvals(pt,color):
    vision.canvas.create_oval(pt[0]-5,pt[1]-5,pt[0]+5,pt[1]+5,fill=color)
def saveCallBack():
    global master
    filename = asksaveasfilename(parent=master)
    if filename=='':
        return
    filename = open(filename, 'w')
    global animator
    s=animator.toString()
    filename.write(s)
    filename.close()
def loadCallBack():
    global master
    filename = askopenfilename(parent=master)
    if filename=='':
        return
    filename = open(filename, 'r')
    string=filename.read()
    global animator
    animator.loadAnimation(string)
    filename.close()
def init():
    global customframes
    customframes=[]
    global animator
    global draw
    global reproductor
    global master
    global avatar
    global onmarchvalue
    onmarchvalue=BooleanVar()
    global toolbar
    toolbar = Frame(master)
    global menubar
    menubar = Menu(toolbar,tearoff=False)
    fileMenu = Menu(toolbar,tearoff=False)
    global recentMenu
    recentMenu = Menu(toolbar,tearoff=False)
     
    menubar.add_cascade(label="Archivo", menu=fileMenu)
    fileMenu.add_command(label="Abrir", command=loadCallBack)
    fileMenu.add_command(label="Guardar", command=saveCallBack)
##    fileMenu.add_cascade(label="Abrir reciente", menu=recentMenu)
    fileMenu.add_command(label="Salir")
    
    animator=AnimationFrame(master)
    draw=DrawFrame(master)
    reproductor=ReproductorFrame(master)
    avatar=AvatarFrame(master)
    customframes.append(animator)
    customframes.append(draw)
    customframes.append(reproductor)
    customframes.append(avatar)
    
    infotoolbar=Frame(master,height=60)
    global labelY
    labelY=Label(infotoolbar,width=20)
    labelY.pack(side='right')
    global labelX
    labelX=Label(infotoolbar,width=10)
    labelX.pack(side='right')
    global labelIndex
    labelIndex=Label(infotoolbar,width=10)
    labelIndex.pack(side='right')
    global labelInfo
    labelInfo=Label(infotoolbar,width=20)
    labelInfo.pack(side='right')
    
    vistasSubMenu=Menu(toolbar,tearoff=False)
    master.configure(menu=menubar)
    wid=1000.0
    h=700.0
    ws = master.winfo_screenwidth()
    hs = master.winfo_screenheight()
    x = (ws/2) - (wid/2) 
    y = (hs/2) - (h/2)
    master.geometry('%dx%d+%d+%d' % (wid, h, x, y))
    master.deiconify()

global tasks
tasks=[]
isdone=False
global animation
animation=0

def avanzarWait(tk,num,txt):
    tk.msg.config(text=txt)
    tk.progress["value"]+=num
class WaitScreen(Tk):
    msg=0
    progress=0
    frame=0
    hilo=0
    def __init__(self,title):
        width=300
        height=60
        Tk.__init__(self)
        self.title(title)
        self.msg=Message(self, text="cargando...",width=600)
        self.msg.pack(expand=1,fill=X,side=TOP)
        self.progress = ttk.Progressbar(self, maximum=100)
        self.progress.pack(expand=1,fill=X,side=TOP)
        self.maxsize(width,height)
        self.minsize(width,height)
        ws = master.winfo_screenwidth()
        hs = master.winfo_screenheight()
        x = (ws/2) - (width/2) 
        y = (hs/2) - (height/2)
        self.geometry('%dx%d+%d+%d' % (width,height, x, y))
    def start(self,funcion):
        self.hilo=threading.Thread(target=funcion)
        self.hilo.setDaemon(True)
        self.hilo.start()
    def avanzar(self,n,title):
        global tasks
        tasks.append(lambda txt=title,tk=self,num=n:avanzarWait(tk,num,txt))
    def terminar(self):
        global tasks
        tasks.append(self.destroy)
init()
new_mainloop()
