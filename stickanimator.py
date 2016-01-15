import sys,math
from numpy import fft,arange, sin, pi
from scipy import signal as sign
import numpy as np 
import time
import ttk
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
    global_time=0
    start=time.time()
    end=time.time()
    while(True):
        master.update()
        end=time.time()
        dt=end-start
        start=end
        reproductor.advanceTime(dt)
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
    rendered=0
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
        self.rendered=[]
        
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
        self.canvas.config(xscrollcommand=self.scrollbarset)
        self.maxText = Text(self.botframe,height=1,width=6,font=self.customFont)
        self.maxText.bind("<Return>", self.changeMax)
        self.maxText.bind("<Button-1>", self.enableMax)
        self.canvas.config(scrollregion=(0,0,self.maxTime/self.spc,0))
        self.maxText.insert(INSERT,str(self.maxTime)+"s")
        
        self.maxText.pack(side=RIGHT)
        self.timepos=0
        self.pointer=0
    def scrollbarset(self,lo,hi):
        self.hbar.set(lo,hi)
        self.canvas.repaint(0)
    def deshacer(self,evt=None):
        if(len(self.undolist)!=0):
            self.redolist.append(self.stack[:])
            if(len(self.redolist)>self.maxdolist):
                self.redolist=self.redolist[1:]
            self.stack=self.undolist.pop()
        self.generatePoses()
    def rehacer(self,evt=None):
        if(len(self.redolist)!=0):
            self.undolist.append(self.stack[:])
            if(len(self.undolist)>self.maxdolist):
                self.undolist=self.undolist[1:]
            self.stack=self.redolist.pop()
        self.generatePoses()
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
        self.generatePoses()
        self.do()
    def pegar(self, evt):
        i=self.pointer
        act=len(self.stack)
        if i>=act:
            n=i-act+1
            self.stack=self.stack+([[]]*n)
        self.stack[self.pointer]=self.clipboard[:]
        self.generatePoses()
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
        #0.0620000362396
        interp=self.getInterpolation()
        self.interpoled=[]
        for v in interp:
            self.interpoled.append(self.separarPose(self.unvectorizePose(v)))
##        self.renderStack()
    def renderStack(self):
        self.rendered=[]
        for i in range(len(self.interpoled)):
            self.rendered.append(self.renderPose(i,self.interpoled[i]))
        self.canvas.repaint(0)
    def generatePoses(self):
##        self.curveBezier()
##        self.curveBezierPoints()
##        self.interpolateStack()
        self.curveSpline()
        self.canvas.repaint(0)
    def curveSpline(self,num=20):#num=2):
        #0.203000068665 num=20
        interp=[]
        for p in self.stack:
            if(len(p)!=0):
                interp.append(self.vectorizePose(self.unirPose(p)))
        if len(interp)<=2:
            self.interpolateStack()
            return
        n=len(interp)
        dim=len(interp[0])
        newint=[interp[0]]
        for i in range(1,len(interp)):
            point=[]
            for j in range(dim):
                point.append((interp[i][j]+interp[i-1][j])/2)
            newint.append(point)
            newint.append(interp[i])
        interp=newint
        K=3
        
        flen=((n-1)*num)+n
        interp_array=[]
        for i in range(dim):
            points=np.array(interp)
            x=bsplineAnt(points[:,i],K,flen)
            interp_array.append(x)
        interp_array=np.array(interp_array).T
        interp_array=list(interp_array)
        init=0
        end=-1
        n=0
        resampled=[]
        bet=0
        for i in range(1,len(self.stack)):
            if len(self.stack[i])==0:
                n+=1
            else:
                resampled+=[interp_array[(num+1)*bet]]
                end=i
                if(n>1):
                    resampled+=Resample(interp_array[1+((num+1)*bet):(num+1)*(bet+1)],n+2)[1:-1]
                else:
                    if(n==1):
                        resampled+=[interp_array[int((1+((num+1)*bet)+(num+1)*(bet+1))/2)]]
##                resampled+=[interp_array[(num+1)*(bet+1)]]
                bet+=1
                init=i                
                n=0
        resampled+=[interp_array[-1]]
        self.interpoled=[]
        for v in resampled:
            self.interpoled.append(self.separarPose(self.unvectorizePose(v)))

    def curveBezierPoints(self,num=2):
        interp=[]
        for p in self.stack:
            if(len(p)!=0):
                interp.append(self.unirPose(p))
        n=len(interp)
        flen=((n-1)*num)+n
        interp=np.array(interp)
        interp_array=[]
        for i in range(len(interp[0])):
            interp_array.append(BezierSpline(interp[:,i],2,flen))
        allresampled=[]
        for inarray in interp_array:
            init=0
            end=-1
            n=0
            resampled=[inarray[0]]
            bet=0
            for i in range(1,len(self.stack)):
                if len(self.stack[i])==0:
                    n+=1
                else:
                    end=i
                    if(n>1):
                        resampled+=Resample(inarray[1+((num+1)*bet):(num+1)*(bet+1)],n)
                    else:
                        if(n==1):
                            resampled+=[inarray[int((1+((num+1)*bet)+(num+1)*(bet+1))/2)]]
                    resampled+=[inarray[(num+1)*(bet+1)]]
                    bet+=1
                    init=i                
                    n=0
            allresampled.append(resampled)

        allresampled=np.array(allresampled).T
        self.interpoled=[]
        for v in allresampled:
            self.interpoled.append(self.separarPose(v))
    def curveBezier(self,num=2):
        
        interp=[]
        for p in self.stack:
            if(len(p)!=0):
                interp.append(self.vectorizePose(self.unirPose(p)))
        n=len(interp)
        flen=((n-1)*num)+n
        interp_array=BezierSpline(interp,2,flen)

        init=0
        end=-1
        n=0
        resampled=[interp_array[0]]

        bet=0
        for i in range(1,len(self.stack)):
            if len(self.stack[i])==0:
                n+=1
            else:
                end=i
                if(n>1):
                    resampled+=Resample(interp_array[1+((num+1)*bet):(num+1)*(bet+1)],n)
                else:
                    if(n==1):
                        resampled+=[interp_array[int((1+((num+1)*bet)+(num+1)*(bet+1))/2)]]
                resampled+=[interp_array[(num+1)*(bet+1)]]
                bet+=1
                init=i                
                n=0
        self.interpoled=[]
        for v in resampled:
            self.interpoled.append(self.separarPose(self.unvectorizePose(v)))
        
    def setMax(self,maxt):
        maxtimestack=len(self.stack)/self.fps
        self.maxTime=max(maxt,maxtimestack)
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
##    def drawStack(self):
##        ind=self.hbar.get()[0]
##        recorrido=ind*self.maxTime/self.spc
##        t=recorrido*self.spc
##        tf=t+(self.canvas.winfo_width()*self.spc)
##        f0=min(len(self.interpoled),int(t*self.fps))
##        ff=min(len(self.interpoled),int(tf*self.fps))
##        for i in range(f0,ff):
##            if i==self.pointer:
##                self.dibujarPose(i,self.stack[i],"blue")
##            else:
##                self.dibujarPose(i,self.stack[i])
    def drawInterpoled(self):
        i=self.pointer
        if i>=0 and i<len(self.interpoled):
            self.dibujarPose(i,self.interpoled[i],"gray70")
    def redraw(self):
        self.drawTimeline()
        self.drawAllTimeMarks()
        self.drawInterpoled()
##        self.drawStack()
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
        self.generatePoses()
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
        ind=self.hbar.get()[0]
        recorrido=ind*self.maxTime/self.spc
        t=recorrido*self.spc
        tf=t+(self.canvas.winfo_width()*self.spc)
        for i in range(int(t),int(tf)+1):
            self.drawTimeMark(float(i))
    def drawTimeMark(self,t):
        x=t/self.spc
        he=self.canvas.winfo_height()
        self.canvas.create_line(x,he-self.h_off,x,he-self.h_off/2,fill="orange",width=3)
        self.canvas.create_text(x,he-self.h_off/2,anchor="nw", text="{0:.2f}".format(t)+"s")        
    def drawTimeline(self):
        wid=self.maxTime/self.spc
        he=self.canvas.winfo_height()
        i=self.hbar.get()[0]
        recorrido=i*self.maxTime/self.spc
        self.canvas.create_line(recorrido,he-self.h_off,recorrido+self.canvas.winfo_width(),he-self.h_off,fill = "yellow",width=6)
        self.canvas.create_line(wid-20,he-self.h_off,wid,he-self.h_off,fill = "red",width=6)
##        self.canvas.create_line(0,he-self.h_off,wid,he-self.h_off,fill = "yellow",width=6)
##        self.canvas.create_line(wid-20,he-self.h_off,wid,he-self.h_off,fill = "red",width=6)
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
        self.generatePoses()
    def deletePose(self,evt):
        if(self.timepos!=0 and self.timepos<len(self.stack) and len(self.stack[self.timepos])!=0):
            if(tkMessageBox.askyesno("Eliminar pose","Seguro que quieres eliminar esta pose?")):
                self.stack[self.timepos]=[]
        self.do()
        self.generatePoses()
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
        self.generatePoses()
    def refreshPose(self,pose):
        self.stack[self.poseE]=self.scalePose(pose)
##        self.generatePoses()
        self.canvas.repaint(0)
    def endEditPose(self):
        self.generatePoses()
        self.poseE=-1
    def scalePose(self,pose):
        escalado=[]
        escalado.append(pose[0])
        for i in range(1,len(pose)):
            escalado.append([])
            for p in pose[i]:
                escalado[-1].append([p[0]*self.rad,p[1]*self.rad])
        return escalado
    def renderPose(self,index,pose,num=8):
        todraw=[]
        ojos=pose[0]
        for i in range(1,len(pose)):
            todraw+=pose[i]
        c=Centroid(todraw)
        p=[self.getStackCanvas(index),(self.canvas.winfo_height())/2]
        todraw=TranslateTo(todraw,[p[0],p[1]])
        head=getCircle(self.rad,num)
        head=TranslateTo(head,todraw[0])
        eyes=self.renderOjos(todraw[0],ojos[0])
        separados=[]
        stroke=[]
        for i in range(1,len(todraw)):
            stroke.append(todraw[i])
            if(len(stroke)==5):
                separados.append(stroke)
                stroke=[]
        res=[]
        res+=[eyes]
        for s in separados:
            res+=toSpline(s,num)
        res=list(res)+list(head)
        return res
    def drawPose(self,pose,color="black"):
        ojos=pose[0][:]
        self.drawOjos(ojos,color)
        todraw=pose[1:]
        num=len(todraw)/6
        for i in range(0,len(todraw),num):
            self.drawPoints(todraw[i:i+num],color)
    def dibujarPose(self,index,pose,color="black",num=9):
        if len(pose)==0:
            return
        todraw=[]
        ojos=pose[0]
        for i in range(1,len(pose)):
            todraw+=pose[i]
        c=Centroid(todraw)
        p=[self.getStackCanvas(index),(self.canvas.winfo_height())/2]
        todraw=TranslateTo(todraw,[p[0],p[1]])
        head=getCircle(self.rad,num)
        head=TranslateTo(head,todraw[0])
        self.dibujarOjos(todraw[0],ojos[0],color)
        separados=[]
        stroke=[]
        for i in range(1,len(todraw)):
            stroke.append(todraw[i])
            if(len(stroke)==5):
                separados.append(stroke)
                stroke=[]
        self.drawAllNormal(separados,color,num)
        self.drawPoints(head,color)
    def drawAllNormal(self,All,color,num=20):
        for a in All:
            self.drawPoints(toSpline(a,num),color)
    def drawPoints(self,todraw,color):
        global primera 
        primera=[]
        for a in todraw:
            drawPluma(self.canvas,a,color)
    def renderOjos(self,head,ojos):
        c=head
        x=c[0]+ojos[0]*self.rad
        y=c[1]+ojos[1]*self.rad
        diametro=self.rad*2
        radOjos=diametro/12
        fullsep=diametro/3 
        sep=fullsep*(1-abs(ojos[0]))
        r=radOjos
        return [x,y,sep,r]
    def dibujarOjos(self,head,ojos,color):
        diametro=self.rad*2
        radOjos=diametro/12
        fullsep=diametro/3
        c=head
        sep=fullsep*(1-abs(ojos[0]))
        x=c[0]+ojos[0]*self.rad
        y=c[1]+ojos[1]*self.rad
        r=radOjos
        self.canvas.create_oval(x+sep/2-r, y-r, x+sep/2+r, y+r,outline=color, width=3)
        self.canvas.create_oval(x-sep/2-r, y-r, x-sep/2+r, y+r,outline=color, width=3)
    def drawOjos(self,ojos,color):
        x=ojos[0]
        y=ojos[1]
        sep=ojos[2]
        r=ojos[3]
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
        self.newPose()
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
        self.canvas.repaint(0)
    def insertPose(self):
        global animator
        p=self.standarPose()
        animator.agregarPose(p)
    def newPose(self):
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
        radOjos=diametro/12
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
def toSpline(points,num=20):
    return BezierSpline(points,2,num)#bspline2D(np.array(points),2,100)
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
    dim=len(p1)
    res=0
    for i in range(dim):
        res+=(p2[i]-p1[i])**2
    return math.sqrt(res)
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
    dim=len(points[0])
    while i<len(points):
        d=Distance(points[i-1],points[i])
        p=[]
        if (D+d)>=I:
            for di in range(dim):
                q=points[i-1][di]+((I-D)/d)*(points[i][di]-points[i-1][di])
                p.append(q)
            points.insert(i,p) 
            newpoints.append(p)
            D=0
        else:
            D+=d
        i+=1
    if(len(newpoints)<n):
        newpoints.append(pts[len(pts)-1])
    return newpoints
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
def getCircle(r,n):
    n=n-1
    newpoints=[]
    for i in range(0,n):
        newpoints.append([r*math.cos(2*math.pi*i/n),r*math.sin(2*math.pi*i/n)])
    newpoints.append([r,0])
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
    reproduccion_toolbar=0
    animframe=0
    play_pause=0
    stop_button=0
    playgif=0
    stopgif=0
    rad=25
    pausegif=0
    progress=0
    playing=0
    globaltime=0
    def __init__(self,master):
        CustomFrame.__init__(self,master,"área de reproducción")
        self.grid(column=1,row=1,sticky=W+E+N+S)
        self.canvas=CustomCanvas(self)
        self.canvas.pack(expand=1,fill=BOTH)
        self.reproduccion_toolbar=Frame(self)
        global animator
        self.animframe=animator
        self.playing=False
        self.canvas.topaint.append(self.redraw)
        self.playgif=PhotoImage(file="play.gif")
        self.stopgif=PhotoImage(file="stop.gif")
        self.pausegif=PhotoImage(file="pause.gif")
        self.bind("<Configure>",self.setHeightToolbar)
        self.play_pause = Button(self.reproduccion_toolbar, command = self.play,image=self.playgif,width="20",height="20")
        self.stop_button = Button(self.reproduccion_toolbar, command = self.stop,image=self.stopgif,width="20",height="20")
        self.reproduccion_toolbar.pack(side=BOTTOM,fill="x")
        self.progress = ttk.Progressbar(self.reproduccion_toolbar, maximum=100)
##        self.progress.bind("<Button-1>", setFrame)
##        self.progress.bind("<Motion>", motionProgress)
##        self.progress.bind("<Leave>", outProgress)
        self.stop_button.pack(side="left")
        self.play_pause.pack(side="left")
        self.progress.pack(side="left",expand=1, fill=BOTH)
    def redraw(self):
        if self.playing:
            if self.animframe.timepos<len(self.animframe.interpoled):
                self.dibujarPose(self.animframe.interpoled[self.animframe.timepos])
            else:
                self.animframe.timepos=0
                self.pause()
    def setHeightToolbar(self,evt):
        self.canvas.config(height=evt.widget.canvas.winfo_height()-26)
        self.reproduccion_toolbar.config(height=26)
    def advanceTime(self,dt):
        if not self.playing:
            return
        self.globaltime+=dt
        n=int(self.globaltime/(1/self.animframe.fps))
        if n>0:
            self.globaltime-=n/self.animframe.fps
            self.animframe.timepos+=n
            self.animframe.canvas.repaint(0)
            self.canvas.repaint(0)
    def play(self):
        self.playing=True
        self.play_pause.config(image=self.pausegif,width="20",height="20",command = self.pause)
    def pause(self):
        self.playing=False
        self.play_pause.config(image=self.playgif,width="20",height="20",command = self.play)
    def stop(self):
        self.pause()
        self.animframe.timepos=0
        self.progress["value"] = 0
        self.animframe.canvas.repaint(0)
        self.canvas.repaint(0)
    def dibujarPose(self,pose,color="black",num=9):
        if len(pose)==0:
            return
        todraw=[]
        ojos=pose[0]
        for i in range(1,len(pose)):
            todraw+=pose[i]
        c=Centroid(todraw)
        p=[self.canvas.winfo_width()/2,self.canvas.winfo_height()/2]
        todraw=TranslateTo(todraw,[p[0],p[1]])
        head=getCircle(self.rad,num*2)
        head=TranslateTo(head,todraw[0])
        self.dibujarOjos(todraw[0],ojos[0],color)
        separados=[]
        stroke=[]
        for i in range(1,len(todraw)):
            stroke.append(todraw[i])
            if(len(stroke)==5):
                separados.append(stroke)
                stroke=[]
        self.drawAllNormal(separados,color,num)
        self.drawPoints(head,color)
    def drawAllNormal(self,All,color,num=20):
        for a in All:
            self.drawPoints(toSpline(a,num),color)
    def drawPoints(self,todraw,color):
        global primera 
        primera=[]
        for a in todraw:
            drawPluma(self.canvas,a,color)
    def dibujarOjos(self,head,ojos,color):
        diametro=self.rad*2
        radOjos=diametro/12
        fullsep=diametro/3
        c=head
        sep=fullsep*(1-abs(ojos[0]))
        x=c[0]+ojos[0]*self.rad
        y=c[1]+ojos[1]*self.rad
        r=radOjos
        self.canvas.create_oval(x+sep/2-r, y-r, x+sep/2+r, y+r,outline=color, width=3)
        self.canvas.create_oval(x-sep/2-r, y-r, x-sep/2+r, y+r,outline=color, width=3)
    
        
class AvatarFrame(CustomFrame):
    canvas=0
    points=[]
    rendered=[]
    splined=[]
    indexes=[]
    between=[0,11,7,5,3,1,0]
    stack=[]
    interpoled=[]
    def __init__(self,master):
        self.rad=10
        CustomFrame.__init__(self,master,"área del avatar")
        self.grid(column=2,row=1,sticky=W+E+N+S)
        self.canvas=CustomCanvas(self)
        self.canvas.pack(expand=1,fill=BOTH)
        self.canvas.bind("<Button-1>",self.click)
        self.canvas.bind("<Button-3>", self.release)
        self.canvas.topaint.append(self.redraw)
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
        v1=self.stack[init]
        v2=self.stack[last]
        inter=[np.linspace(i,j,n) for i,j in zip(v1,v2)]
        inter=np.array(inter).T
        if init!=0:
            inter=inter[1:]
        return list(inter)+self.getInterpolation(last)
    def interpolateStack(self):
        interp=self.getInterpolation()
        self.interpoled=[]
        for v in interp:
            self.interpoled.append(v)
        
    def curveSpline(self,num=2):
        interp=[]
        for p in self.stack:
            if(len(p)!=0):
                interp.append(p)
        if len(interp)<=2:
            return
        n=len(interp)
        dim=len(interp[0])
        newint=[interp[0]]
        for i in range(1,len(interp)):
            point=[]
            for j in range(dim):
                point.append((interp[i][j]+interp[i-1][j])/2)
            newint.append(point)
            newint.append(interp[i])
        interp=newint
        K=3
        
        flen=((n-1)*num)+n
        interp_array=[]
        for i in range(dim):
            points=np.array(interp)
            x=bsplineAnt(points[:,i],K,flen)
            interp_array.append(x)
        interp_array=np.array(interp_array).T
        interp_array=list(interp_array)
        init=0
        end=-1
        n=0
        resampled=[]
        bet=0
        for i in range(1,len(self.stack)):
            if len(self.stack[i])==0:
                n+=1
            else:
                resampled+=[interp_array[(num+1)*bet]]
                end=i
                if(n>1):
                    resampled+=Resample(interp_array[1+((num+1)*bet):(num+1)*(bet+1)],n+2)[1:-1]
                else:
                    if(n==1):
                        resampled+=[interp_array[int((1+((num+1)*bet)+(num+1)*(bet+1))/2)]]
##                resampled+=[interp_array[(num+1)*(bet+1)]]
                bet+=1
                init=i                
                n=0
        resampled+=[interp_array[-1]]
        print len(resampled)
        self.interpoled=[]
        for v in resampled:
            self.interpoled.append(v)
        self.canvas.repaint(0)
    def curves(self,num=1):
        if(len(self.points)<3):
            self.canvas.repaint(0)
            return
        dim=len(self.points[0])
        K=3

     
        exp=5
        n=num+1
        rendered1=[]

        for i in range(1,len(self.points)):
            for k in range(n):
                p=[]
                for j in range(dim):
                    vI=self.points[i-1][j]
                    vF=self.points[i][j]
                    vD=(vF-vI)/2
                    mul=float(k)/(n)
                    mul=2*(0.5-mul)
                    s=1
                    if(mul!=0):
                        s=abs(mul)/mul
                    p.append(vI+vD-((vD*mul)**exp)/(vD*s)**(exp-1))
                rendered1.append(p)
        rendered1.append(self.points[-1][:])
        self.rendered=rendered1

        
##        for i in range(1,len(self.points)):
##            for k in range(n):
##                p=[]
##                for j in range(dim):
##                    vI=self.points[i-1][j]
##                    vF=self.points[i][j]
##                    vD=(vF-vI)/2
##                    mul=float(k)/(n)
##                    mul=2*(0.5-mul)
##                    
##                    s=1
##                    if(mul!=0):
##                        s=abs(mul)/mul
##                    if mul==0:
##                        mul=0.01
##                    p.append(vI+vD-vD*s*abs(mul)**(1.0/exp))
####                    p.append(vI+vD-vD*s*abs(mul)**0.5)
##                rendered1.append(p)
##        rendered1.append(self.points[-1][:])
##        self.rendered=rendered1

        n=len(self.points)
        flen=((n-1)*num)+n
        rendered2=[]
        for i in range(dim):
##            points=np.array(self.points)
##            x=bsplineAnt(points[:,i],K,flen)
            points=np.array(self.rendered)
            x=bsplineAnt(points[:,i],K,len(self.rendered)*10)
            rendered2.append(x)
        rendered2=np.array(rendered2).T
        self.splined=rendered2






 

##        self.splined=[]
##        for i in range(len(rendered1)):
####            n=i/(num+1)
####            im=(num+1)*(n+0.5)
####            mul1=2*abs(i-im)/(num+1)
####            if mul1==0:
####                mul1=0.01
####            mul2=1.0-mul1
##            mul2=0.5**(exp/5)
##            mul1=1.0-mul2
##            p=[]
##            for j in range(dim):
##                p.append(rendered1[i][j]*0.75+rendered2[i][j]*0.25)
##            self.splined.append(p)
##        self.splined=Resample(self.splined,len(self.splined))
##   

##        interp=[]
##        n=len(self.points)
##        flen=((n-1)*num)+n
##        interp_array=BezierSpline(self.points,K,flen)
##        distances=[]
##        for p in self.points:
##            np=[99999]
##            distances.append(np)
##        self.indexes=[-1]*len(self.points)
##        for i in range(len(interp_array)):
##            for j in range(len(self.points)):
##                p=interp_array[i]
##                d=Distance(p,self.points[j])
##                if d<distances[j]:
##                    distances[j]=d
##                    self.indexes[j]=i
##                    
##    
##        init=0
##        end=-1
##        n=0
##        resampled=[interp_array[0]]
##        bet=0
##        for i in range(1,len(self.points)):
##            if len(self.points[i])==0:
##                n+=1
##            else:
##                end=i
##                if(n>1):
##                    resampled+=Resample(interp_array[1+((num+1)*bet):(num+1)*(bet+1)],n)
##                else:
##                    if(n==1):
##                        resampled+=[interp_array[int((1+((num+1)*bet)+(num+1)*(bet+1))/2)]]
##                resampled+=[interp_array[(num+1)*(bet+1)]]
##                bet+=1
##                init=i                
##                n=0
##        self.rendered=interp_array
        self.canvas.repaint(0)
    def redraw(self):
        r=self.rad
        for p in self.stack:
            if len(p)!=0:
                self.canvas.create_oval(p[0]-r/2,p[1]-r/2,p[0]+r/2,p[1]+r/2,fill="red")
        r=self.rad*0.5
        n=5+1
##        for i in range(len(self.rendered)):
##            p=self.rendered[i]
##            self.canvas.create_oval(p[0]-r/2,p[1]-r/2,p[0]+r/2,p[1]+r/2,fill="blue")
##        r=self.rad*0.75
##        for i in range(len(self.splined)):
##            p=self.splined[i]
##            self.canvas.create_oval(p[0]-r/2,p[1]-r/2,p[0]+r/2,p[1]+r/2,fill="yellow")

        for i in range(len(self.interpoled)):
            p=self.interpoled[i]
            self.canvas.create_oval(p[0]-r/2,p[1]-r/2,p[0]+r/2,p[1]+r/2,fill="yellow")

    def click(self,evt):
##        self.points.append([evt.x,evt.y])
        
        n=5
        if len(self.between)>0:
            n=self.between.pop(0)
        self.stack=self.stack+([[]]*n)
        self.stack.append([evt.x,evt.y])
##        self.interpolateStack()
        self.curveSpline()
##        self.curves()
    def release(self,evt):
        self.stack=self.stack[:-1]
##        self.interpolateStack()
        self.curveSpline()
##        self.curves()
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
def drawPoints(canvas,points,color):
    global primera
    primera=[]
    for p in points:
        drawPluma(canvas,p,color)
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
