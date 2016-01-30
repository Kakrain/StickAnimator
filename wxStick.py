# -*- coding: cp1252 -*-
import wx
import math
from PIL import Image
import numpy as np
import wx.lib.scrolledpanel as scrolled
import scipy.interpolate as si
import time
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
def centerWindow(win):
    dw, dh = wx.DisplaySize()
    w, h = win.GetSize()
    x = (dw - w)/2
    y = (dh - h)/2
    win.SetPosition((x, y))
def piltoimage(pil, alpha=True):
    if alpha:
        image = apply( wx.EmptyImage, pil.size)
        image.SetData( pil.convert("RGB").tobytes() )
        image.SetAlphaData(pil.convert("RGBA").tobytes()[3::4])
    else:
        image = wx.EmptyImage(pil.size[0], pil.size[1])
        new_image = pil.convert('RGB')
        data = new_image.tobytes()
        image.SetData(data)
    return image
class CustomButton(wx.StaticBitmap):
    pressed=0
    normal=0
    def __init__(self,panel,image,func,text="press me!"):
        self.normal=image
        wx.StaticBitmap.__init__(self,panel, -1,image, (0, 0), (image.GetWidth(), image.GetHeight()))
        self.Bind(wx.EVT_LEFT_DOWN,func)
        self.SetBackgroundColour((255,255,255))
        self.SetToolTip(wx.ToolTip(text))
    def addHoldingImage(self,image):
        self.pressed=image
        self.Bind(wx.EVT_ENTER_WINDOW,self.enter)
        self.Bind(wx.EVT_LEAVE_WINDOW,self.leave)
    def enter(self,evt):
        self.SetBitmap(self.pressed)
    def leave(self,evt):
        self.SetBitmap(self.normal)
        
     
class CustomPanel(scrolled.ScrolledPanel):
    bg=0
    pin=0
    widthback=0
    heightback=0
    sizepin=30
    mainwindow=0
    sizer=0
    label=0
    font=0
    pos=0
    span=0
    topaint=0
    dc=0
    title=0
    def __init__(self,mainwindow,sizer,p=(0,0),sp=(1,1),title="desconocido"):
        scrolled.ScrolledPanel.__init__(self,mainwindow.panel)#, size=(300, 250)
        self.mainwindow=mainwindow
        self.pos=p
        self.span=sp
        self.bg=wx.Bitmap("cuadros.gif")
        path = "pin.png"
        Pimage = Image.open(path)
        Pimage = Pimage.resize((self.sizepin,self.sizepin), Image.ANTIALIAS)
        self.pin= wx.BitmapFromImage(piltoimage(Pimage))
        self.pin=CustomButton(self,self.pin,self.expand,"expand")
        self.topaint=[]
        self.SetBackgroundColour(wx.WHITE)
        path = "pinO.png"
        Pimage = Image.open(path)
        Pimage = Pimage.resize((self.sizepin,self.sizepin), Image.ANTIALIAS)
        self.pin.addHoldingImage(wx.BitmapFromImage(piltoimage(Pimage)))
                
        self.sizer=wx.BoxSizer(wx.VERTICAL)
        self.title=wx.BoxSizer(wx.HORIZONTAL)
        self.label=wx.StaticText(self,label=title)
        self.label.SetBackgroundColour((255,255,255))
        
        self.title.Add(self.pin,flag=wx.ALIGN_LEFT)
        self.title.Add(self.label,flag=wx.ALIGN_LEFT)

        self.sizer.Add(self.title,flag=wx.ALIGN_TOP|wx.EXPAND,proportion=1)
        self.SetSizer(self.sizer)
        self.Layout()
        self.font=wx.Font(14, wx.DEFAULT, wx.NORMAL, wx.NORMAL, False,u'MV Boli')
        self.label.SetFont(self.font)

        self.widthback, self.heightback = self.bg.GetSize()
        sizer.Add(self, pos=p,span=sp, flag=wx.EXPAND)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
    def regrid(self):
        global main
        main.bagSizer.Add(self, pos=self.pos,span=self.span, flag=wx.EXPAND)
    def expand(self,evt):
        global main
        for p in main.bagSizer.GetChildren():
            p.GetWindow().Hide()
        main.bagSizer.Clear()
        self.Show()
        main.bagSizer.Add(self,pos=(0,0),flag=wx.EXPAND)
        s=main.GetClientSize()
        self.SetDimensions(0,0,s[0],s[1])
        self.pin.Bind(wx.EVT_LEFT_DOWN,self.shrink)
    def shrink(self,evt):
        global main
        main.bagSizer.Clear()
        for p in main.customs:
            p.Show()
            p.regrid()
        s=main.GetClientSize()
        foco=main.IsMaximized()
        if(foco):
            main.Maximize(False)
        main.SetClientSize((s[0]+1,s[1]+1))
        main.SetClientSize((s[0]-1,s[1]-1))
        if(foco):
            main.Maximize(True)
        self.pin.Bind(wx.EVT_LEFT_DOWN,self.expand)
    def OnSize(self, size):
        self.Layout()
        self.Refresh()
    def OnEraseBackground(self, evt):
        pass

    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        self.Draw(dc)

    def Draw(self, dc):
        cliWidth, cliHeight = self.GetClientSize()
        if not cliWidth or not cliHeight:
            return
        dc.Clear()
        col=int(math.ceil(float(cliWidth)/self.widthback))
        row=int(math.ceil(float(cliHeight)/self.heightback))
        for i in range(col):
            for j in range(row):
                dc.DrawBitmap(self.bg,i*self.widthback,j*self.heightback)
        for f in self.topaint:
            f(dc)       
        
    def getPosition(self):
        offset=(8,31)
        p=self.GetScreenPosition()
        ps=self.mainwindow.GetScreenPosition()
        return (p[0]-ps[0]-offset[0],p[1]-ps[1]-offset[1])
class MainWindow(wx.Frame):
    panel=0
    mainsizer=0
    bagSizer=0
    customs=0
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, 'StickAnimator')
        self.panel = wx.Panel(self, wx.ID_ANY)
        self.mainsizer=wx.BoxSizer(wx.VERTICAL)
        self.bagSizer=wx.GridBagSizer(hgap=5, vgap=5)
        self.customs=[]
        self.customs.append(AnimationFrame(self,self.bagSizer))
        self.customs.append(DrawFrame(self,self.bagSizer))
        self.customs.append(ReproductorFrame(self,self.bagSizer,self.customs[0]))
        self.customs.append(CustomPanel(self,self.bagSizer,title="área del avatar",p=(1,2)))
        self.bagSizer.AddGrowableRow(0)
        self.bagSizer.AddGrowableRow(1)
        self.bagSizer.AddGrowableCol(0)
        self.bagSizer.AddGrowableCol(1)
        self.bagSizer.AddGrowableCol(2)
        self.mainsizer.Add(self.bagSizer,flag=wx.EXPAND,proportion=1)

        self.bagSizer.Fit(self)
        self.mainsizer.Fit(self)
        self.SetSizeWH(1000,500)
        self.panel.SetSizerAndFit(self.mainsizer)
        self.panel.GetSizer().Layout()

        menubar = wx.MenuBar()
        fileMenu = wx.Menu()

        gitem = fileMenu.Append(wx.ID_ANY, 'Guardar', 'guardar la animación')
        self.Bind(wx.EVT_MENU, self.saveCallBack, gitem)

        citem = fileMenu.Append(wx.ID_ANY, 'Cargar', 'cargar una animación')
        self.Bind(wx.EVT_MENU, self.loadCallBack, citem)

        fitem = fileMenu.Append(wx.ID_ANY, 'Cerrar', 'Cerrar la aplicación')
        self.Bind(wx.EVT_MENU, self.Salir, fitem)



        anim=self.customs[0]
        editionMenu=wx.Menu()

        copitem=editionMenu.Append(wx.ID_ANY,"Copiar","copiar la pose")
        self.Bind(wx.EVT_MENU,anim.copiar,copitem)

        coritem=editionMenu.Append(wx.ID_ANY,"Cortar","cortar la pose")
        self.Bind(wx.EVT_MENU,anim.cortar,coritem)

        pegitem=editionMenu.Append(wx.ID_ANY,"Pegar","pegar la pose")
        self.Bind(wx.EVT_MENU,anim.pegar,pegitem)

        desitem=editionMenu.Append(wx.ID_ANY,"Deshacer","deshacer la última acción")
        self.Bind(wx.EVT_MENU,anim.deshacer,desitem)

        reitem=editionMenu.Append(wx.ID_ANY,"Rehacer","rehacer la última acción")
        self.Bind(wx.EVT_MENU,anim.rehacer,reitem)

        timeitem=editionMenu.Append(wx.ID_ANY,"Cambiar el tiempo máximo", "cambia el tiempo máximo del área de animación")
        self.Bind(wx.EVT_MENU,self.configTime,timeitem)
        
        self.accel_tbl = wx.AcceleratorTable([(wx.ACCEL_CTRL, ord('C'),copitem.GetId()),
                                              (wx.ACCEL_CTRL, ord('X'),coritem.GetId()),
                                              (wx.ACCEL_CTRL, ord('V'),pegitem.GetId()),
                                              (wx.ACCEL_CTRL, ord('Z'),desitem.GetId()),
                                              (wx.ACCEL_CTRL, ord('R'),reitem.GetId()),
                                             ])
        self.SetAcceleratorTable(self.accel_tbl)
 
        menubar.Append(fileMenu, '&File')
        menubar.Append(editionMenu, '&Edición')
        self.SetMenuBar(menubar)

        self.Show(True)
    def configTime(self,evt):
        anim=self.customs[0]
        s=ask(self,"escriba el tiempo máximo a usarse en el área de animación", default_value=str(anim.maxTime))
        s=''.join(c for c in s if c.isdigit() or c=='.')
        anim.setMax(float(s))
    def saveCallBack(self,evt):
        saveFileDialog = wx.FileDialog(self, "Guardar animación", "", "","ANIM files (*.anim)|*.anim", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if saveFileDialog.ShowModal() == wx.ID_CANCEL:
            return
        filename=saveFileDialog.GetPath()
        filename = open(filename, 'w')
        animator=self.customs[0]
        s=animator.toString()
        filename.write(s)
        filename.close()
    def loadCallBack(self,evt):
        if wx.MessageBox("Cargar animacion?", "confirmar por favor",wx.ICON_QUESTION | wx.YES_NO, self) == wx.NO:
            return
        openFileDialog = wx.FileDialog(self, "Cargar animación", "", "","ANIM files (*.anim)|*.anim", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return
        filename=openFileDialog.GetPath()
        filename = open(filename, 'r')
        string=filename.read()
        animator=self.customs[0]
        animator.loadAnimation(string)
        filename.close()
    def Salir(self, e):
        self.Close()
##    creditos a https://github.com/philwilliammee/wx_python_obj_viewer
class AvatarFrame(CustomPanel):
    def __init__(self,mainwindow,sizer,animframe):
        CustomPanel.__init__(self,mainwindow,sizer,title="área de reproducción",p=(1,1))
    
class ReproductorFrame(CustomPanel):
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
    start=0
    def __init__(self,mainwindow,sizer,animframe):
        CustomPanel.__init__(self,mainwindow,sizer,title="área de reproducción",p=(1,1))
        self.reproduccion_toolbar=wx.BoxSizer(wx.HORIZONTAL)
        self.stop_button=CustomButton(self,wx.Bitmap("stopL.gif"),self.stop,"detener la animación")
        self.stop_button.addHoldingImage(wx.Bitmap("stop.gif"))        
        self.play_pause=CustomButton(self,wx.Bitmap("playL.gif"),self.play,"reproducir la animación")
        self.play_pause.addHoldingImage(wx.Bitmap("play.gif"))
        self.progress = wx.Gauge(self)
        self.animframe=animframe
        self.reproduccion_toolbar.Add(self.stop_button,flag=wx.ALIGN_LEFT|wx.LEFT)
        self.reproduccion_toolbar.Add(self.play_pause,flag=wx.ALIGN_LEFT|wx.LEFT)
        self.reproduccion_toolbar.Add(self.progress,proportion=1)
        self.sizer.Add(self.reproduccion_toolbar,flag=wx.ALIGN_BOTTOM|wx.EXPAND)

        self.Layout()
        self.Refresh()
        
        self.playing=False
        self.topaint.append(self.redraw)
    def redraw(self,dc):
        if self.playing:
            if self.animframe.timepos<len(self.animframe.interpoled):
                self.dibujarPose(self.animframe.interpoled[self.animframe.timepos],dc)
            else:
                self.animframe.timepos=0
                self.pause(0)
    def startTimer(self):
        self.start=time.time()
        self.updateTime()
    def updateTime(self):
        end=time.time()
        dt=end-self.start
        self.advanceTime(dt)
        if self.playing:
            wx.CallLater(1000.0/self.animframe.fps,self.updateTime)
        self.start=end
    def advanceTime(self,dt):
        if not self.playing:
            return
        self.globaltime+=dt
        n=int(self.globaltime/(1/self.animframe.fps))
        if n>0:
            self.globaltime-=n/self.animframe.fps
            self.animframe.timepos+=n
            self.progress.SetValue(self.progress.GetValue()+n)
            self.animframe.Refresh()
            self.Refresh()
    def play(self,evt):
        self.playing=True
        self.play_pause.SetToolTip(wx.ToolTip("pausar la animación"))
        normal=wx.Bitmap("pauseL.gif")
        pressed=wx.Bitmap("pause.gif")
        self.play_pause.normal=normal
        self.play_pause.pressed=pressed
        self.play_pause.SetBitmap(self.play_pause.normal)
        self.play_pause.Bind(wx.EVT_LEFT_DOWN,self.pause)
        self.startTimer()
    def pause(self,evt):
        self.playing=False
        self.play_pause.SetToolTip(wx.ToolTip("reproducir la animación"))
        normal=wx.Bitmap("playL.gif")
        pressed=wx.Bitmap("play.gif")
        self.play_pause.normal=normal
        self.play_pause.pressed=pressed
        self.play_pause.SetBitmap(self.play_pause.normal)
        self.play_pause.Bind(wx.EVT_LEFT_DOWN,self.play)
    def stop(self,evt):
        self.pause(0)
        self.animframe.timepos=0
        self.progress.SetValue(0)    
        self.animframe.Refresh()
        self.Refresh()
    def dibujarPose(self,pose,dc,num=9):
        if len(pose)==0:
            return
        todraw=[]
        ojos=pose[0]
        for i in range(1,len(pose)):
            todraw+=pose[i]
        c=Centroid(todraw)
        size=self.GetSize()
        p=[size[0]/2,size[1]/2]
        todraw=TranslateTo(todraw,[p[0],p[1]])
        head=getCircle(self.rad,num)
        head=TranslateTo(head,todraw[0])
        self.dibujarOjos(todraw[0],ojos[0],dc)
        separados=[]
        stroke=[]
        for i in range(1,len(todraw)):
            stroke.append(todraw[i])
            if(len(stroke)==5):
                separados.append(stroke)
                stroke=[]
        drawAllNormal(self,separados,dc)
        drawPoints(self,head,dc)
    def dibujarOjos(self,head,ojos,dc):
        diametro=self.rad*2
        radOjos=diametro/12
        fullsep=diametro/3
        c=head
        sep=fullsep*(1-abs(ojos[0]))
        x=c[0]+ojos[0]*self.rad
        y=c[1]+ojos[1]*self.rad
        r=radOjos
        dc.DrawCircle(x+sep,y, r)
        dc.DrawCircle(x-sep,y, r)
    
class AnimationFrame(CustomPanel):
    h_off=60
    pointer=0
    timepos=0
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
    def __init__(self,mainwindow,sizer):
        CustomPanel.__init__(self,mainwindow,sizer,title="área de animación",p=(0,0),sp=(1,3))
        self.spc=5.0/1000#segundos por cada pixel de canvas
        self.maxTime=10.0#en segundos
        self.Bind(wx.EVT_SCROLLWIN,self.scroll)
        self.SetupScrolling(scroll_y=False)
        self.line=wx.StaticLine(self, -1, size=(10, -1))
        self.sizer.Add(self.line, 0, 0, 5)
        self.stack=[]
        self.interpoled=[]
        self.rendered=[]       
        self.topaint.append(self.redraw)
        
        self.Bind(wx.EVT_LEFT_DOWN,self.clickedTime)
        self.Bind(wx.EVT_MOTION,self.moveTime)
        self.Bind(wx.EVT_LEFT_UP,self.releaseTime)
        self.Bind(wx.EVT_LEFT_DCLICK,self.doubleTime)
        self.Bind(wx.EVT_KEY_DOWN,self.deletePose)

        self.setLine(self.maxTime/self.spc)
        self.timepos=0
        self.pointer=0
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
    def deletePose(self,evt):
        if wx.WXK_DELETE!=evt.GetKeyCode():
            return
        if(self.timepos!=0 and self.timepos<len(self.stack) and len(self.stack[self.timepos])!=0):
            if wx.MessageDialog(self, "Seguro que quieres eliminar esta pose?","Eliminar pose", wx.YES_NO | wx.ICON_QUESTION).ShowModal() == wx.ID_YES:
                self.stack[self.timepos]=[]
        self.do()
        self.generatePoses()    
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
    def doubleTime(self,evt):
        i=self.getStackPos(self.recalculateCanvas(evt.x))
        if i!=0 and i<len(self.stack) and len(self.stack[i])!=0:
            global main
            draw=main.customs[1]
            draw.initEditPose(i)
            self.poseE=i
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
    def do(self):
        self.redolist=[]
        self.undolist.append(self.stack[:])
        if(len(self.undolist)>self.maxdolist):
            self.undolist=self.undolist[1:]
    def toString(self):
        s=""
        s+=str(self.maxTime)
        s+="@"
        for pose in self.stack:
            s+=self.poseToString(pose)
            s+="#"
        s=s[:-1]
        return s
    
    def interpolateStack(self):
        if len(self.stack)<=1:
            self.interpoled=self.stack[:]
            return
        interp=self.getInterpolation()
        self.interpoled=[]
        for v in interp:
            self.interpoled.append(self.separarPose(self.unvectorizePose(v)))
    def scalePose(self,pose):
        escalado=[]
        escalado.append(pose[0])
        for i in range(1,len(pose)):
            escalado.append([])
            for p in pose[i]:
                escalado[-1].append([p[0]*self.rad,p[1]*self.rad])
        return escalado
    def setMax(self,maxt):
        maxtimestack=len(self.stack)/self.fps
        self.maxTime=max(maxt,maxtimestack)
        self.setLine(self.maxTime/self.spc)
        self.Refresh()
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
        self.Refresh()
    def endEditPose(self):
        self.generatePoses()
        self.poseE=-1
    def generatePoses(self):
        self.curveSpline()
        self.Refresh()
        global main
        main.customs[2].progress.SetRange(len(self.interpoled))
    def curveSpline(self,num=20):
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
                bet+=1
                init=i                
                n=0
        resampled+=[interp_array[-1]]
        self.interpoled=[]
        for v in resampled:
            self.interpoled.append(self.separarPose(self.unvectorizePose(v)))
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
        
    def setLine(self,length):
        self.line.Destroy()
        length+=30
        self.line=wx.StaticLine(self, -1, size=(length, -1))
        self.sizer.Add(self.line, flag=wx.ALIGN_BOTTOM)
        self.SetupScrolling(scroll_y=False)
    def clickedTime(self,evt):
        self.poseS=-1
        if(len(self.stack)==0):
            return
        i=self.getStackPos(self.recalculateCanvas(evt.x))
        self.timepos=i
        if(i<len(self.stack) and i!=0):
            self.poseS=i
        self.Refresh()
    def moveTime(self,evt):
        i=self.getStackPos(self.recalculateCanvas(evt.x))
        self.pointer=i
        self.Refresh()
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
    def scroll(self,evt):
        self.Refresh()
    def redraw(self,dc):
        global pen
        pen.SetStyle(wx.SOLID)
        pen.SetWidth(10)
        pen.SetColour("yellow")
        dc.SetPen(pen)
        
        self.drawAllTimeMarks(dc)
        self.drawTimeline(dc)
        setLapizClaro(dc) 
        self.drawInterpoled(dc)
        self.drawStack(dc)
        if self.poseS!=-1:
            setPluma(dc,"red")
            self.dibujarPose(self.pointer,self.stack[self.poseS],dc)
        self.drawPointer(dc)
        self.drawTimePos(dc)
    def drawStack(self,dc):
        size=self.GetSize()
        recorrido=self.CalcUnscrolledPosition((0,0))[0]
        t=recorrido*self.spc
        tf=t+(size[0]*self.spc)
        f0=min(len(self.interpoled),int(t*self.fps))
        ff=min(len(self.interpoled),int(tf*self.fps))
        for i in range(f0,ff):
            if i==self.pointer:
                setPluma(dc,"blue")
                self.dibujarPose(i,self.stack[i],dc)
            else:
                setPluma(dc,"black")
                self.dibujarPose(i,self.stack[i],dc)
        
    def drawInterpoled(self,dc):
        i=self.pointer
        if i>=0 and i<len(self.interpoled):
            self.dibujarPose(i,self.interpoled[i],dc)
    def dibujarPose(self,index,pose,dc,num=9):
        if len(pose)==0:
            return
        todraw=[]
        ojos=pose[0]
        for i in range(1,len(pose)):
            todraw+=pose[i]
        c=Centroid(todraw)
        size=self.GetSize()
        p=[self.getStackCanvas(index),size[1]/2]
        todraw=TranslateTo(todraw,[p[0],p[1]])
        head=getCircle(self.rad,num)
        head=TranslateTo(head,todraw[0])
        self.dibujarOjos(todraw[0],ojos[0],dc)
        separados=[]
        stroke=[]
        for i in range(1,len(todraw)):
            stroke.append(todraw[i])
            if(len(stroke)==5):
                separados.append(stroke)
                stroke=[]
        drawAllNormal(self,separados,dc)
        drawPoints(self,head,dc)
    def dibujarOjos(self,head,ojos,dc):
        diametro=self.rad*2
        radOjos=diametro/12
        fullsep=diametro/3
        c=head
        sep=fullsep*(1-abs(ojos[0]))
        x=c[0]+ojos[0]*self.rad
        y=c[1]+ojos[1]*self.rad
        r=radOjos
        dc.DrawCircle(x+sep,y, r)
        dc.DrawCircle(x-sep,y, r)
    def recalculateCanvas(self,x):
        recorrido=self.CalcUnscrolledPosition((0,0))[0]
        return recorrido+x
    def getStackSeconds(self,i):
        return float(i)/self.fps
    def drawTimePos(self,dc):
        setPluma(dc,"black")
        x=self.getStackCanvas(self.timepos)
        y=self.GetSize()[1]
        dc.DrawLine(x,0,x,y)
        dc.DrawLabel("{0:.2f}".format(self.getStackSeconds(self.timepos))+"s",wx.Rect(x,y-30,10,10))
    def drawPointer(self,dc):
        setLapiz(dc)
        x=self.getStackCanvas(self.pointer)
        y=self.GetSize()[1]
        dc.DrawLine(x,0,x,y)
        dc.DrawLabel("{0:.2f}".format(self.getStackSeconds(self.pointer))+"s",wx.Rect(x,y-30,10,10))
    def getStackCanvas(self,i):
        return i*((1/self.fps)/self.spc)-self.CalcUnscrolledPosition((0,0))[0]
    def getStackPos(self,x):
        return int(x/((1/self.fps)/self.spc))
    def drawTimeline(self,dc):
        wid=self.maxTime/self.spc
        he=self.GetSize()[1]
        pI=(0,he-self.h_off)
        pF=(self.GetSize()[0],he-self.h_off)
        dc.DrawLine(pI[0],pI[1],pF[0],pF[1])
        pen.SetColour("red")
        dc.SetPen(pen)
        recorrido=self.CalcUnscrolledPosition((0,0))[0]
        pI=(wid-0+14-recorrido,he-self.h_off)
        pF=(self.GetVirtualSize()[0]-recorrido+14,he-self.h_off)
        dc.DrawLine(pI[0],pI[1],pF[0],pF[1])
    def drawAllTimeMarks(self,dc):
        recorrido=self.CalcUnscrolledPosition((0,0))[0]
        t=recorrido*self.spc
        tf=t+(self.GetSize()[0]*self.spc)
        for i in range(int(t),int(tf)+1):
            self.drawTimeMark(float(i),dc)
    def drawTimeMark(self,t,dc):
        recorrido=self.CalcUnscrolledPosition((0,0))[0]
        x=t/self.spc
        x-=recorrido
        he=self.GetSize()[1]
        pI=(x,he-self.h_off)
        pF=(x,he-self.h_off/2)
        dc.DrawLine(pI[0],pI[1],pF[0],pF[1])
        dc.DrawLabel("{0:.2f}".format(t)+"s",wx.Rect(x,he-self.h_off/2,10,10))      
    
class DrawFrame(CustomPanel):
    strokes=0
    currentpoints=0
    head=0
    lastsize=[]
    finished=False
    controlPoints=[]
    selectedCP=0
    proportions=0
    clicked=False
    finishedsizer=0
    ant=[]
    def __init__(self,mainwindow,sizer):
        CustomPanel.__init__(self,mainwindow,sizer,title="área de dibujo",p=(1,0))
        self.currentpoints=[]
        self.strokes=[]
        self.head=getCircle(0.09,250)
        self.proportions=[0.58,0.67,0.67,0.89,0.89]
        s=self.GetClientSize()
        self.head=TranslateTo(self.head,[s[0]*0.5,s[1]*0.25])
        self.lastsize=[float(s[0]),float(s[1])]
        self.topaint.append(self.redraw)

        self.Bind(wx.EVT_LEFT_DOWN, self.clickedDraw)
        self.Bind(wx.EVT_MOTION, self.paintDraw)
        self.Bind(wx.EVT_LEFT_UP, self.releaseDraw)
        self.Bind(wx.EVT_RIGHT_UP, self.clicked3Draw)
        self.Bind(wx.EVT_LEAVE_WINDOW,self.leave)

        self.addFinishedPanel()
        self.hideFinishedPanel()
        
        self.SetCursor(wx.StockCursor(wx.CURSOR_PENCIL))
        self.label.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))
        self.pin.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))
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
        size=self.GetSize()
        cm=[size[0]/2,size[1]/2]
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
        global main
        animator=main.customs[0]
        self.newPose()
        ojos=animator.stack[i][0][0]
        headp=animator.stack[i][1][0]

        diametro=Distance(self.head[0],self.head[int(len(self.head)/2)])
        r=diametro/2
        
        self.head=TranslateTo(self.head,headp)
        self.strokes=animator.stack[i][2:][:]
        self.centrarTodo()
        self.hechoAction()
        self.showFinishedPanel(True)
        self.controlPoints[1].locations[0]=ojos[0]
        self.controlPoints[1].locations[1]=ojos[1]
        self.Refresh()
    
    def leave(self,evt):
        self.clicked=False
        self.selectedCP=0
    def hideFinishedPanel(self):
        self.sizer.Hide(self.finishedsizer,recursive=True)
    def showFinishedPanel(self,editing=False):
        self.sizer.Show(self.finishedsizer,recursive=True)
        if not editing:
            self.finishedsizer.GetChildren()[0].GetWindow().Hide()
        self.Layout()
    def addFinishedPanel(self):        
        self.finishedsizer=wx.BoxSizer(wx.VERTICAL)
        doneb=CustomButton(self,wx.Bitmap("doneL.gif"),self.donePose,"terminar de editar")
        doneb.addHoldingImage(wx.Bitmap("done.gif"))
        self.finishedsizer.Add(doneb,flag=wx.ALIGN_RIGHT)
        insb=CustomButton(self,wx.Bitmap("insertL.gif"),self.insertPose,"insertar como pose principal")
        insb.addHoldingImage(wx.Bitmap("insert.gif"))
        self.finishedsizer.Add(insb,flag=wx.ALIGN_RIGHT)
        newb=CustomButton(self,wx.Bitmap("newL.gif"),self.newPose,"nuevo sketch")
        newb.addHoldingImage(wx.Bitmap("new.gif"))
        self.finishedsizer.Add(newb,flag=wx.ALIGN_RIGHT)
        self.title.Add(self.finishedsizer,proportion=1,flag=wx.ALIGN_RIGHT)
        self.title.Layout()
    def donePose(self,evt):
        self.newPose()
        self.head=self.ant[0][:]
        self.strokes=self.ant[1][:]
        self.lastsize=self.ant[2][:]
        self.ant=[]
        global main
        animator=main.customs[0]
        animator.endEditPose()
        self.hideFinishedPanel()
        if len(self.strokes)==5:
            self.hechoAction()
        self.Refresh()
    def getPose(self):
        pose=[]
        pose.append(np.array([Centroid(self.head)]))
        for s in self.strokes:
            pose.append(s[:])
        return pose
    
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
    def unir(self,strokes):
        unidos=[]
        for s in strokes:
            unidos+=list(s)
        return unidos
    def separar(self,pose):
        separados=[]
        separados.append([pose[0]])
        stroke=[]
        for i in range(1,len(pose)):
            stroke.append(pose[i])
            if(len(stroke)==5):
                separados.append(stroke)
                stroke=[]
        return separados      
   
    def insertPose(self,evt):
        global main
        animator=main.customs[0]
        p=self.standarPose()
        animator.agregarPose(p)
    def newPose(self):
        s=self.GetClientSize()
        self.lastsize=[s[0],s[1]]
        self.head=TranslateTo(self.head,[s[0]*0.5,s[1]*0.25])
        self.Bind(wx.EVT_LEFT_DOWN, self.clickedDraw)
        self.Bind(wx.EVT_MOTION, self.paintDraw)
        self.Bind(wx.EVT_LEFT_UP, self.releaseDraw)
        self.Bind(wx.EVT_RIGHT_UP, self.clicked3Draw)
        self.currentpoints=[]
        self.strokes=[]
        self.SetCursor(wx.StockCursor(wx.CURSOR_PENCIL))
        self.label.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))
        self.pin.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))
        self.controlPoints=[controlHead(self)]
        self.hideFinishedPanel()
        self.finished=False
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
    def getMaxLength(self,i):
        diametro=Distance(self.head[0],self.head[int(len(self.head)/2)])
        length=diametro*2.4
        return self.proportions[i]*length
    def dibujarOjos(self,dc):
        diametro=Distance(self.head[0],self.head[int(len(self.head)/2)])
        rad=diametro/2
        radOjos=diametro/12
        fullsep=diametro/6
        c=Centroid(self.head)
        sep=fullsep*(1-abs(self.controlPoints[1].locations[0]))
        x=c[0]+self.controlPoints[1].locations[0]*rad
        y=c[1]+self.controlPoints[1].locations[1]*rad
        r=1.3*radOjos
        dc.DrawCircle(x+sep,y, r)
        dc.DrawCircle(x-sep,y, r)
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
        
    def paintDraw(self,event):
        if not self.clicked:
            return
        if self.selectedCP!=0:
            self.selectedCP.moveTo(event.x,event.y)
            self.Refresh()
            return
        self.currentpoints.append([event.x,event.y])
        self.Refresh()
    def clickedDraw(self,event):
        global primera
        primera=[]
        self.currentpoints=[]
        self.clicked=True
        for cp in self.controlPoints:
            if(cp.isSelected(event.x,event.y)):
                self.selectedCP=cp
                return
        self.currentpoints.append([event.x,event.y])
        
    def releaseDraw(self,event):
        self.clicked=False
        if self.selectedCP!=0:
            self.selectedCP=0
            return   
        self.currentpoints.append([event.x,event.y])
        self.strokes.append(toStick(self.completePoints(self.currentpoints)))
        if(len(self.currentpoints)<10 or len(self.strokes[len(self.strokes)-1])<=1):
            self.clicked3Draw(0)
        self.currentpoints=[]
        if(len(self.strokes)==5):
            self.hechoAction()
        self.limitAll()
        self.Refresh()
    def clicked3Draw(self,event ):
        if len(self.strokes)>0:
            popped=self.strokes.pop()
            self.Refresh()
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
            global main
            animator=main.customs[0]
            p=self.standarPose()
            animator.refreshPose(p)
        self.Refresh()
    def selectRelease(self,evt):
        self.selectedCP=0  
    def hechoAction(self):
        self.Bind(wx.EVT_LEFT_DOWN, self.selectClick)
        self.Bind(wx.EVT_MOTION, self.selectMove)
        self.Bind(wx.EVT_LEFT_UP, self.selectRelease)
        
        self.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))
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
        
        self.showFinishedPanel()
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
    def redraw(self,dc):
        setLapiz(dc)
        drawPoints(self,self.currentpoints,dc)
        self.head=self.normalize(self.head)
        for i in range(len(self.strokes)):
            self.strokes[i]=self.normalize(self.strokes[i])
        self.normalizePositionAll()
        setPluma(dc)
        drawAllNormal(self,self.strokes,dc)
        drawPoints(self,self.head,dc)
        s=self.GetClientSize()
        self.lastsize=[float(s[0]),float(s[1])]
        if self.finished:
            self.dibujarOjos(dc)
        setFill(dc,"red")
        for cp in self.controlPoints:
                cp.drawCP(dc)
        
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
        s=self.GetClientSize()
        newpoints=ScaleToY(newpoints,(maxy-miny)*s[1]/self.lastsize[1])
        newpoints=TranslateTo(newpoints,[c[0]*s[0]/self.lastsize[0],c[1]*s[1]/self.lastsize[1]])
        return newpoints
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
    def limitAll(self):
        for i in range(len(self.strokes)):
            for j in range(len(self.strokes[i])):
                self.limitPoint(i,j)
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
def drawAllNormal(canvas,All,dc,num=20):
        for a in All:
            drawPoints(canvas,toSpline(a,num),dc)
def drawPoints(canvas,todraw,dc):
    global primera
    primera=[]
    for a in todraw:
        draw(canvas,a,dc)    
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
def Distance(p1, p2):
    dim=len(p1)
    res=0
    for i in range(dim):
        res+=(p2[i]-p1[i])**2
    return math.sqrt(res)    
def toSpline(points,num=20):
    return BezierSpline(points,2,num)
def toStick(points):
    return BezierSpline(points,2,5)
def setLapizClaro(dc):
    global pen
    pen.SetStyle(wx.SOLID)
    pen.SetWidth(3)
    pen.SetColour(wx.Colour(200,200,200))
    dc.SetPen(pen)
def setLapiz(dc):
    global pen
    pen.SetStyle(wx.SOLID)
    pen.SetWidth(2.5)
    pen.SetColour(wx.Colour(100,100,100))
    dc.SetPen(pen)
def setPluma(dc,color=wx.BLACK):
    global pen
    pen.SetStyle(wx.SOLID)
    pen.SetWidth(1)
    pen.SetColour(color)
    dc.SetPen(pen)
def setFill(dc,color="red"):
    global pen
    pen.SetStyle(wx.PENSTYLE_TRANSPARENT)
    pen.SetWidth(1)
    dc.SetBrush(wx.Brush('red'))
    dc.SetPen(pen)
def ask(parent,message, default_value=''):
    dlg = wx.TextEntryDialog(parent, message, defaultValue=default_value)
    dlg.ShowModal()
    result = dlg.GetValue()
    dlg.Destroy()
    return result
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
     
def draw(canvas,pt,dc):
    global primera
    if len(primera)==0:
        primera=pt
    else:
        dc.DrawLine(primera[0],primera[1],pt[0],pt[1])
        primera=pt
def getCircle(r,n):
    n=n-1
    newpoints=[]
    for i in range(0,n):
        newpoints.append([r*math.cos(2*math.pi*i/n),r*math.sin(2*math.pi*i/n)])
    newpoints.append([r,0])
    return newpoints
def ScaleToY(pts, size):
    B=BoundingBox(pts)
    newpoints=[]
    return pts*[size/B[3],size/B[3]]
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
def TranslateTo(pts,p):
   c=Centroid(pts)
   return np.subtract(np.add(pts,[p]*len(pts)),[c]*len(pts))
def Centroid(pts):
   x = 0
   y = 0
   for a in pts:
      x += a[0]
      y += a[1]
   x /= len(pts)
   y /= len(pts)
   return [x,y]

class controlPoint(object):
    rad=7
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
            self.moveBackwards(i,j)
            self.dragForward(i,j,offset)
        self.draw.limitAllNF()
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
    def drawCP(self,dc):
        p=self.draw.strokes[self.locations[0][0]][self.locations[0][1]]
        dc.DrawCircle(p[0],p[1], self.rad)
class controlHead(controlPoint):
    def getPosition(self):
        return self.draw.head[int(len(self.draw.head)*0.75)]
    def isSelected(self,x,y):
        return Distance([x,y],self.getPosition())<=self.rad
    def moveTo(self,x,y):
        p=self.getPosition()
        c=Centroid(self.draw.head)
        self.draw.head=TranslateTo(self.draw.head,[x-(p[0]-c[0]),y-(p[1]-c[1])])
    def drawCP(self,dc):
        p=self.getPosition()
        dc.DrawCircle(p[0],p[1], self.rad)
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
    def drawCP(self,dc):
        p=self.getPosition()
        dc.DrawCircle(p[0],p[1], self.rad)
global main
global primera
global pen

primera=[]
app = wx.App(0)
pen=wx.Pen(wx.BLACK,2.5)
main = MainWindow()
centerWindow(main)
app.MainLoop()
