import wx
import sys,math
from PIL import Image
def piltoimage(pil, alpha=True):
    """Convert PIL Image to wx.Image."""
    if alpha:
        image = apply( wx.EmptyImage, pil.size )
        image.SetData( pil.convert( "RGB").tobytes() )
        image.SetAlphaData(pil.convert("RGBA").tobytes()[3::4])
    else:
        image = wx.EmptyImage(pil.size[0], pil.size[1])
        new_image = pil.convert('RGB')
        data = new_image.tobytes()
        image.SetData(data)
    return image
class CustomPanel(wx.Panel):
    bg=0
    pin=0
    widthback=0
    heightback=0
    mainwindow=0
    def __init__(self,mainwindow,sizer,p=(0,0),sp=(1,1),title="desconocido"):
        wx.Panel.__init__(self,mainwindow.panel, size=(200, 100))
        self.mainwindow=mainwindow
        self.bg=wx.Bitmap("cuadros.gif")
        path = "pin.png"
        Pimage = Image.open(path)
        Pimage = Pimage.resize((40,40), Image.ANTIALIAS)
        self.pin= wx.BitmapFromImage(piltoimage(Pimage))
        
        self.widthback, self.heightback = self.bg.GetSize()
        sizer.Add(self, pos=p,span=sp, flag=wx.EXPAND)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)

    def OnEraseBackground(self, evt):
        
        # yanked from ColourDB.py
        dc = evt.GetDC()
        if not dc:
            dc = wx.ClientDC(self)
            rect = self.GetUpdateRegion().GetBox()
            dc.SetClippingRect(rect)
        dc.Clear()
        cliWidth, cliHeight = self.GetClientSize()
        col=int(math.ceil(float(cliWidth)/self.widthback))
        row=int(math.ceil(float(cliHeight)/self.heightback))
        for i in range(col):
            for j in range(row):
                dc.DrawBitmap(self.bg,i*self.widthback,j*self.heightback)
        dc.DrawBitmap(self.pin, 0, 0)
        print float(cliWidth)/2
    def getPosition(self):
        offset=(8,31)
        p=self.GetScreenPosition()
        ps=self.mainwindow.GetScreenPosition()
        return (p[0]-ps[0]-offset[0],p[1]-ps[1]-offset[1])
class MainWindow(wx.Frame):
    panel=0
    mainsizer=0
    topsizer=0
    botsizer=0
    bagSizer=0
    
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, 'StickAnimator',size=(1000, 500))
        self.panel = wx.Panel(self, wx.ID_ANY)
        self.sizer = wx.GridBagSizer(4, 4)
        self.mainsizer    = wx.BoxSizer(wx.VERTICAL)
        self.bagSizer    = wx.GridBagSizer(hgap=5, vgap=5) 
        
        self.topsizer    = wx.BoxSizer(wx.HORIZONTAL)
        self.botsizer  = wx.BoxSizer(wx.HORIZONTAL)
        CustomPanel(self,self.bagSizer,title="area de animacion",p=(0,0),sp=(1,3))
        
        
        CustomPanel(self,self.bagSizer,title="area de diseño",p=(1,0))
        CustomPanel(self,self.bagSizer,title="area de reproduccion",p=(1,1))
        CustomPanel(self,self.bagSizer,title="area del avatar",p=(1,2))
        self.bagSizer.AddGrowableRow(0)
        self.bagSizer.AddGrowableRow(1)
        self.bagSizer.AddGrowableCol(0)
        self.bagSizer.AddGrowableCol(1)
        self.bagSizer.AddGrowableCol(2)
        self.mainsizer.Add(self.bagSizer,flag=wx.EXPAND,proportion=1)

        self.bagSizer.Fit(self)
        self.mainsizer.Fit(self)
        
        self.panel.SetSizerAndFit(self.mainsizer)
        self.panel.GetSizer().Layout()
        self.Show(True)
    
class CustomFrame(wx.Frame):
    expandbutton=0
    toolbar=0
    name=0
    customFont=0
    expgif=0
    shrgif=0
    prop=0
    def __init__(self,master,name):
        wx.Frame.__init__(self,master,bd=3,background='white')
        self.name=name
        self.expgif=PhotoImage(file="expand.gif")
        self.shrgif=PhotoImage(file="shrink.gif")
        self.customFont = tkFont.Font(family='Buxton Sketch', size=14)
##        self.toolbar=Frame(self,background="white",bd=2,cursor="arrow")
##        self.toolbar.pack(side=TOP, fill="x")
##        self.expandbutton = Button(self.toolbar,command=self.expand,image=self.expgif,width="20",height="20")
##        self.expandbutton.pack(side="right")
##        title=Label(self.toolbar,text=self.name,font=self.customFont,background="white",fg="black")
##        title.pack(side="left")
    


app = wx.App(0)
frame = MainWindow()
app.MainLoop()
