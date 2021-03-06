# -*- coding: cp1252 -*-
from OpenGL.GL import * #@UnusedWildImport
import ntpath
import os.path
import math
from PIL import Image
import numpy as np
class MD5Model:
    filePath=""
    parent_path=""
    name=""
    numverts=0
    numtris=0
    numweights=0
    m_iMD5Version=10
    m_iNumJoints=0
    m_iNumMeshes=0
    scale=1.0
    swapYZ=False
    center=0
    m_Joints=[]
    m_Meshes=[]
    m_Animation=None
    m_LocalToWorldMatrix=[[1,1,1,1],
                          [1,1,1,1],
                          [1,1,1,1],
                          [1,1,1,1]]
    def setMaxSize(self,value):
        self.scale=value/self.getMaxLength()
    def __init__(self,swap=False):
        self.swapYZ=swap
    def LoadModel(self,filename):
        if not os.path.isfile(filename):
            print "MD5Model::LoadModel: Failed to find file: "+filename
            return False
        self.filePath=filename[:]
        self.parent_path, self.name = ntpath.split(filename)
        
        self.m_Joints=[]
        self.m_Meshes=[]

        function=0
        for line in open(self.filePath, "r"):
            words=line.replace("\t"," ").split(" ")
            if function!=0:
                function=function(words)
                continue
            if words[0] == "MD5Version":
                self.m_iMD5Version=int(words[1])
                assert(self.m_iMD5Version==10)
            if words[0]== "commandline":
                continue
            if words[0]== "numJoints":
                self.m_iNumJoints=int(words[1])
                continue
            if words[0] == "numMeshes":
                self.m_iNumMeshes=int(words[1])
                continue
            if words[0] == "joints":
                function=self.readJoint
                continue
            if words[0] == "mesh":
                self.m_Meshes.append(Mesh())
                continue
            if len(words)>1 and words[1] == "shader":
                self.m_Meshes[-1].m_Shader=words[2].replace("\"", "").replace("\n", "")
                texturePath=self.parent_path+"\\"+self.m_Meshes[-1].m_Shader
                path=os.path.splitext(texturePath)
                if path[1]=="":
                    texturePath+=".tga"
                if os.path.isfile(texturePath):
                    self.m_Meshes[-1].m_TexID=loadTex(texturePath)
                continue
            if len(words)>1 and words[1] == "numverts":
                self.numverts=int(words[2])
                function=self.readVert
                continue
            if len(words)>1 and words[1] == "numtris":
                self.numtris=int(words[2])
                function=self.readTris
                continue
            if len(words)>1 and words[1] == "numweights":
                self.numweights=int(words[2])
                function=self.readWeights
                continue
        self.arreglar()
        self.center=self.getWeightedCenter()
        animpath=os.path.splitext(self.filePath)[0]+".md5anim"
        if os.path.isfile(animpath):
            self.m_Animation=MD5Animation()
            self.m_Animation.setMesh(self)
            self.m_Animation.loadAnimation(animpath)
        self.Render=self.newrender
    def getMaxLength(self):
        maxv=[-99999999,-99999999,-99999999]
        minv=[99999999,99999999,99999999]
        for m in self.m_Meshes:
            for v in m.m_Verts:
                maxv[0]=max(maxv[0],v.m_Pos[0])
                maxv[1]=max(maxv[1],v.m_Pos[1])
                maxv[2]=max(maxv[2],v.m_Pos[2])
                minv[0]=min(minv[0],v.m_Pos[0])
                minv[1]=min(minv[1],v.m_Pos[1])
                minv[2]=min(minv[2],v.m_Pos[2])
        val=[maxv[2]-minv[2],maxv[1]-minv[1],maxv[0]-minv[0]]
##        return min(val[0],min(val[1],val[2]))
        return math.sqrt(val[0]**2+val[1]**2+val[2]**2)
    def getWeightedCenter(self):
        c=[0,0,0]
        num=0
        for m in self.m_Meshes:
            num+=len(m.m_Verts)
            for v in m.m_Verts:
                c=sumV(c,v.m_Pos)
        return mulS(c,1.0/num)
    def getCenter(self):
        c=[0,0,0]
        for m in self.m_Meshes:
            c=sumV(c,m.getCenter())
        return mulS(c,1.0/len(self.m_Meshes))
    def arreglar(self):
        for mesh in self.m_Meshes:
            self.PrepareMesh(mesh)
            self.PrepareNormals(mesh)
            if self.swapYZ:
                for vertex in mesh.m_Verts:
                    temp=vertex.m_Pos[1]
                    vertex.m_Pos[1]=vertex.m_Pos[2]
                    vertex.m_Pos[2]=temp

                    temp=vertex.m_Normal[1]
                    vertex.m_Normal[1]=vertex.m_Normal[2]
                    vertex.m_Normal[2]=temp
    def PrepareMesh(self,mesh,skel=0):
        if skel==0:
            mesh.m_PositionBuffer=[]
            mesh.m_Tex2DBuffer=[]
            for i in range(len(mesh.m_Verts)):
                mesh.m_Verts[i].m_Pos=[0,0,0]
                mesh.m_Verts[i].m_Normal=[0,0,0]
                for j in range(mesh.m_Verts[i].m_WeightCount):
                    weight=mesh.m_Weights[mesh.m_Verts[i].m_StartWeight + j]
                    joint=self.m_Joints[weight.m_JointID]
                    rotPos =vectorQ(weight.m_Pos,joint.m_Orient)
                    mesh.m_Verts[i].m_Pos =sumV(mesh.m_Verts[i].m_Pos,mulS(sumV(joint.m_Pos,rotPos),weight.m_Bias))
                mesh.m_PositionBuffer.append(mesh.m_Verts[i].m_Pos)
                mesh.m_Tex2DBuffer.append(mesh.m_Verts[i].m_Tex0)
        else:
            for i in range(len(mesh.m_Verts)):
                vert = mesh.m_Verts[i]
                mesh.m_PositionBuffer[i]=[0,0,0]
                mesh.m_NormalBuffer[i]=[0,0,0]
                for j in range(vert.m_WeightCount):
                    weight = mesh.m_Weights[vert.m_StartWeight+j]
                    joint = skel.m_Joints[weight.m_JointID]
                    rotPos =vectorQ(weight.m_Pos,joint.m_Orient)
                    mesh.m_PositionBuffer[i]=sumV(mesh.m_PositionBuffer[i],mulS(sumV(joint.m_Pos,rotPos),weight.m_Bias))
                    mesh.m_NormalBuffer[i]=sumV(mesh.m_NormalBuffer[i],mulS(vectorQ(vert.m_Normal,joint.m_Orient),weight.m_Bias))
        return True
    def PrepareNormals(self,mesh):
        mesh.m_NormalBuffer=[]
        #Loop through all triangles and calculate the normal of each triangle
        for i in range(len(mesh.m_Tris)):
            v0=mesh.m_Verts[ mesh.m_Tris[i].m_Indices[0]].m_Pos
            v1=mesh.m_Verts[ mesh.m_Tris[i].m_Indices[1]].m_Pos
            v2=mesh.m_Verts[ mesh.m_Tris[i].m_Indices[2]].m_Pos
            normal=cross(restaV(v2,v0),restaV(v1,v0))
            mesh.m_Verts[ mesh.m_Tris[i].m_Indices[0] ].m_Normal=sumV(mesh.m_Verts[ mesh.m_Tris[i].m_Indices[0] ].m_Normal,normal)
            mesh.m_Verts[ mesh.m_Tris[i].m_Indices[1] ].m_Normal=sumV(mesh.m_Verts[ mesh.m_Tris[i].m_Indices[1] ].m_Normal,normal)
            mesh.m_Verts[ mesh.m_Tris[i].m_Indices[2] ].m_Normal=sumV(mesh.m_Verts[ mesh.m_Tris[i].m_Indices[2] ].m_Normal,normal)
        ##Now normalize all the normals
        for i in range(len(mesh.m_Verts)):
            vert=mesh.m_Verts[i]
            normal=normalize(vert.m_Normal)
            mesh.m_NormalBuffer.append(normal)
            vert.m_Normal =[0,0,0]
            for j in range(vert.m_WeightCount):
                weight=mesh.m_Weights[vert.m_StartWeight + j]
                joint = self.m_Joints[weight.m_JointID]
                vert.m_Normal=sumV(vert.m_Normal,mulS(qv_mult(joint.m_Orient,normal),weight.m_Bias))
        return True
    def Render(self):
        pass   
    def newrender(self):
        glPushMatrix()       
##        glMultMatrixf(self.m_LocalToWorldMatrix)
        glScalef(self.scale,self.scale,self.scale)
        glTranslatef(-self.center[0],-self.center[1],-self.center[2])
        for mesh in self.m_Meshes:
            mesh.RenderMesh()
##            mesh.RenderNormals()
        if not (self.m_Animation is None):
            self.m_Animation.Render()
        glPopMatrix()
    def readWeights(self,words):
        if len(self.m_Meshes[-1].m_Weights)==self.numweights:
            return 0
        weight=Weight()
        weight.m_JointID=int(words[3])
        weight.m_Bias=float(words[4])
        weight.m_Pos[0]=float(words[6])
        weight.m_Pos[1]=float(words[7])
        weight.m_Pos[2]=float(words[8])
        self.m_Meshes[-1].m_Weights.append(weight)
        return self.readWeights
        
    def readTris(self,words):
        if len(self.m_Meshes[-1].m_Tris)==self.numtris:
            return 0
        tri=Triangle()
        tri.m_Indices[0]=int(words[3])
        tri.m_Indices[1]=int(words[4])
        tri.m_Indices[2]=int(words[5])
        self.m_Meshes[-1].m_Tris.append(tri)
        self.m_Meshes[-1].m_IndexBuffer.append(tri.m_Indices[0])
        self.m_Meshes[-1].m_IndexBuffer.append(tri.m_Indices[1])
        self.m_Meshes[-1].m_IndexBuffer.append(tri.m_Indices[2])
        return self.readTris

    def readVert(self,words):
        if len(self.m_Meshes[-1].m_Verts)==self.numverts:
            return 0
        vert=Vertex()
        vert.m_Tex0[0]=float(words[4])
        vert.m_Tex0[1]=float(words[5])
        vert.m_StartWeight=int(words[7])
        vert.m_WeightCount=int(words[8])
        self.m_Meshes[-1].m_Verts.append(vert)
        self.m_Meshes[-1].m_Tex2DBuffer.append(vert.m_Tex0)
        return self.readVert
    def readJoint(self,words):
        if len(self.m_Joints)==self.m_iNumJoints:
            return 0
        joint=Joint()
        joint.m_Name=words[1].replace("\"", "")
        joint.m_ParentID=int(words[2]) 
        joint.m_Pos[0]=float(words[4])
        joint.m_Pos[1]=float(words[5])
        joint.m_Pos[2]=float(words[6])
        joint.m_Orient[0]=float(words[9])
        joint.m_Orient[1]=float(words[10])
        joint.m_Orient[2]=float(words[11])
        joint.m_Orient=ComputeQuatW( joint.m_Orient )
        self.m_Joints.append(joint)
        return self.readJoint
    def Update(self,fDeltaTime):
        self.m_Animation.Update(fDeltaTime)
        skeleton = self.m_Animation.m_AnimatedSkeleton
        for mesh in self.m_Meshes:
            self.PrepareMesh(mesh,skeleton)
    
def loadTex(filename):
    img = Image.open(filename) # .jpg, .bmp, etc. also work
    img_data = np.array(list(img.getdata()), np.int8)
    glEnable(GL_TEXTURE_2D)
    texture = glGenTextures(1)
    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.size[0], img.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    return texture
def vectorQ(v,q):
    u=[q[0], q[1], q[2]]
    s = q[3]
    return sumV(sumV(mulS(u,dot(u, v)*2.0),mulS(v,s*s-dot(u,u))),mulS(cross(u,v),2*s))
def Quat_multVec(q, v):
    out=[0,0,0,0]
    out[0] = - (q[0] * v[0]) - (q[1] * v[1]) - (q[2] * v[2])
    out[1] =   (q[3] * v[0]) + (q[1] * v[2]) - (q[2] * v[1])
    out[2] =   (q[3] * v[1]) + (q[2] * v[0]) - (q[0] * v[2])
    out[3] =   (q[3] * v[2]) + (q[0] * v[1]) - (q[1] * v[0])
    return out
def Quat_rotatePoint(q,v):
    tmp=[0,0,0,0]
    inv=[0,0,0,0]
    final=[0,0,0,0]
    inv[1]=-q[0]
    inv[2]=-q[1]
    inv[3]=-q[2]
    inv[0]=q[3]

    inv=normalize(inv)
    tmp=Quat_multVec(q,v)
    final=q_mult(tmp, inv)
    out=[0,0,0]
    out[0]=final[0]
    out[1]=final[1]
    out[2]=final[2]
    return out
def dot(v1,v2):
    r=0
    for i in range(len(v1)):
        r+=v1[i]*v2[i]
    return r
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
def mulS(v,s):
    return [x*s for x in v]
def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return c
def restaV(v1,v2):
    r=[]
    for i in range(len(v1)):
        r.append(v1[i]-v2[i])
    return r
def mulV(v1,v2):
    r=[]
    for i in range(len(v1)):
        r.append(v1[i]*v2[i])
    return r
def sumV(v1,v2):
    r=[]
    for i in range(len(v1)):
        r.append(v1[i]+v2[i])
    return r
def q_conjugate(q):
    w, x, y, z = q
    return [w, -x, -y, -z]
def q_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return [x, y, z, w]
def qv_mult(q1, v1):
    q2 = [0.0] + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]        
def ComputeQuatW(quat):
    t = 1.0-(quat[0]*quat[0])-(quat[1]*quat[1])-(quat[2]*quat[2])
    if t < 0.0:
        quat[3]=0.0
    else:
        quat[3]=-math.sqrt(t)
    return quat
class Vertex:
    def __init__(self):
        self.m_Pos=[0,0,0]
        self.m_Normal=[0,0,0]
        self.m_Tex0=[0,0]
        self.m_StartWeight=0
        self.m_WeightCount=0
class Triangle:
    def __init__(self):
        self.m_Indices=[0,0,0]

class Weight:
    def __init__(self):
        self.m_JointID=0
        self.m_Bias=0.0
        self.m_Pos=[0,0,0]
 
class Joint:                           
    def __init__(self):
        self.m_Name=""
        self.m_ParentID=0
        self.m_Pos=[0,0,0]
        self.m_Orient=[0,0,0,0]
    def __str__(self):
        s=self.m_Name+"{ parent: "+str(self.m_ParentID)+" position: "+str(self.m_Pos)+" orient: "+str(self.m_Orient)
        return s
    def copy(self):
        nj=Joint()
        nj.m_Name=self.m_Name
        nj.m_ParentID=self.m_ParentID
        nj.m_Pos[0]=self.m_Pos[0]
        nj.m_Pos[1]=self.m_Pos[1]
        nj.m_Pos[2]=self.m_Pos[2]
        
        nj.m_Orient[0]=self.m_Orient[0]
        nj.m_Orient[1]=self.m_Orient[1]
        nj.m_Orient[2]=self.m_Orient[2]
        nj.m_Orient[3]=self.m_Orient[3]
        return nj
    def copyToSkeletonJoint(self):
        r=SkeletonJoint()
        r.m_ParentID=self.m_ParentID
        r.m_Pos[0]=float(self.m_Pos[0])
        r.m_Pos[1]=float(self.m_Pos[1])
        r.m_Pos[2]=float(self.m_Pos[2])
        r.m_Orient[0]=float(self.m_Orient[0])
        r.m_Orient[1]=float(self.m_Orient[1])
        r.m_Orient[2]=float(self.m_Orient[2])
        r.m_Orient[3]=float(self.m_Orient[3])
        return r
class Mesh:
    def __init__(self):
        self.m_Shader=""
        self.m_Verts=[]
        self.m_Tris=[]
        self.m_Weights=[]
        self.m_TexID=0

        self.m_PositionBuffer=[]
        self.m_NormalBuffer=[]
        self.m_Tex2DBuffer=[]
        self.m_IndexBuffer=[]
 
    def getCenter(self):
        c=[0,0,0]
        for vertex in self.m_Verts:
            c[0]+=vertex.m_Pos[0]
            c[1]+=vertex.m_Pos[1]
            c[2]+=vertex.m_Pos[2]
        c[0]/=len(self.m_Verts)
        c[1]/=len(self.m_Verts)
        c[2]/=len(self.m_Verts)
        return c
    def mover(self,m):
        for vertex in self.m_Verts:
            vertex.m_Pos[0]-=m[0]
            vertex.m_Pos[1]-=m[1]
            vertex.m_Pos[2]-=m[2]
    def centrar(self):
        c=self.getCenter()
        for vertex in self.m_Verts:
            vertex.m_Pos[0]-=c[0]
            vertex.m_Pos[1]-=c[1]
            vertex.m_Pos[2]-=c[2]
    def RenderMesh(self):
        glColor3f( 1.0, 1.0, 1.0 )
        glEnableClientState( GL_VERTEX_ARRAY )
        glEnableClientState( GL_TEXTURE_COORD_ARRAY )
        glEnableClientState( GL_NORMAL_ARRAY )
        
        glBindTexture( GL_TEXTURE_2D, self.m_TexID )
        glVertexPointer( 3, GL_FLOAT, 0,self.m_PositionBuffer)
        glNormalPointer( GL_FLOAT, 0 ,self.m_NormalBuffer)
        glTexCoordPointer( 2, GL_FLOAT, 0,self.m_Tex2DBuffer)
        glDrawElements( GL_TRIANGLES, len(self.m_IndexBuffer), GL_UNSIGNED_INT,self.m_IndexBuffer)
 
        glDisableClientState( GL_NORMAL_ARRAY )
        glDisableClientState( GL_TEXTURE_COORD_ARRAY )
        glDisableClientState( GL_VERTEX_ARRAY )
        glBindTexture( GL_TEXTURE_2D, 0 )
    def RenderNormals(self):
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_LIGHTING)
        glColor3f(1.0,1.0,0.0)
        glBegin(GL_LINES)
        for i in range(len(self.m_PositionBuffer)):
            p0=self.m_PositionBuffer[i]
            p1=sumV(self.m_PositionBuffer[i],self.m_NormalBuffer[i])
            glVertex3fv(p0)
            glVertex3fv(p1)
        glEnd()
        glPopAttrib()
    def RenderSkeleton(self,joints):
        glPointSize(5.0)
        glColor3f(1.0,0.0,0.0)
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glBegin(GL_POINTS)
        for joint in joints:
            glVertex3fv(joint.m_Pos)
        glEnd()
        glColor3f( 0.0, 1.0, 0.0 )
        glBegin( GL_LINES )
        for j0 in joints:
            if j0.m_ParentID!=-1:
                j1 = joints[j0.m_ParentID]
                glVertex3fv(j0.m_Pos)
                glVertex3fv(j1.m_Pos)
        glEnd()
        glPopAttrib()
 
    def CheckAnimation(self,animation):
        if self.m_iNumJoints != animation.GetNumJoints():
            return False 
        for i in range(len(self.m_Joints)):
            if ( m_Joints[i].m_Name !=  animation.GetJointInfo( i ).m_Name or
             m_Joints[i].m_ParentID !=  animation.GetJointInfo( i ).m_ParentID ):
                return False
        return True
class MD5Animation:
    model=0
    def setMesh(self,model):
        self.model=model
    def __init__(self):
        self.m_JointInfos=[]
        self.m_Bounds=[]
        self.m_BaseFrames=[]
        self.m_Frames=[]
        self.m_Skeletons=[]
        self.m_AnimatedSkeleton=FrameSkeleton()

        self.m_iMD5Version=0
        self.m_iNumFrames=0
        self.m_iNumJoints=0
        self.m_iFrameRate=0
        self.m_iNumAnimatedComponents=0

        self.m_fAnimDuration=0.0
        self.m_fFrameDuration=0.0
        self.m_fAnimTime=0.0
    def GetNumJoints(self):
        return self.m_iNumJoints
    def GetJointInfo(self,index):
        assert(index<self.m_JointInfos.size())
        return self.m_JointInfos[index]
    def getBlankSkeleton(self):
        skeleton=FrameSkeleton()
        for joint in self.model.m_Joints:
            skeleton.m_Joints.append(joint.copy())
        return skeleton       
    def setSkeleton(self,skeleton):
        for i in range(self.m_iNumJoints):
            joint = skeleton.m_Joints[i]
            self.m_AnimatedSkeleton.m_Joints[i].m_ParentID = joint.m_ParentID
            self.m_AnimatedSkeleton.m_Joints[i].m_Pos =joint.m_Pos
            self.m_AnimatedSkeleton.m_Joints[i].m_Orient=joint.m_Orient
        for mesh in self.model.m_Meshes:
            self.model.PrepareMesh(mesh,skeleton)
    def InterpolateSkeletons(self,skeleton0,skeleton1,fInterpolate):
        for i in range(self.m_iNumJoints):
            joint0 = skeleton0.m_Joints[i]
            joint1 = skeleton1.m_Joints[i]
            self.m_AnimatedSkeleton.m_Joints[i].m_ParentID = joint1.m_ParentID
            self.m_AnimatedSkeleton.m_Joints[i].m_Pos =lerp(joint0.m_Pos,joint1.m_Pos,fInterpolate)
            self.m_AnimatedSkeleton.m_Joints[i].m_Orient=mix(joint0.m_Orient,joint1.m_Orient,fInterpolate)
    def loadAnimation(self,filename):
        self.filePath=filename[:]
        self.parent_path, self.name = ntpath.split(filename)
        
        self.m_JointInfos=[]
        self.m_Bounds=[]
        self.m_BaseFrames=[]
        self.m_Frames=[]
        self.m_AnimatedSkeleton.m_Joints=[]
        self.m_iNumFrames = 0
    
        function=0
        for line in open(self.filePath, "r"):
            words=line.replace("\t"," ").split(" ")
            if function!=0:
                function=function(words)
                continue
            if words[0] == "MD5Version":
                self.m_iMD5Version=int(words[1])
                continue
            if words[0] == "commandline":
                continue
            if words[0] == "numFrames":
                self.m_iNumFrames=int(words[1])
                continue
            if words[0] == "numJoints":
                self.m_iNumJoints=int(words[1])
                continue
            if words[0] == "frameRate":
                self.m_iFrameRate=int(words[1])
                continue
            if words[0] == "numAnimatedComponents":
                self.m_iNumAnimatedComponents=int(words[1])
                continue
            if words[0] == "hierarchy":
                function=self.readHierarchy
                continue
            if words[0] == "bounds":
                function=self.readBounds
                continue
            if words[0] == "baseframe":
                function=self.readBaseFrame
                continue
            if words[0] == "frame":
                frame=FrameData()
                frame.m_iFrameID=int(words[1])
                self.m_Frames.append(frame)
                function=self.readFrameData
                continue
        # Make sure there are enough joints for the animated skeleton.
        for i in range(self.m_iNumJoints):
            self.m_AnimatedSkeleton.m_Joints.append(SkeletonJoint())
        self.m_fFrameDuration = 1.0/float(self.m_iFrameRate)
        self.m_fAnimDuration = (self.m_fFrameDuration*float(self.m_iNumFrames))
        self.m_fAnimTime = 0.0

        assert(len(self.m_JointInfos)==self.m_iNumJoints)
        assert(len(self.m_Bounds)==self.m_iNumFrames)
        assert(len(self.m_BaseFrames)==self.m_iNumJoints)
        assert(len(self.m_Frames)==self.m_iNumFrames)
        assert(len(self.m_Skeletons)==self.m_iNumFrames)
        return True
    def BuildFrameSkeleton(self,frameData):
        skeleton=FrameSkeleton()
        for i in range(len(self.m_JointInfos)):
            j=0
            jointInfo=self.m_JointInfos[i]
            animatedJoint = self.m_BaseFrames[i].copy()
            animatedJoint.m_ParentID = jointInfo.m_ParentID
            if self.m_JointInfos[i].m_Flags & 1: ## Pos.x
                animatedJoint.m_Pos[0] = float(frameData.m_FrameData[self.m_JointInfos[i].m_StartIndex+j])
                j+=1
            if self.m_JointInfos[i].m_Flags & 2: ## Pos.y
                animatedJoint.m_Pos[1] = float(frameData.m_FrameData[self.m_JointInfos[i].m_StartIndex+j])
                j+=1
            if self.m_JointInfos[i].m_Flags & 4: ## Pos.z
                animatedJoint.m_Pos[2] = float(frameData.m_FrameData[self.m_JointInfos[i].m_StartIndex+j])
                j+=1
            if self.m_JointInfos[i].m_Flags & 8: ## Orient.x
                animatedJoint.m_Orient[0] = float(frameData.m_FrameData[self.m_JointInfos[i].m_StartIndex+j])
                j+=1
            if self.m_JointInfos[i].m_Flags & 16: ## Orient.y
                animatedJoint.m_Orient[1] = float(frameData.m_FrameData[self.m_JointInfos[i].m_StartIndex+j])
                j+=1
            if self.m_JointInfos[i].m_Flags & 32: ## Orient.z
                animatedJoint.m_Orient[2] = float(frameData.m_FrameData[self.m_JointInfos[i].m_StartIndex+j])
                j+=1
            animatedJoint.m_Orient=ComputeQuatW(animatedJoint.m_Orient)
            if animatedJoint.m_ParentID >= 0: ## Has a parent joint
                parentJoint = skeleton.m_Joints[animatedJoint.m_ParentID]
                rotPos =vectorQ(animatedJoint.m_Pos,parentJoint.m_Orient)
                animatedJoint.m_Pos = sumV(parentJoint.m_Pos,rotPos)
                animatedJoint.m_Orient = q_mult(parentJoint.m_Orient,animatedJoint.m_Orient)
                animatedJoint.m_Orient = normalize(animatedJoint.m_Orient)
            skeleton.m_Joints.append(animatedJoint)
        self.m_Skeletons.append(skeleton)
    def readFrameData(self,words):
        if len(self.m_Frames[-1].m_FrameData)==self.m_iNumAnimatedComponents:
            self.BuildFrameSkeleton(self.m_Frames[-1])
            return 0
        self.m_Frames[-1].m_FrameData.append(float(words[1]))
        self.m_Frames[-1].m_FrameData.append(float(words[2]))
        self.m_Frames[-1].m_FrameData.append(float(words[3]))
        self.m_Frames[-1].m_FrameData.append(float(words[4]))
        self.m_Frames[-1].m_FrameData.append(float(words[5]))
        self.m_Frames[-1].m_FrameData.append(float(words[6]))
        return self.readFrameData
    def readBaseFrame(self,words):
        if len(self.m_BaseFrames)==self.m_iNumJoints:
            return 0
        baseFrame=BaseFrame()
        baseFrame.m_Pos[0]=float(words[2])
        baseFrame.m_Pos[1]=float(words[3])
        baseFrame.m_Pos[2]=float(words[4])

        baseFrame.m_Orient[0]=float(words[7])
        baseFrame.m_Orient[1]=float(words[8])
        baseFrame.m_Orient[2]=float(words[9])
        self.m_BaseFrames.append(baseFrame)
        return self.readBaseFrame
    def readBounds(self,words):
        if len(self.m_Bounds)==self.m_iNumFrames:
            return 0
        bound=Bound()
        bound.m_Min[0]=float(words[2])
        bound.m_Min[1]=float(words[3])
        bound.m_Min[2]=float(words[4])

        bound.m_Max[0]=float(words[7])
        bound.m_Max[1]=float(words[8])
        bound.m_Max[2]=float(words[9])

        self.m_Bounds.append(bound)
        return self.readBounds
    def readHierarchy(self,words):
        if len(self.m_JointInfos)==self.m_iNumJoints:
            return 0
        joint=JointInfo()
        joint.m_Name=words[1].replace("\"", "")
        joint.m_ParentID=int(words[2])
        joint.m_Flags=int(words[3])
        joint.m_StartIndex=int(words[4])
        self.m_JointInfos.append(joint)
        return self.readHierarchy
    ## Update this animation's joint set.
    def Update(self,fDeltaTime):
        if self.m_iNumFrames < 1:
            return
        self.m_fAnimTime+=fDeltaTime
        while self.m_fAnimTime>self.m_fAnimDuration:
                self.m_fAnimTime -= self.m_fAnimDuration
        while self.m_fAnimTime<0.0:
            self.m_fAnimTime+=self.m_fAnimDuration
        ## Figure out which frame we're on
        fFramNum=self.m_fAnimTime*float(self.m_iFrameRate)
        iFrame0=int(math.floor(fFramNum))
        iFrame1=int(math.ceil(fFramNum))
        iFrame0=iFrame0%self.m_iNumFrames
        iFrame1=iFrame1%self.m_iNumFrames
        fInterpolate=math.fmod(self.m_fAnimTime,self.m_fFrameDuration)/self.m_fFrameDuration
        self.InterpolateSkeletons(self.m_Skeletons[iFrame0],self.m_Skeletons[iFrame1],fInterpolate)
    ##Draw the animated skeleton
    def Render(self):
        glPointSize(5.0)
        glColor3f(1.0,0.0,0.0)
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        joints = self.m_AnimatedSkeleton.m_Joints
##        // Draw the joint positions
        glBegin(GL_POINTS)
        for joint in joints:
            glVertex3fv(joint.m_Pos)
        glEnd()

##    // Draw the bones
        glColor3f(0.0,1.0,0.0)
        glBegin(GL_LINES)
        for j0 in joints:
            if j0.m_ParentID!=-1:
                j1 = joints[j0.m_ParentID]
                glVertex3fv(j0.m_Pos)
                glVertex3fv(j1.m_Pos)
        glEnd()
        glPopAttrib()

def lerp(v1,v2,p):
    r=[0,0,0]
    r[0]=v1[0]+(v2[0]-v1[0])*p
    r[1]=v1[1]+(v2[1]-v1[1])*p
    r[2]=v1[2]+(v2[2]-v1[2])*p
    return r
def mix(Orient0,Orient1,fInterpolate):
    orientf=[0,0,0,0]
    orientf[0]=(Orient1[0]*fInterpolate)+(Orient0[0]*(1-fInterpolate))
    orientf[1]=(Orient1[1]*fInterpolate)+(Orient0[1]*(1-fInterpolate))
    orientf[2]=(Orient1[2]*fInterpolate)+(Orient0[2]*(1-fInterpolate))
    orientf[3]=(Orient1[3]*fInterpolate)+(Orient0[3]*(1-fInterpolate))
    return orientf
class JointInfo:
    m_Name=0
    m_ParentID=0
    m_Flags=0
    m_StartIndex=0
    def __init__(self):
        self.m_Name=""
        self.m_ParentID=0
        self.m_Flags=0
        self.m_StartIndex=0
class Bound:
    m_Min=0
    m_Max=0
    def __init__(self):
        self.m_Min=[0,0,0]
        self.m_Max=[0,0,0]
class BaseFrame:
    m_Pos=0
    m_Orient=0
    m_ParentID=-1
    def __init__(self):
        self.m_Pos=[0,0,0]
        self.m_Orient=[0,0,0,0]
    def copy(self):
        newbs=BaseFrame()
        newbs.m_ParentID=self.m_ParentID
        
        newbs.m_Pos[0]=self.m_Pos[0]
        newbs.m_Pos[1]=self.m_Pos[1]
        newbs.m_Pos[2]=self.m_Pos[2]
        
        newbs.m_Orient[0]=self.m_Orient[0]
        newbs.m_Orient[1]=self.m_Orient[1]
        newbs.m_Orient[2]=self.m_Orient[2]
        newbs.m_Orient[3]=self.m_Orient[3]
        return newbs
        
class FrameData:
    m_iFrameID=0
    m_FrameData=0
    def __init__(self):
        self.m_iFrameID=0
        self.m_FrameData=[]

class SkeletonJoint:
    m_ParentID=0
    m_Pos=[0,0,0]
    m_Orient=[0,0,0,0]
    m_Name=0
    def __init__(self,copy=0):
        self.m_Name=""
        if copy==0:
            self.m_ParentID=-1
            self.m_Pos=[0,0,0]
        else:
            self.m_Pos=copy.m_Pos
            self.m_Orient=copy.m_Orient
class FrameSkeleton:
    def __str__(self):
        s=""
        for joint in self.m_Joints:
            s+=str(joint)+"\n"
        return s
    def __init__(self):
        self.m_Joints=[]
