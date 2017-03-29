
'''
Module hitfruit. Contains functions to label and segmentation/skeletonisation of pictures of harvested plants

@author: Moises Exposito-Alonso (moisesexpositoalonso@gmail.com)

'''
import os, time, pandas, sys
import math
import cv2
import mahotas as mh
import numpy as np 
from PIL import Image
import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl
# import pymorph as pm
# import networkx as nx
# from scipy import ndimage as nd
# import skimage.transform as transform
# import skimage.io as sio
# import scipy.misc as sm


sys.path.append('/home/moisesexpositoalonso/ebio/abt6/mexposito/mpy')
sys.path.append('/ebio/abt6/mexposito/mpy')
from moi import *

sys.path.append('/home/moisesexpositoalonso/ebio/abt6_projects7/ath_1001G_field/hippo')
sys.path.append('/ebio/abt6/mexposito/ebio/abt6_projects7/ath_1001G_field/hippo')
from hippo import *

def view_ask(thefile, windowsnap=True):
    """
    Human interactive labeling of images.

    Parameters
    ----------
    thefile : str
        An image file 
    
    Returns
    -------
    result : list
        The list contains the values of the answer from the human looking at the image

    Notes
    -----
    This function needs to have all terminal windows 

    """
    # thefile=str(thefile)
    # print 'this is the file to analyse inside function', thefile

    ## Open the image
    p = subprocess.Popen(["eog" ,'--disable-gallery', '--new-instance',thefile], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE )
        # com=p.communicate() # does not work. because viewer is constant process it waits infinity
        # print(com)
    
    # If need help snapping window for better visualization. CAREFUL highly customized
    if(windowsnap==True):
        # waits 1 second for the image to be loaded 
        time.sleep(0.5) 
    
        # command to snap the window into my left screen. If not using same settings, comment out
        # cmd=str("wmctrl -r " + os.path.basename(thefile) +' -e 0,0,0,1280,1400') # snaps one side
        # #cmd=str("wmctrl -r " + os.path.basename(thefile) +' -e 0,0,0,2560,1400') # snaps to oposite screen
        # subprocess.call(cmd,shell=True)
    
    
    # command to put the terminal window to fron to be able to easily answer the questions 
    subprocess.call( str('wmctrl -a Terminal'),shell=True)
   
    # BIG CONTROLER

    try:
    ## Starts the questions
    # question 1

        tray=raw_input("Tray number [1:349] (..=repeated) (.= curate) > ")
        
        if(tray==""):
            theanswer=None
        elif(tray==".." ):
            theanswer=["curate"]*5
        elif(tray=="."):
            theanswer=["rep"]*5

        # elif(float(tray) in range(0,350)):
        elif(str(tray) in str(range(0,350))):
    # question 2
            pos=""
            while pos not in ['a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7']:
                try:
                    pos=raw_input("Pot position [A:E] [1:8]> ")
                except KeyboardInterrupt:
                    raise ("Stoped manually by user")
    # question 3            
            plant=["",""]
            while plant[0] not in ['e','r']:
                try:
                    plant=raw_input("Inflorescence (e) or rosette (r), ind ("") or pop(>2, or -9 if missing) [example r21] > ")
                except KeyboardInterrupt:
                    raise ("Stoped manually by user") 
            print 'the plant input is correct', plant

        # fill in question 4 based on 3            
            if(len(plant)==1):
                print 'this is an individual tray'
                ptype='ind'
                nind='NA'           
            else:
                ptype='pop'
                nind=plant[1:]
                print nind
                while nind not in str(range(-9,50)):
                    nind=raw_input("There was an error in the pop count. input number: ")

            theanswer=[str(int(tray)),str(pos),str(plant[0]),str(ptype),str(nind)]   

            
        # exceptions
        else:
            theanswer=None
    
    except KeyboardInterrupt:
        raise Exception(' Manually stopped, skip loop and finish! ')
    
    ## Kill image viewer to start again with the next image
    p.kill()
    return theanswer


def removepathdot(listpaths):
    newpaths=[f[1:] for f in listpaths]
    return newpaths

def get_imageindex_massive(path):
    if "tueharvest_image_index.csv" in os.listdir(path):
        print "tueharvest_image_index.csv found in folder!"
        imageindex= pandas.read_csv(path+"tueharvest_image_index.csv",error_bad_lines=False).sort_values(by='pathimage',ascending=True).values.tolist() ## IMPORTANT!

    else:
        files=getfileslower("JPG",path)
        # imageindex=[[x]for x in files]
        imageindex=[[x] for x in files]
    return imageindex


def save_imageindex(path,iid):
    if "tueharvest_image_index.csv" in os.listdir(path):
        try:
            answer=raw_input("\ntueharvest_image_index.csv exists. decide: overwrite(yes,y), backup (back,b), stop (no,n)\n")
            #if answer == "no" or answer=="n":
            #    raise Exception (" >>stopped to avoid overwriting!<<")
        except KeyboardInterrupt:
            answer="b"

        if answer in ["backup" ,"b",""]:
            subprocess.call(str("cp " + path+ "tueharvest_image_index.csv " + path + "tueharvest_image_index.csv_backup"+time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())+".csv") ,shell=True)
        if answer in ["yes","y","backup","b",""]:
            iid.to_csv(path+'tueharvest_image_index.csv',index=False)

    else:
        iid.to_csv(path+'tueharvest_image_index.csv',index=False)


def float2int(x):
    try:
        x=int(float(x))
    except Exception:
        "donothing"
    return x



def removeframe(img):
    ''' remove frame of the metal frame of photobox '''
    x1=200
    y1=600
    x2=4700
    y2=3200
    img=img[y1:y2,x1:x2]
    return img

def denoise(img):
    denoised=cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    return denoised


def hsvrange(ihsv):
    """ 
    Segment HSV transformed image with fixed ranges
    Arguments:
    -----------
        ihsv: numpy.ndarray
            Input image of three channels, HSV
    Returns:
    -----------
        numpy.ndarray
            The original image segmented to those areas within the range
    """
    h_upper=70
    h_lower=30

    s_upper=255
    s_lower=65

    v_upper=220
    v_lower=20
    

    hue=ihsv[:,:,0]
    saturation=ihsv[:,:,1]
    value=ihsv[:,:,2]
    
    h_mask = cv2.inRange(hue, h_lower,h_upper)
    s_mask = cv2.inRange(saturation, s_lower,s_upper)
    v_mask = cv2.inRange(value, v_lower,v_upper)

    result = cv2.bitwise_and(img,img, mask= h_mask)
    result = cv2.bitwise_and(result,result, mask= s_mask)
    result = cv2.bitwise_and(result,result, mask= v_mask)

    return result

def dealwhitelabel(im,dil=50,ero=15, what='get',bound=False, ontooriginal=True):
    """ 
    Locate a big white object in a picture by erode/dilute algorithms
    Arguments:
    -----------
        im: numpy.ndarray
            Input image of one channel, typically grey
        dil: float/interger
            Number of iterations in dilute algorithm
        erode: float/interger
            Number of iterations in erode algorithm. Normally more than dilute 
            to be able to get ride of small objects
        what: string, either 'get' 'remove'
            Whether you want to extract the big object or remove it.
        bound: Logical False/True
            If true, and what=='get', it will crop the image to the detected white area
    Returns:
    -----------
        numpy.ndarray
            The segmented image
    """
    # Otsu adaptive segmentation
    imth=otsuthres(im)

    # Denoise and contour algorithms
    tomask=erode(imth,iterations=ero)
    tomask=dilate(tomask,iterations=dil)

    # want to output orignal or the otsu-segmented one
    if ontooriginal==False:
        imoriginal=imth
    else:
        imoriginal=im

    # mask
    if what=='get':
        ir=getmask(imoriginal,tomask)
    elif what=='remove':
        ir=rmmask(imoriginal,tomask)
    else:
        raise NameError('You need to input get or remove as flags for what')

    # boundingbox in the big white part
    if bound ==True:
        ir=boundingbox(ir)
    return ir


##########################################################################################
# hitfruit class
##########################################################################################

class hitfruit(object):
    ''' class that contains image and the methods to analyse it '''
    def __init__(self,image):
        self.image = image
        self.ip=''
        self.hsv =''

        self.sk=''
        self.count =''
        self.skcount =''
        self.hog =''
        self.branches =''
        self.ends =''
        self.branchcount =''
        self.endscount =''
        self.bridges =''
        self.bridgescounts =''

     
    def segment(self):
        self.image=readcolimage(self.image)     
        self.ip=removeframe(self.image)
        self.ip=denoise(self.ip)
        self.ip=rgb2hsv(self.ip)
        self.ip=maskhsvdenoise(self.ip)
        return self

    def skeleton(self):      
        self.image=removeframe(readgreyimage(self.image))
        self.ip=dealwhitelabel(self.image, what='remove',ontooriginal=False)
        # self.ip=openclose(self.ip,open=False)
        self.sk =mh.thin( self.ip>127 )
        self.sk=pruning(self.sk, size=15)
        return self
    
    def denoise(self): # not useful
        self.ip= erode(self.ip,iterations=1)
        self.ip= dilate(self.ip,iterations=2)
        # self.image= denoise(self.image)
        # self.image=cv2.fastNlMeansDenoisingMulti(self.image, 2, 5, None, 4, 7, 35)
        return self
    
    def countnonzero(self):
        self.count=cv2.countNonZero(self.ip) 
        self.skcount=cv2.countNonZero(self.sk*1) 
        return self

    def branchedpoints(self):
        self.branches = branchedPoints(self.sk>0)
        self.branchcount =cv2.countNonZero(self.branches) 
        return self

    def endpoints(self):
        self.ends = endPoints(self.sk>0)
        self.endscount =cv2.countNonZero(self.ends) 
        return self

    # def bridges(self):
    #     print 'running bridges'
    #     towork=self.sk
    #     # towork=m.thin(towork, m.endpoints('homotopic'), 15)

    #     seA1 = np.array([[False, True, False],
    #                     [False,  True, False],
    #                     [True, False,  True]], dtype=bool)

    #     seB1 = np.array([[False, False, False],
    #                     [True, False,  True],
    #                     [False,  True, False]], dtype=bool)

    #     seA2 = np.array([[False, True, False],
    #                     [True,  True,  True],
    #                     [False, False, False]], dtype=bool)

    #     seB2 = np.array([[True, False,  True],
    #                     [False, False, False],
    #                     [False, True, False]], dtype=bool)
    #     hmt1 = m.se2hmt(seA1, seB1)
    #     hmt2 = m.se2hmt(seA2, seB2)
    #     towork = m.union(m.supcanon(towork, hmt1), m.supcanon(towork, hmt2))
    #     print towork.shape()
    #     towork = m.dilate(towork, m.sedisk(10))     
    #     towork = m.blob(m.label(towork), 'centroid')
    #     self.bridges = m.overlay(towork, m.dilate(self.image,m.sedisk(5)))
    #     # self.bridgescounts=towork.max()
    #     return self

    def zeroone(self):
        self.image=(self.image>127)*1
        return self

    def gethog(self):
        # hog = cv2.HOGDescriptor()
        self.hog = hog.compute(i.image)
        return self


def save_object(obj, filename):
    ''' 
    save an binary object using pickle
    
    Usage:
    -----------
    save_object(company1, 'company1.pkl')

    '''
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



def branchedPoints(skel, showSE=True):
    """
    The branching function was written by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699

    """
    # X=[]
    # #cross X
    # X0 = np.array([[0, 1, 0], 
    #                [1, 1, 1], 
    #                [0, 1, 0]])
    # X1 = np.array([[1, 0, 1], 
    #                [0, 1, 0], 
    #                [1, 0, 1]])
    # X.append(X0)
    # X.append(X1)
    
    #T like
    T=[]
    #T0 contains X0
    T0=np.array([[2, 1, 2], 
                 [1, 1, 1], 
                 [2, 2, 2]])
            
    T1=np.array([[1, 2, 1], 
                 [2, 1, 2],
                 [1, 2, 2]])  # contains X1
  
    T2=np.array([[2, 1, 2], 
                 [1, 1, 2],
                 [2, 1, 2]])
    
    T3=np.array([[1, 2, 2],
                 [2, 1, 2],
                 [1, 2, 1]])
    
    T4=np.array([[2, 2, 2],
                 [1, 1, 1],
                 [2, 1, 2]])
    
    T5=np.array([[2, 2, 1], 
                 [2, 1, 2],
                 [1, 2, 1]])
    
    T6=np.array([[2, 1, 2],
                 [2, 1, 1],
                 [2, 1, 2]])
    
    T7=np.array([[1, 2, 1],
                 [2, 1, 2],
                 [2, 2, 1]])
    T.append(T0)
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)
    
    #Y like
    Y=[]
    Y0=np.array([[1, 0, 1], 
                 [0, 1, 0], 
                 [2, 1, 2]])
    
    Y1=np.array([[0, 1, 0], 
                 [1, 1, 2], 
                 [0, 2, 1]])
    
    Y2=np.array([[1, 0, 2], 
                 [0, 1, 1], 
                 [1, 0, 2]])
    
    Y2=np.array([[1, 0, 2], 
                 [0, 1, 1], 
                 [1, 0, 2]])
    
    Y3=np.array([[0, 2, 1], 
                 [1, 1, 2], 
                 [0, 1, 0]])
    
    Y4=np.array([[2, 1, 2], 
                 [0, 1, 0], 
                 [1, 0, 1]])
    Y5=np.rot90(Y3)
    Y6 = np.rot90(Y4)
    Y7 = np.rot90(Y5)
    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)
    
    bp = np.zeros(skel.shape, dtype=int)
    # for x in X:
    #     bp = bp + mh.morph.hitmiss(skel,x)
    for y in Y:
        bp = bp + mh.morph.hitmiss(skel,y)
    for t in T:
        bp = bp + mh.morph.hitmiss(skel,t)
    return bp

def endPoints(skel):

    """
    The endpoints function was written by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699

    """
    
    endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])
    
    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])
    
    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])
    
    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])
    
    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])
    
    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])
    
    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])
    
    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])
    
    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7)
    ep8=mh.morph.hitmiss(skel,endpoint8)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    ep = ep1+ep3+ep5+ep7 # removing some that I think might not be interesting. MUCH BETTER!
    return ep

def pruning(skeleton, size):

    """
    The prunning function was written by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699

    """

    '''remove iteratively end points "size" 
       times from the skeleton
    '''
    for i in range(0, size):
        endpoints = endPoints(skeleton)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton,endpoints)
    return skeleton

def readcounts(files,path):
    counts=[]
    # def makecount(f,data)

    for f in files:
        with open(os.path.join(path,f),'r') as fr:
            data = fr.read()
            # print data
            counts.append(data.split('\t'))
            
    return counts

def getqsub():
    res=subprocess.check_output(['qstat' ,'-u' ,'mexposito' ] ) 
    res=[x for x in res.split("\n")]
    qwait=len(res)
    return qwait