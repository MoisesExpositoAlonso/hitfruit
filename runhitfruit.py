#!/usr/bin/env python

'''

hitfruit script for segmentation/skeletonisation of pictures of harvested plants

@author: Moises Exposito-Alonso (moisesexpositoalonso@gmail.com)

'''

##########################################################################################
### Set up
##########################################################################################
import time, datetime, os, sys, pandas,argparse
import cv2
import numpy as np
from PIL import Image
from subprocess import *

# mydirectory=os.getcwd()
# if mydirectory =="/home/moisesexpositoalonso":
#     os.chdir("/home/moisesexpositoalonso/ebio/abt6_projects7/ath_1001G_field/hitfruit")
# print "working in directory ", os.getcwd()

# import my packages
sys.path.append('/home/moisesexpositoalonso/ebio/abt6/mexposito/mpy')
sys.path.append('/home/moisesexpositoalonso/ebio/abt6_projects7/ath_1001G_field/hippo')
sys.path.append('/ebio/abt6/mexposito/mpy')
sys.path.append('/ebio/abt6/mexposito/ebio/abt6_projects7/ath_1001G_field/hippo')
from moi import *
from hippo import *
from hitfruit import *

# start cronometer
cronometer.start()

##########################################################################################
### Read arguments
##########################################################################################

print '\n......................................................................................\n'
print '   hitfruit: script for segmentation/skeletonisation of pictures of harvested plants'
print '   Moises Exposito-Alonso (moisesexpositoalonso@gmail.com)'
print '\n......................................................................................\n'

import argparse
parser = argparse.ArgumentParser(description='hitfruit: segmentation/skeletonisation of harvested pictures')
parser.add_argument("image", help="provide the file name of an image to analyse",
                    type=str)
# parser.add_argument("-a","--analysis", help="provide a flag of analysis [sk=skeletonise/seg=segment]",
#                     type=str,default='sk' , choices=['sk','skeletonise','seg','segment']	)

parser.add_argument("-s","--save", help="provide a logic flag to decide whether to save images",
                    type=str,default='True',choices=['False','True'])

parser.add_argument("-o","--outputpath", help="provide the output path for storing pixel count",
                    type=str,default=''	)


args = parser.parse_args()
print 'Interpreted arguments:'
print '   image file:',args.image
# print '   type of analysis:',args.analysis
print '   save images:',args.save
print '   outputpath:',args.outputpath
print '\n......................................................................................\n'


##########################################################################################
### Generate output folder and names
##########################################################################################

outim=os.path.join(args.outputpath,os.path.basename(args.image)+"_proc")
outcsv=os.path.join(args.outputpath,os.path.basename(args.image)+"_cout")

##########################################################################################
### Run hitfruit
##########################################################################################
print '\nRunning hitfruit ...'

i=hitfruit(args.image).skeleton().countnonzero().endpoints().branchedpoints()

print '  total area', i.count
print '  total skeleton area', i.skcount
print '  branching points',i.branchcount
print '  end points',i.endscount,'\n'

print '... finished hitfruit\n'

### save count value

# write_sv(csvname=outcsv, thelist= [args.image, i.count , i.skcount , i.branchcount , i.endscount ]  )

# table=pandas.DataFrame([args.image, i.count , i.skcount , i.branchcount , i.endscount ], 
# 						columns=['imager','totarea','skarea','branches','ends'])
# table.to_csv(outcsv+'.csv',index=False)

line= [ str(x) for x in  [args.image, i.count , i.skcount , i.branchcount , i.endscount ]  ]
 
target=open(outcsv+'.tsv', 'w')
target.write("\t".join(line) )
target.close()

### save image

if str2bool(args.save)==True:
	print 'saving images'

	saveimagejpeg(image=i.ip, name=outim)
	
	saveimagejpeg(image=i.sk*255, name=outim+"_sk")
	
	bigdot=dilate((i.ends*255).astype(np.float32) ,iterations=5)
	saveimagejpeg(image=i.sk*255+bigdot, name=outim+"_ends")
	
	bigdot=dilate(  (i.branches*255).astype(np.float32)  ,iterations=5)
	saveimagejpeg(image=i.sk*255+bigdot, name=outim+"_branches")
else:
	print 'NOT saving images'

print "job done"  

# stop cronometer
cronometer.stop()
