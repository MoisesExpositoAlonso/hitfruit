#!/usr/bin/env python

'''
Master script to get segmentation of harvesting images

@author: Moises Exposito-Alonso (moisesexpositoalonso@gmail.com)

'''

##########################################################################################
# standard libraries
##########################################################################################
import time, datetime, os, sys
from subprocess import *
import cv2
import numpy as np
import mahotas
from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from skimage.morphology import skeletonize


##########################################################################################
# path
##########################################################################################
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

##########################################################################################
# read index
##########################################################################################

# get the index file

# LOCATION='mad'
LOCATION='tue'


index=pandas.read_csv('../indexharvest/'+LOCATION+'harvest_image_index.csv',error_bad_lines=False).values.tolist()

#example
# index=pandas.read_csv('exampleindex.csv',error_bad_lines=False).values.tolist()  # this hould go as example
# print index

##########################################################################################
# send child processes for inflorescence or for rosette
##########################################################################################


#subset?
# index=index[:50]
# index=index[:50]

counter=1
for line in index:
	## versions without qsub
	## cmdlist=['./runhitfruit.py' , str(line[0]), '--save', str(counter %2 ==0) ] 
	## cmdlist=['./qsub-skeleton.sh' , str(line[0]), '--save', str(counter %2 ==0) ]

	##MADRID
	# cmdlist=['qsub','./qsub-skeleton.sh' , str(line[0]), '--save', str(counter %5 ==0), '--outputpath', '../skeletons/mad' ]
	
	##TUEBINGEN
	cmdlist=['qsub','./qsub-skeleton.sh' , str(line[0]), '--save', str(counter %100 ==0), '--outputpath', '../skeletons/'+LOCATION ]
	
	print " ".join(cmdlist)

	Popen(cmdlist)  # counter %2 is a trick to get only save =True only when counter is divisible by two, so every 2
	
	counter=counter+1


print "MASTER SCRIPT FINISHED !"
