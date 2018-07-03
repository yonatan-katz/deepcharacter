#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 21:12:55 2018

@author: yonic
"""

import numpy as np
import cv2
import sys
from optparse import OptionParser



def play_file(fname,p1,p2):
    cap = cv2.VideoCapture(fname)   
    
    count = 0
    font = cv2.FONT_HERSHEY_PLAIN
    while(cap.isOpened()):
        ret, frame = cap.read()
    
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        cv2.rectangle(frame, p1, p2, (255,0,0), 5)   
        cv2.putText(frame,str(count),(30,30), font, 2,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        count +=1
    
    if cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Can't open!")



if __name__ == "__main__":   
   parser = OptionParser()
   parser.add_option("-f","--file",dest="file",help="file to read")   
   parser.add_option("--p1",dest="p1",default="10,10",help="upper left point of rectangle: 10,10")
   parser.add_option("--p2",dest="p2",default="800,800",help="lower right point of rectangle: 800,800")
   
   (options, args) = parser.parse_args()    
   if not options.file:
       parser.print_help()
       sys.exit(1)    
       
   s = options.p1.split(",")
   p11 = int(s[0])
   p12 = int(s[1])
   
   s = options.p2.split(",")
   p21 = int(s[0])
   p22 = int(s[1])
   
   play_file(options.file,(p11,p12),(p21,p22))
   
   
   
   
       
   