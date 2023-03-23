#https://github.com/Clar1ty1/ts1001
#pip install opencv-python
#pip install gdown

import skimage as sk
import numpy as np
from matplotlib import pyplot
import cv2




cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret: break # break if no next frame
    
    cv2.imshow(frame) # show frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # on press of q break
        break

def mexhat(x,y,s):
    return 1/(np.pi*s**4)*(1-1/2*(x**2+y**2)/s**2)*np.exp(-(x**2+y**2)/(2*s**2))

def gauss2d(x,y,s):
    return np.exp(-(x**2+y**2)/(2*s**2))/(2*np.pi*s**2)

def genGaussKernel(n, size,s, fnc):
    x=np.linspace(-n,n,size)
    y=np.linspace(-n,n,size)

    g=np.zeros((size,size))

    for i,xi in enumerate(x):
        for j,yi in enumerate(y):
            g[i,j]=fnc(xi,yi,s)
    
    return g