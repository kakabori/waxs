import numpy as np
from numpy import pi, sin, cos, tan, exp, sqrt
from matplotlib.pyplot import *

def ccd_to_qspace(img):
    """
    
    """
    px = np.arange(img.shape[0])
    pz = np.arange(img.shape[1])
    rmax = 10
    dr = 0.01
    phimax = 90
    dphi = 0.1
    r = np.linspace(0, 10, rmax/dr+1)
    phi = np.linspace(0, 90, phimax/dphi+1)
    rv, phiv = np.meshgrid(r, phi)
    #rv = rv.flatten()
    #phiv = phiv.flatten()
    img_q = np.zeros(rv.shape)   
    xv = rv * cos(phiv*pi/180)
    zv = rv * sin(phiv*pi/180)
    for index in np.ndindex(img_q.shape):
        a = (np.abs(px-xv[index])).argmin()
        b = (np.abs(pz-zv[index])).argmin()
        img_q[index] = img[a, b]
    return r, phi, img_q
        
image = np.zeros((10, 10))
#image[1,1] = 1
#image[2,2] = 2
image[3,3] = 1
#image[4,4] = 4
figure()
imshow(image, cmap='gray',interpolation='nearest')
r, phi, timg = ccd_to_qspace(image)
figure()
pcolormesh(r, phi, timg, cmap='gray')
#plt.grid(True)
#plt.show(block=False)

#image[0,5] = 1
#image[5,5] = 1
#image[1,0] = 1
