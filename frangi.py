from scipy.ndimage import gaussian_filter,gaussian_filter1d,sobel
import numpy as np
import cv2
from cv2 import imread,imwrite
import sys

#Blob and structureness parameters as defined in `Frangi-Net: A Neural Network Approach to Vessel Segmentation`
beta = 0.5
c = 1

def build_image_tensor(image,sigma=1):
    """
    Builds image tensor H matrix
    Returns Hxx, Hxy, Hyy
    """
    Hxx = gaussian_filter1d(image,sigma,axis=1,order=2)
    Hyy = gaussian_filter1d(image,sigma,axis=0,order=2)
    Hxy = gaussian_filter(image,sigma,order=1)
    return Hxx, Hxy, Hyy

def calculate_eigs(Hxx,Hxy,Hyy):
    """
    Calculates eigen values for pixel point
    """
    l1 = (Hxx + Hyy) + np.sqrt( (Hxx-Hyy)**2 +4*Hxy**2 )
    l1/=2
    l2 = (Hxx + Hyy) - np.sqrt( (Hxx-Hyy)**2 +4*Hxy**2 )
    l2/=2
    return l1,l2

def calculate_vesselness(l1,l2,beta,c):
    """
    Calculates vesselness value
    """
    vesselness = np.zeros(shape=l1.shape)
    
    R = np.abs(l1)/np.abs(l2)
    S = np.sqrt(l1**2 + l2**2)
    V0_l2pos = np.exp( -(R**2)/(2*beta**2) )
    V0_l2pos *= (1 - np.exp( -(S**2)/(2*c**2) ))
    vesselness = np.where(l2<0,0,V0_l2pos)
    print(np.max(vesselness))
    # vesselness = np.where(l2<0,0,1)
    return vesselness

def filter_image(image,sigma=1,beta=.5,c=1):
    """
    Filters image using Frangi filter
    Returns vesselness image and image minus vesselness image
    """
    tmp_image = image.copy()
    Hxx,Hxy,Hyy = build_image_tensor(image,sigma=sigma)
    l1,l2 = calculate_eigs(Hxx,Hxy,Hyy)
    vesselness = calculate_vesselness(l1,l2,beta,c)
    sub = tmp_image-vesselness
    sub = clean_image(sub)
    vesselness = clean_image(vesselness)
    return vesselness,sub

def clean_image(image):
    """
    Ensures image is in correct format
    """
    image = image*255
    image = image.astype(int)
    image = np.where(image<0,0,image)
    image = np.where(image>255,255,image)
    image = np.where(image==np.nan,0,image)
    image = np.where(image==np.inf,0,image)
    return image

if __name__ == "__main__":

    beta,c,sigma = map(float,sys.argv[1:])
    img_path = r"C:\Users\chill\Documents\CV Term Project\Workspace\IR Image2.png"
    
    image = imread(img_path)
    out_im = image.copy()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # print(image)
    # image = gaussian_filter(image,sigma=1,order=0)
    
    image= image/255
    vesselness, sub = filter_image(image,sigma=sigma,beta=beta,c=c)
    imwrite("vesselness.png",vesselness)

    print(out_im.shape)
    print(vesselness.shape)
    out_im[:,:,1] = vesselness + out_im[:,:,1]
    imwrite("sub.png",out_im)

# 7 .0014 6