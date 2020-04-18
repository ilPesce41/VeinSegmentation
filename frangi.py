"""
@author: Cole Hill
University of South Florida
Spring 2020
Computer Vision
"""

from scipy.ndimage import gaussian_filter,gaussian_filter1d,sobel
import numpy as np
import cv2
from cv2 import imread,imwrite
import sys
import numba
import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool,freeze_support

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
    Hxy = gaussian_filter(image,sigma,order=2)
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
    
    R = np.abs(l1)/(np.abs(l2)+1e-12)
    S = np.sqrt(l1**2 + l2**2)
    V0_l2pos = np.exp( -(R**2)/(2*beta**2) )
    V0_l2pos *= (1 - np.exp( -(S**2)/(2*c**2) ))
    vesselness = np.where(l2>0,0,V0_l2pos)
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
    vesselness = invert_vesselness(vesselness,image)
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

@numba.jit
def thresh(img,val):
    for i in numba.prange(img.shape[0]):
        for j in numba.prange(img.shape[1]):
            if img[i,j]<val:
                img[i,j] = 0
    return img

@numba.jit
def invert_vesselness(vesselness,image):
    ret = vesselness.copy()
    for i in numba.prange(image.shape[0]):
        for j in numba.prange(image.shape[1]):
            if image[i,j]>0:
                v = vesselness[i,j]
                if v>0.001:
                    ret[i,j]=0
                else:
                    ret[i,j]=1
    return ret

def process_image(img_path,beta,c,sigma,vessel_mask='vesselness.png',highlighted='sub.png'):
    image = imread(img_path)
    image = cv2.resize(image,(image.shape[1]*3,image.shape[0]*3))
    out_im = image.copy()

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = thresh(image,70)
    image = cv2.normalize(image, image, 0, 255,norm_type=cv2.NORM_MINMAX)

    image= image/255
    vesselness, sub = filter_image(image,sigma=sigma,beta=beta,c=c)
    imwrite(vessel_mask,vesselness)


    for i in range(out_im.shape[0]):
        for j in range(out_im.shape[1]):
            if vesselness[i,j]>0 :#and vesselness[i,j]<250:
                out_im[i,j] = [0,255,0]
    imwrite(highlighted,out_im)

def batch_process_image(img_path):
    img_path = img_path[0]
    beta,c,sigma = 1 ,.00001 , 30
    data_set_path = r"C:\Users\chill\Documents\CV Term Project\Images"
    data_set_path = r"C:\Users\chill\Documents\CV Term Project\TopOfHand"
    image = os.path.split(img_path)[-1]
    vessel_path = os.path.join('toh_vessel_images',"v_"+image)
    highlight_path = os.path.join('toh_highlight_images',"h_"+image)
    selected_path  = os.path.join('toh_selected_images',image)

    shutil.copyfile(img_path,selected_path)
    process_image(img_path,beta,c,sigma,vessel_mask=vessel_path,highlighted=highlight_path)


if __name__ == "__main__":

    freeze_support()

    # beta,c,sigma = map(float,sys.argv[1:])
    beta,c,sigma = 1 ,.00006 , 10
    # img_path = r"C:\Users\chill\Documents\CV Term Project\Images\002_l_850_05.jpg"
    # process_image(img_path,beta,c,sigma)

    data_set_path = r"C:\Users\chill\Documents\CV Term Project\Images"
    data_set_path = r"C:\Users\chill\Documents\CV Term Project\TopOfHand"
    images = os.listdir(data_set_path)
    # def file_select(x):
    #     if x.endswith('.jpg') and '_' in x:
    #         if x.split("_")[2]=='850':
    #             return True
    #     return False

    # images = list(filter(file_select,images))
    
    os.makedirs("toh_selected_images",exist_ok=True)
    os.makedirs("toh_vessel_images",exist_ok=True)
    os.makedirs("toh_highlight_images",exist_ok=True)

    p = Pool(5)
    batch_size=5
    batches = []

    

    for image in tqdm(images):
        img_path = os.path.join(data_set_path,image)
        batches.append(tuple([img_path]))

        if len(batches)==batch_size:
            p.map(batch_process_image,batches)
            batches = []
    
    if len(batches)>0:
        p.map(batch_process_image,batches)
