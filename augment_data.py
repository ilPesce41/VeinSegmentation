from skimage.transform import rotate
import numba
from numba import prange
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from cv2 import imwrite, imread
from cv2 import resize
import os
from multiprocessing import Pool, freeze_support
from tqdm import tqdm

def elastic_transform(args):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

       <script src="https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a.js"></script>
    """
    image, alpha, sigma, random_state = args
    tmp = image.copy()
    # assert len(image.shape)==2
    if random_state is None:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(seed=random_state)

    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    for i in range(tmp.shape[2]):
        tmp[:,:,i] = map_coordinates(image[:,:,i], indices, order=1).reshape(shape)
    return tmp

@numba.jit
def flip(img):
    tmp = img.copy()
    for i in prange(tmp.shape[0]):
        for j in prange(tmp.shape[1]):
            img[i,j] = tmp[j,i]
    return img


def random_rotation(files,images):
    """
    Apply random rotation to set of images (Same rotation for all of them)
    """
    angle = (np.random.random()*360)-180
    rot_images = [rotate(x,angle) for x in images]
    for i,im in enumerate(rot_images):
        im = im * 255
        imwrite(files[i],im.astype(int))

@numba.jit
def apply_wrap(image,dx,dy):
    tmp = image.copy()
    xi=0
    yi=0
    for x in prange(image.shape[0]):
        for y in prange(image.shape[1]):
            image[x,y] = tmp[(x+dx)%256,(y+dy)%256]
    return image

def random_wrap(files):
    """
    Apply random rotation to set of images (Same rotation for all of them)
    """
    dx = np.random.randint(0,256)
    dy = np.random.randint(0,256)
    # dx,dy=200,200
    images = [imread(x) for x in files]

    for i in range(len(images)):
        images[i] = resize(images[i],(256,256))
    x_lim,y_lim = 256,256
    wrap_images = [apply_wrap(x,dx,dy) for x in images]
    # wrap_images = [x for x in images]
    return wrap_images


def transform_pairs(args):
    images, out_ims, alpha, sigma, random_state = args
    images = [(imread(x),alpha,sigma,random_state) for x in images]
    # p = Pool(3)
    outputs = map(elastic_transform,images)
    for x,y in zip(outputs,out_ims):
        imwrite(y,x)

def get_images(dir):
    ims = os.listdir(dir)
    ims = [os.path.join(dir,x) for x in ims]
    return ims

def apply_rotwrap(pair):
    ims = random_wrap(pair)
    random_rotation(pair,ims)

if __name__ == "__main__":
    freeze_support()
    dirs = ['toh_highlight_images', 'toh_selected_images', 'toh_vessel_images']
    
    out_dir = 'toh_output'
    os.makedirs(out_dir,exist_ok=True)
    inputs = []
    p2 = Pool(10)
    pairs = []
    for idx in tqdm(range(2000,20000)):
        files = map(get_images,dirs)
        for images in zip(*files):
            out_ims = [os.path.join(out_dir,"s_{}_out_".format(idx)+os.path.split(x)[-1]) for x in images]
            pairs.append(tuple(out_ims))
            inputs.append(tuple([images,out_ims,3000,30,idx]))
            if len(inputs)%10==0:
                p2.map(transform_pairs,inputs)
                inputs = []
    p2.map(transform_pairs,inputs)

    subs = []
    p = Pool(10)
    for pair in tqdm(pairs):
        subs.append(pair)
        if len(subs)%10==0:
            p.map(apply_rotwrap,subs)
            subs=[]
    p.map(apply_rotwrap,subs)
   
    