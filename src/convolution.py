
import numpy as np
import imageio 
import matplotlib.pyplot as plt
import time
from Fourier import *

def padding(img, kernel):

    new_height = img.shape[0] + (kernel.shape[0]//2)*2
    new_width = img.shape[1] + (kernel.shape[1]//2)*2
    k_height = kernel.shape[0]//2

    padded_img = np.zeros((new_height, new_width))

    padded_img[k_height:new_height-k_height, k_height:new_width-k_height] = img[:,:]

    return padded_img


def generate_gauss_mask(sigma ,K=1):
    
    side=np.ceil(1+8*sigma)
    y, x = np.mgrid[-side//2+1:(side//2)+1, -side//2+1:(side//2)+1]
    ker_coef = K/(2*np.pi*sigma**2)
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))

    return g, ker_coef

def conv2D(image, kernel, stride=1, pad='zero'): 

    for i in range(2): 
        kernel = np.rot90(kernel)
    
    # The kernel is quadratic, thus we only need one of its dimensions
    half_dim = kernel.shape[0]//2
   
    if pad == 'zero': 
        conv_image = np.zeros(image.shape)
        pad_image = padding(image, kernel)
    else: 
        conv_image = np.zeros((image.shape[0]-kernel.shape[0], image.shape[1]-kernel.shape[1]))
        pad_image = image[:,:]

    for i in range(half_dim, conv_image.shape[0]+half_dim, stride): 
        for j in range(half_dim, conv_image.shape[1]+half_dim, stride): 
            
            conv_image[i-half_dim, j-half_dim] = np.sum(pad_image[i-half_dim:i+half_dim+1,j-half_dim:j+half_dim+1]*kernel)

    return conv_image


def conv2DSep(image, kernel, coef, stride=1, pad='zero'): 

    for i in range(2): 
        kernel =np.rot90(kernel)

    # The kernel is quadratic, thus we only need one of its dimensions
    half_dim = kernel.shape[0]//2

    ker1 = np.array(kernel[half_dim, :])
    ker2 = np.array(kernel[:, half_dim])
   
    if pad == 'zero': 
        conv_image = np.zeros(image.shape)
        pad_image = padding(image, kernel)
    else: 
        conv_image = np.zeros((image.shape[0]-kernel.shape[0], image.shape[1]-kernel.shape[1]))
        pad_image = image[:,:]

    for i in range(half_dim, conv_image.shape[0]+half_dim, stride): 
        for j in range(half_dim, conv_image.shape[1]+half_dim, stride): 
            
            conv_image[i-half_dim, j-half_dim] = pad_image[i-half_dim:i+half_dim+1,j-half_dim:j+half_dim+1]@ker1@ker2.T*coef
 
    return conv_image


if __name__ == '__main__':

    path = 'C:/Users/gregor.kajda/OneDrive - insidemedia.net/Desktop/private_files/CNN/data/luna.JPG'
    image = imageio.imread(path, as_gray=True)

    gauss, ker = generate_gauss_mask(sigma=1)
    #start_time = time.time()
    #conv_img = conv2DSep(image, gauss, coef=ker, pad='zero')
    #print(time.time() - start_time)

    start_time = time.time() 
    fourier_conv(image, gauss*ker)
    print(time.time() - start_time)

    #fourier_img = np.fft.fft2(image)
    #fourier_img = np.fft.fftshift(fourier_img)

    """
    shifted_img = np.zeros((image.shape[0]*2, image.shape[1]*2))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            shifted_img[i, j] = image[i, j]*((-1)**(i+j))

    plt.imshow(shifted_img, cmap='gray', vmin=0, vmax=255, aspect='auto')
    plt.show()

    fourier_img = np.fft.fft2(shifted_img)

    plt.imshow(fourier_img.real, cmap='gray', vmin=0, vmax=255, aspect='auto')
    plt.show()

    inv_fourier_img = np.fft.ifft2(fourier_img)

    plt.imshow(inv_fourier_img.real, cmap='gray', vmin=0, vmax=255, aspect='auto')
    plt.show()

    shifted_back = np.zeros((image.shape[0]*2, image.shape[1]*2))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            shifted_back[i, j] = shifted_img[i, j]*((-1)**(i+j))

    shifted_back = shifted_back[:image.shape[0], :image.shape[1]]
    plt.imshow(shifted_back, cmap='gray', vmin=0, vmax=255, aspect='auto')
    plt.show()
    """


  
