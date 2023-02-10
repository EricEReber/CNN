
import numpy as np
import imageio 
import matplotlib.pyplot as plt
import time
import sys 

sys.setrecursionlimit(10**4)



def DFT_2D(img): 
    # Slow naive implementation of DFT/FFT
    N, M = img.shape

    fourier_img = np.zeros((img.shape), dtype=np.complex128)

    for u in range(N): 
        for v in range(M): 
            for x in range(N):
                for y in range(M): 

                    fourier_img[u, v] += img[x, y]*np.exp(-2j*np.pi*((u*x)/N + (v*y)/N))

    return fourier_img

def FFT_2D(img): 
    
    N,M = img.shape # In the case of a grayscale image with only one color channel 
    
    if N == 2:
        return DFT_2D(img)

    # We divide the image into even and odd parts

    even_even = img[0::2, 0::2]
    even_odd = img[0::2, 1::2]
    odd_even = img[1::2, 0::2]
    odd_odd = img[1::2, 1::2]

    # Recursively divide until a base case of M = 2 or N = 2 is reached
    F_even_even = FFT_2D(even_even) 
    F_even_odd = FFT_2D(even_odd)
    F_odd_even = FFT_2D(odd_even)
    F_odd_odd = FFT_2D(odd_odd)

    W_u = np.zeros((N//2))
    for i in range(N//2): 
        W_u[i] = np.exp(-2j*np.pi*i/N)

    W_v = np.zeros((M//2))
    for k in range(M//2):
        W_v[k] = np.exp(-2j*np.pi*k/M)

    W_uv = np.zeros((N//2,M//2))
    for x in range(N//2): 
        for y in range(M//2): 
            W_uv[x,y] = np.exp(-2j*np.pi*(x+y)/N*M)

    F_img_u_v = (1/4)*(F_even_even + (F_even_odd*W_v) + (F_odd_even*W_u) + (F_odd_odd*W_uv))
    F_img_u_Nv = (1/4)*(F_even_even - (F_even_odd*W_v) + (F_odd_even*W_u) - (F_odd_odd*W_uv))
    F_img_Mu_v = (1/4)*(F_even_even + (F_even_odd*W_v) - (F_odd_even*W_u) - (F_odd_odd*W_uv))
    F_img_Mu_Nv = (1/4)*(F_even_even - (F_even_odd*W_v) - (F_odd_even*W_u) + (F_odd_odd*W_uv))

    img1 = np.hstack((F_img_u_v, F_img_Mu_v))
    img2 = np.hstack((F_img_u_Nv, F_img_Mu_Nv))
    fourier_img = np.vstack((img1, img2))

    return fourier_img
    

path = 'C:/Users/gregor.kajda/OneDrive - insidemedia.net/Desktop/private_files/Neural-Network/data/luna.JPG'
image = imageio.imread(path, as_gray=True)

shifted_image = np.zeros((image.shape))
for x in range(image.shape[0]):
    for y in range(image.shape[1]): 
        shifted_image[x,y] = image[x,y]*((-1)**(x+y))

im = FFT_2D(shifted_image)

plt.imshow(im.real, cmap='gray', vmin=0, vmax=255, aspect='auto')
plt.show()

