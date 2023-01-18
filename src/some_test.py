

import numpy as np
import imageio 
import matplotlib.pyplot as plt
import time

def DFT_2D(img): 
    # Slow naive implementation of DFT/FFT
    N, M = img.shape

    fourier = np.zeros((img.shape), dtype=np.complex128)

    for u in range(N): 
        for v in range(M): 
            for x in range(N):
                for y in range(M): 

                    fourier[u,v] += (img[x, y] * (np.cos( ( 2 * np.pi * ( (u*x / N) + (v*y / M) ) ) ) 
                                              - ( 1j*np.sin ( (2*np.pi*( (u*x/N) + (v*y/M) ) ) ) ) ) )
    return fourier


def IDFT_2D(fourier_img):
        # Slow naive implementation of DFT/FFT
    N, M = fourier_img.shape

    inv_fourier_img = np.zeros((fourier_img.shape), dtype=np.complex128)

    for u in range(N): 
        for v in range(M): 
            for x in range(N):
                for y in range(M): 

                    fourier_img[u, v] += inv_fourier_img[x, y]*np.exp(-2j*np.pi*((u*x)/N + (v*y)/M))

    return fourier_img

def FFT_2D(img): 
    
    N,M = img.shape # In the case of a grayscale image with only one color channel 
    
    if N == 2:
        return DFT_2D(img)

    # We divide the image into even and odd parts
    else: 
        # Recursively divide until a base case of M = 2 or N = 2 is reached
        F_even_even = FFT_2D(img[0::2, 0::2])
        F_even_odd = FFT_2D(img[0::2, 1::2])
        F_odd_even = FFT_2D(img[1::2, 0::2])
        F_odd_odd = FFT_2D(img[1::2, 1::2])

        W_u = np.zeros((N), dtype=np.complex128)
        for i in range(N): 
            W_u[i] = (np.cos((2*np.pi*i)/N) 
                        - (1j*np.sin((2*np.pi*i)/N)) )


        W_v = W_u.T# = np.zeros((N), dtype=np.complex128)
        #for k in range(N):
         #   W_v[k] = (np.cos((2*np.pi*k)/N) 
          #              - (1j*np.sin((2*np.pi*k)/N)) )

        W_uv = np.zeros((N, N), dtype=np.complex128)
        for u in range(N): 
            for v in range(N): 
                W_uv[i] = (np.cos((2*np.pi*(u+v)/N)) 
                        - (1j*np.sin((2*np.pi*(u+v)/N))))
        

        F_img_u_v = (F_even_even + (F_even_odd*W_v[:N//2]) + (F_odd_even*W_u[:N//2]) + (F_odd_odd*W_uv[:N//2, :N//2]))
        F_img_Mu_v = (F_even_even + (F_even_odd*W_v[:N//2]) + (F_odd_even*W_u[N//2:]) + (F_odd_odd*W_uv[N//2:, :N//2]))
        F_img_u_Nv = (F_even_even + (F_even_odd*W_v[N//2:]) + (F_odd_even*W_u[:N//2]) + (F_odd_odd*W_uv[:N//2, N//2:]))
        F_img_Mu_Nv = (F_even_even + (F_even_odd*W_v[N//2:]) + (F_odd_even*W_u[N//2:]) + (F_odd_odd*W_uv[N//2 :, N//2 :]))

        img1 = np.hstack((F_img_u_v, F_img_u_Nv))
        img2 = np.hstack((F_img_Mu_v, F_img_Mu_Nv))
        fourier_img = np.vstack((img1, img2))

    return fourier_img


def IFFT_2D(fourier_img): 

    M, N = image.shape

    comp_img = FFT_2D(np.conjugate(fourier_img)*(N**2))

    inv_img = np.conjugate(comp_img)

    return inv_img.real

    # we divide the image further into quarters until a base case of dimension size 2 is reached
    #print('Done')
    #return np.zeros((3,3))