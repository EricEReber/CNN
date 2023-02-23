import numpy as np
import imageio.v3 as imageio 
import matplotlib.pyplot as plt
from alt_backprog_conv import _pad_2d_channel 

def _extract_windows(imgs_batch, fil_size):
    '''
    imgs_batch: [batch_size, channels, img_width, img_height]
    fil_size: int
    '''
    # pad the images
    imgs_batch_pad = np.pad(imgs_batch, ((0,0),(0,0),(fil_size//2,fil_size//2),(fil_size//2,fil_size//2)), mode='constant')

    # get all patches using numpy's stride_tricks
    batch_size, channels, img_width, img_height = imgs_batch_pad.shape

    patch_shape = (batch_size, channels, fil_size, fil_size, img_width-fil_size+1, img_height-fil_size+1)

    patch_strides = (channels*img_width*img_height, img_width*img_height, img_width, 1, img_height, 1)

    patches = np.lib.stride_tricks.as_strided(imgs_batch_pad, shape=patch_shape, strides=patch_strides)

    # reshape and return the patches
    patches = patches.transpose(4,5,0,1,2,3)
    return patches.reshape(-1, batch_size, channels, fil_size, fil_size)

def _feedforward(batch, kernel):

    windows = _extract_windows(X, kernel.shape[2])
    windows = windows.transpose(1, 0, 2, 3, 4).reshape(batch.shape[0]) 

    kernel = kernel.transpose(0, 2, 3, 1).reshape(kernel.shape[0]*kernel.shape[1]*kernel.shape[2], -1) 

    output = (windows@kernel).reshape(batch.shape[0], batch.shape[2], batch.shape[3], -1)

    # The output is reshaped and rearranged to appropriate shape
    return output.transpose(0, 3, 1, 2)

def _backpropagate(batch, kernel, output_grad):

    # Computing the kernel gradient
    windows = _extract_windows(batch, kernel.shape[2]).reshape(batch.shape[0] * batch.shape[2] * batch.shape[3], -1)
    output_grad_tr = output_grad.transpose(0, 2, 3, 1).reshape(batch.shape[0] * batch.shape[2] * batch.shape[3], -1)

    kernel_grad = (windows.T@output_grad_tr).reshape(kernel.shape[0], kernel.shape[2], kernel.shape[3], kernel.shape[1])
    kernel_grad = kernel_grad.transpose(0, 3, 1, 2)

    # Computing the input gradient 
    windows = _extract_windows(output_grad, kernel.shape[2]).transpose(1, 0, 2, 3, 4)
    windows = windows.reshape(batch.shape[0] * batch.shape[2] * batc.shape[3], -1)

    kernel_r = kernel.reshape(batch.shape[1], -1)
    input_grad = (windows@kernel_r.T).reshape(batch.shape[0], batch.shape[2], batch.shape[3], kernel.shape[0])
    input_grad = input_grad.transpose(0, 3, 1, 2) 

    return kernel_grad, input_grad


if __name__ == "__main__":

    kernel_height = 3
    kernel_width = 3 
    out_channels = 3


    # Shape og input is [batch_size, in_channels, img_height, img_width] 

    img_path = "/home/gregz/Files/CNN/data/luna.JPG"
    image = imageio.imread(img_path) 
    images = np.ndarray((1, image.shape[2], image.shape[0], image.shape[1]))
    

    params = np.ndarray((3, out_channels, kernel_height, kernel_width))
    
    for i in range(params.shape[0]): 
        for j in range(params.shape[1]): 
            params[i, j, :, :] = np.random.rand(kernel_height, kernel_width)

    images[0, :, :, :] = image[:,:,:].transpose(2,0,1)
   
    
    out_grad = np.ndarray((1, out_channels, image.shape[0], image.shape[1]))
    
    for i in range(out_grad.shape[0]): 
        for j in range(out_grad.shape[1]): 
            out_grad[i,j, :, :] = np.random.rand(image.shape[0], image.shape[1])
    
    images2 = np.stack([_pad_2d_channel(obs, kernel_height// 2)
                              for obs in images])
    
    import time
    start = time.time() 
    image_patches = _extract_windows(images2, kernel_height) 
    print(f'Time for rewritten method: {time.time() - start}') 


