import numpy as np
import imageio.v3 as imageio 
import matplotlib.pyplot as plt
from alt_backprog_conv import _pad_2d_channel, _output_matmul, _get_image_patches, _param_grad_matmul, _input_grad_matmul


def _padding(batch, kernel_size):

        # TODO: Need fixing to output so the channels are merged back together after padding is finished!

        new_height = batch[0, 0, :, :].shape[0] + (kernel_size // 2) * 2
        new_width = batch[0, 0, :, :].shape[1]  + (kernel_size // 2) * 2
        k_height = kernel_size // 2

        new_tensor = np.ndarray(
            (batch.shape[0], batch.shape[1], new_height, new_width)
        )

        for img in range(batch.shape[0]):

            padded_img = np.zeros(
                    (batch.shape[1], new_height, new_width)
            )
            padded_img[ :, 
                k_height : new_height - k_height, k_height : new_width - k_height
            ] = batch[img, :, :, :]
            new_tensor[img, :, :, :] = padded_img[:, :, :]

        return new_tensor


def _extract_windows(batch, kernel_size, stride=1):
    '''
    imgs_batch: [batch_size, channels, img_width, img_height]
    fil_size: int
    '''
    # pad the images
    batch_pad = _padding(batch, kernel_size)
    
    windows = []
    img_height, img_width = batch.shape[2:]

    # For each location in the image...
    for h in range(kernel_size//2, img_height+1, stride):
        for w in range(kernel_size//2, img_width+1, stride):

            # ...get an image patch of size [fil_size, fil_size]
            window = batch_pad[:, :, h-kernel_size//2:h+kernel_size//2+1, w-kernel_size//2:w+kernel_size//2+1]
            windows.append(window)

    # print(f'Length of patches {len(patches)}')
    # Stack, getting an output of size
    # [img_height * img_width, batch_size, n_channels, fil_size, fil_size]
    return np.stack(windows)


def _feedforward(batch, kernel):

    windows = _extract_windows(batch, kernel.shape[2])
    windows = windows.transpose(1, 0, 2, 3, 4).reshape(batch.shape[0], batch.shape[2] * batch.shape[3], -1)

    kernel = kernel.transpose(0, 2, 3, 1).reshape(kernel.shape[0]*kernel.shape[2]*kernel.shape[3], -1) 

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
    windows = windows.reshape(batch.shape[0] * batch.shape[2] * batch.shape[3], -1)

    kernel_r = kernel.reshape(batch.shape[1], -1)
    input_grad = (windows@kernel_r.T).reshape(batch.shape[0], batch.shape[2], batch.shape[3], kernel.shape[0])
    input_grad = input_grad.transpose(0, 3, 1, 2) 

    return kernel_grad, input_grad


if __name__ == "__main__":
    import time

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
    
    
    start = time.time() 
    image_patches = _extract_windows(images, kernel_height) 
    print(f'Time for rewritten method: {time.time() - start}') 
    
    image_patches2 = _get_image_patches(images, kernel_height)
    
    assert np.allclose(image_patches, image_patches2, atol=10e-6)
    
    # out = _feedforward(images, params)
    # 
    # ker, inp = _backpropagate(images, params, out_grad)
    # ker2 = _param_grad_matmul(images, params, out_grad) 
    # inp2 = _input_grad_matmul(images, params, out_grad)
    #
    # assert np.allclose(ker, ker2, atol=10e-6)
    # assert np.allclose(inp, inp2, atol=10e-6)
