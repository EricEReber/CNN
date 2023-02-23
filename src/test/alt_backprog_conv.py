import numpy as np 
import imageio.v3 as imageio
import matplotlib.pyplot as plt



def _pad_1d(inp,
            num):
    z = np.array([0])
    z = np.repeat(z, num)
    return np.concatenate([z, inp, z])

def _pad_1d_batch(inp,
                  num):
    outs = [_pad_1d(obs, num) for obs in inp]
    return np.stack(outs)


def _pad_2d_obs(inp, 
                num):
    '''
    Input is a 2 dimensional, square, 2D Tensor
    '''
    inp_pad = _pad_1d_batch(inp, num)

    other = np.zeros((num, inp.shape[0] + num * 2))

    return np.concatenate([other, inp_pad, other])

def _pad_2d(inp,
            num):
    '''
    Input is a 3 dimensional tensor, first dimension batch size
    '''
    outs = [_pad_2d_obs(obs, num) for obs in inp]

    return np.stack(outs)

def _pad_2d_channel(inp, 
                    num):
    '''
    inp has dimension [num_channels, image_width, image_height] 
    '''
    return np.stack([_pad_2d_obs(channel, num) for channel in inp])

def _get_image_patches(imgs_batch,
                       fil_size):
    '''
    imgs_batch: [batch_size, channels, img_width, img_height]
    fil_size: int
    '''
    # pad the images
    imgs_batch_pad = np.stack([_pad_2d_channel(obs, fil_size // 2)
                              for obs in imgs_batch])
    patches = []
    img_height = imgs_batch_pad.shape[2]

    # For each location in the image...
    for h in range(img_height-fil_size+1):
        for w in range(img_height-fil_size+1):

            # ...get an image patch of size [fil_size, fil_size]
            patch = imgs_batch_pad[:, :, h:h+fil_size, w:w+fil_size]
            patches.append(patch)

    # print(f'Length of patches {len(patches)}')
    # Stack, getting an output of size
    # [img_height * img_width, batch_size, n_channels, fil_size, fil_size]
    return np.stack(patches)

def _output_matmul(input_,
                   param):
    '''
    conv_in: [batch_size, in_channels, img_width, img_height]
    param: [in_channels, out_channels, fil_width, fil_height]
    '''

    param_size = param.shape[2]
    batch_size = input_.shape[0]
    img_height = input_.shape[2]
    patch_size = param.shape[0] * param.shape[2] * param.shape[3]

    patches = _get_image_patches(input_, param_size)

    patches_reshaped = (
      patches
      .transpose(1, 0, 2, 3, 4)
      .reshape(batch_size, img_height * img_height, -1)
      )

    param_reshaped = param.transpose(0, 2, 3, 1).reshape(patch_size, -1)

    output = np.matmul(patches_reshaped, param_reshaped)

    output_reshaped = (
      output
      .reshape(batch_size, img_height, img_height, -1)
      .transpose(0, 3, 1, 2)
    )

    return output_reshaped

def _param_grad_matmul(input_,
                       param,
                       output_grad):
    '''
    input_: [batch_size, in_channels, img_width, img_height]
    param: [in_channels, out_channels, fil_width, fil_height]
    output_grad: [batch_size, out_channels, img_width, img_height]
    '''

    param_size = param.shape[2]
    batch_size = input_.shape[0]
    img_size = input_.shape[2] ** 2
    in_channels = input_.shape[1]
    out_channels = output_grad.shape[1]
    patch_size = param.shape[0] * param.shape[2] * param.shape[3]

    patches = _get_image_patches(input_, param_sizes)

    patches_reshaped = (
        patches
        .reshape(batch_size * img_size, -1)
        )

    output_grad_reshaped = (
        output_grad
        .transpose(0, 2, 3, 1)
        .reshape(batch_size * img_size, -1)
    )

    param_reshaped = param.transpose(0, 2, 3, 1).reshape(patch_size, -1)

    param_grad = np.matmul(patches_reshaped.transpose(1, 0),
                           output_grad_reshaped)

    param_grad_reshaped = (
        param_grad
        .reshape(in_channels, param_size, param_size, out_channels)
        .transpose(0, 3, 1, 2)
    )

    return param_grad_reshaped

def _input_grad_matmul(input_,
                       param,
                       output_grad):

    param_size = param.shape[2]
    batch_size = input_.shape[0]
    img_height = input_.shape[2]
    in_channels = input_.shape[1]
    
    output_grad_patches = _get_image_patches(output_grad, param_size)
    
    output_grad_patches_reshaped = (
        output_grad_patches
        .transpose(1, 0, 2, 3, 4)
        .reshape(batch_size * img_height * img_height, -1)
    )
    
    param_reshaped = (
        param
        .reshape(in_channels, -1)
    )
    
    input_grad = np.matmul(output_grad_patches_reshaped,
                           param_reshaped.transpose(1, 0))
    
    input_grad_reshaped = (
        input_grad
        .reshape(batch_size, img_height, img_height, 3)
        .transpose(0, 3, 1, 2)
    )

    return input_grad_reshaped

if __name__ == "__main__": 

    img_path = "/home/gregz/Files/CNN/data/luna.JPG"
    image = imageio.imread(img_path) 
    print(f'{image.shape[0]=}')
    print(f'{image.shape[1]=}')
    print(f'{image.shape[2]=}')
    images = np.ndarray((1, image.shape[2], image.shape[0], image.shape[1]))
    

    params = np.ndarray((3, 12, 3, 3))
    for i in range(params.shape[0]): 
        for j in range(params.shape[1]): 
            params[i, j, :, :] = np.random.rand(3,3)

    images[0, :, :, :] = image[:,:,:].transpose(2,0,1)
   
    
    out_grad = np.ndarray((1, 12, image.shape[0], image.shape[1]))

    for i in range(out_grad.shape[0]): 
        for j in range(out_grad.shape[1]): 
            out_grad[i,j, :, :] = np.random.rand(image.shape[0], image.shape[1])
    
    out_images = _input_grad_matmul(images, params, out_grad)
