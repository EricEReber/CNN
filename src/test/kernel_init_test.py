
import numpy as np 

kernel_tensor = np.ndarray(
                            (
                            3, 64, 
                            3, 3,
                            )
                        )

for i in range(kernel_tensor.shape[0]): 
    for j in range(kernel_tensor.shape[1]): 
        kernel_tensor[i,j,:,:] = np.random.rand(3, 3)

print(kernel_tensor[0,0,:,:])

#for img in range(kernel_tensor.shape[3]): 
#        for ch in range(kernel_tensor.shape[2]): 
#            kernel_tensor[:,:, ch, img] = np.where(kernel_tensor[:,:, ch, img] > np.ones(kernel_tensor[:,:, ch, img].shape)*0.8, kernel_tensor[:,:, ch, img], np.zeros(kernel_tensor[:,:, ch, img].shape))

#print(kernel_tensor[0,0,:,:])

def RELU(x: np.ndarray):
    return np.where(x > np.zeros(x.shape), x, np.zeros(x.shape))

kernel_tensor = RELU(kernel_tensor)

flat_tensor = kernel_tensor.reshape(-1, kernel_tensor.shape[3])

#print(flat_tensor[:, 0])

kernel_back = flat_tensor.reshape(kernel_tensor.shape)

print(kernel_back[0,0,:,:])