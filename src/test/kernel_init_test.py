
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
print(kernel_tensor[1,0,:,:])