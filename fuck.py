import numpy as np
import torchvision.models.vgg

a = [[1,2,3,4,5,6,7,8,9,10,11,12], [11,22,33,44,55,66,77,88,99,1010,1111,1212]]
b = [[1,2,3,4,5,6,7,8,9,10,11,12], [11,22,33,44,55,66,77,88,99,1010,1111,1212]]
n_a = np.array(a).reshape(-1, 3, 2, 2)
n_b = np.array(b).reshape(-1, 3, 2, 2)
n_a = np.append(n_a, n_b, axis=0)
print(n_a)
