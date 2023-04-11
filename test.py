import numpy as np

x = np.array([[1,2], [1, 2]])
try:
    print(np.linalg.inv(x))
except:
    pass
print(np.linalg.pinv(x))