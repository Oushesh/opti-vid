## This serves a simple pipeline.
## alternatively consult rapids.ai
## Cupy--> CUDA code level code with Dask is awsome

## 10x faster on multi-gpu hardware


#Can be written and tested in jupyterlab.

import numpy as np
## This could be any detector function here.
## TODO: perform inference on GPU.
def gather_from_detector():
    return np.random.random((1000,1000))

def smooth(x):
    out = np.empty_like(x)
    for i in range(1,x.shape([0])-1):
        for j in range(1,x.shape[1]-1):


#TODO: complete
