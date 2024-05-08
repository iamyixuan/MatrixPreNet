import numpy as np
import torch

from NeuralPC.utils.dirac import DDOpt_torch as DDOpt

if __name__ == "__main__":
    # generate data 
    dataPath = "/Users/yixuan.sun/Documents/projects/Preconditioners/datasets/Dirac/precond_data/config.l8-N1600-b2.0-k0.276-unquenched.x.npy"
    U1 = np.load(dataPath)
    U1 = torch.from_numpy(U1).to(torch.cdouble)
    U1 = torch.exp(1j * U1)
    print(U1.shape)

    def gen_x(size):
        real =torch.randn(size, 8, 8, 2)
        imag = torch.randn(size, 8, 8, 2)
        return real + 1j * imag



    x = []
    y =[]

    for j in range(10):
        inputs = gen_x(U1.shape[0])
        x.append(inputs)
        y.append(DDOpt(inputs, U1, 0.276))

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    print(x.shape, y.shape)


    data = {"x": x, "y": y}
    
    import pickle
    with open("./data/linear_inv_data.pkl", "wb") as f:
        pickle.dump(data, f)
    
