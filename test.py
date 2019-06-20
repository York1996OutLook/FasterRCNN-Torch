import torch
x=[1,1,1,1,1]
x=torch.Tensor(x)
y=x.clone()*2
loss_func=torch.nn.MSELoss()
a=loss_func(x,y)
