import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from VGG_block_module import VGG_NET

def compare_initializations(device,rand_seed=42):
    # 1.load data
    torch.manual_seed(rand_seed)
    X = torch.randn((64,3,32,32),device=device)
    Y = torch.randint(0,10,(64,),device=device)
    results = {}
    for init_method in ['default','kaiming']:
        vgg_net = VGG_NET(10,init_method).to(device)
        opt = torch.optim.SGD(vgg_net.parameters(),lr=1e-2)

        # 2.start training
        vgg_net.train()
        Y_pred = vgg_net(X)
        loss = F.cross_entropy(Y_pred,Y)
        loss.backward()

        grad_norms = []
        for p in vgg_net.parameters(): # the iterator will return all the parameters of each layers(one layer each time)
            if p.grad is not None:
                grad_norms.append(p.grad.cpu().norm().item()) # get the L2 norm of the gradient
        avg_grad = np.mean(grad_norms)
        results[init_method] = {
            'initial_loss':loss.item(),
            'avg_grad_norm(L2)':avg_grad,
        }
        print(f"\n[{init_method.upper()}]")
        print(f"  Initial Loss: {loss.cpu().item():.4f}")
        print(f"  Avg Grad Norm: {avg_grad:.4f}")
    return results

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compare_initializations(device)