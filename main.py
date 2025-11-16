import torch
from mamba2 import Mamba2Simple
from convkan import ConvKAN, LayerNorm2D

from timeit import default_timer as timer

import numpy as np

device = torch.device("cuda")


'''

Idea is to split image into chunks and then 
pass them to Mamba Model, and then it will
generate more tokens which will represents
estimated map.  

Input image will have size of 224 x 224.

'''

head1 = Mamba2Simple(
    d_model=32*32,
    d_state=32,
    d_conv=4,
    expand=2,
    use_mem_eff_path=False
).to(device=device)


torch.save(head1.state_dict(),"mamaba.pt")


def main():
    
    with torch.no_grad():
    
        _input = torch.rand((1,1,32*32)).to(device=device)
        _input1 = torch.rand((1,1,32*32)).to(device=device)
        
        
        start = timer()
        
        out = head1(_input)
        
        end = timer()
        
        print(out.shape)
        print(f"Time in: {end-start} s")
        
        start = timer()
        
        out = head1(_input1)
        
        end = timer()
        
        print(out.shape)
        print(f"Time in: {end-start} s")
        
    input()


if __name__ == "__main__":
    main()