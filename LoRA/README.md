# LoRA : Low-Rank Adaptation of Large Language Models

paper: https://arxiv.org/pdf/2106.09685.pdf

In the paper, it is applied to large language models but it can be applied to any model.

I have done the experiment with a simple model trained on MNIST, classify digits. 


## Summary of the LoRA algorithm: 

Let's assume our deep learning network has only one weight matrix W, so our model is:

`output = Wx + b`  

Now we want to fine tune to improve the result or adapt it to a new task. Let's say our `W` shape is `(d,k)`. 

Problems:
- `W` has a large amount of parameters. 
- We want to train multiple versions of the network for different tasks, therefore we need save `W` weights for each task. It will take a lot of storage. And deploying multiple network with this size is not efficient. 

Solution:

LoRA algorithm:
Let's create two new matrix `A` and `B` such that their shape is:    
`A = (r,k)`  
`B = (d,r)`

where `r` is rank and `r < min(d, k)`.

so when we multiply ` B `and `A `we will get the same shape as `W, (d,k)`

Now our model with LoRA becomes: 

`output = (W + BA)x + b`

And during fine tuning we only train `B` and `A` matrix weights and freeze `W` and `b`. 

### LoRA help:
- It decreases the number of parameters we need to train. 
- And we can only save LoRA weights for different tasks and load as required. This reduces the compute required for deployment. 





## Other math concepts to read about which are related to LoRA: 
- Single Value Decomposition (SVD)