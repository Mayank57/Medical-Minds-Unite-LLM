from transformers import AutoTokenizer, BioGptModel
import torch
import torch.nn as nn
class Ensembler(nn.Module):
  def __init__(self):
    super().__init__()
    self.inputModel = AutoTokenizer.from_pretrained("microsoft/biogpt")
    self.model1 = BioGptModel.from_pretrained("microsoft/biogpt")
    self.model2 = BioGptModel.from_pretrained("microsoft/biogpt")

    self.linearTransform = nn.linear(1024,4)
    self.softmax = nn.Softmax(dim=1)  


  def forward(self,x):

    input = self.inputModel(x, return_tensors="pt")
    y1 = self.model1(**input).last_hidden_state
    y2 = self.model2(**input).last_hidden_state
    y= torch.cat((y1,y2),dim=2)

    print("Y Shape",y.shape)
    ones = torch.ones(y.size(2))

    n,h,w = y.shape

    y_sampled = torch.empty(n,h,w//2)

    for i in range(h):
      indices = torch.multinomial(ones, w//2, replacement=False)
      y_sampled[:,i,:] = y[:,i, indices]
    
    y = self.linearTransform(y_sampled)
    print("After Linear: ",y.shape )
    y = self.softmax(y)
    print("After Softmax: ",y.shape)

    return y


e = Ensembler()
e.train()
output = e.forward("Hello")
print(output.shape)
