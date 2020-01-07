import torch
model = torch.load('model.pkl')
print(model.embedder)
print(model.proj)
input()
print(model.embedder.model.state_dict)
input()
print(model.proj.state_dict)