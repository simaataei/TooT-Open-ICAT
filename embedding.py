
from model import SubstrateRepresentation
from Data_preparation_DeepIon import X_train, y_train, X_test, y_test
import torch 
import torch.nn as nn
from torch.nn import functional as F




model = SubstrateRepresentation().cuda()
model.eval()
sequence_output_train= []

with torch.no_grad():
  for sample in X_train:
    sequence_output_train.append(model(sample).tolist())

with open('Dataset/DeepIon_emb_train.txt', 'w') as file:
    for item in sequence_output_train:
        file.write(f"{item}\n") 




'''
sequence_output_val= []

with torch.no_grad():
  for sample in X_val:
    sequence_output_val.append(model(sample).tolist())

with open('Dataset/bin_emb_val.txt', 'w') as file:
    for item in sequence_output_val:
        file.write(f"{item}\n")

'''

sequence_output_test= []

with torch.no_grad():
  for sample in X_test:
    sequence_output_test.append(model(sample).tolist())

with open('Dataset/DeepIon_emb_test.txt', 'w') as file:
    for item in sequence_output_test:
        file.write(f"{item}\n")





