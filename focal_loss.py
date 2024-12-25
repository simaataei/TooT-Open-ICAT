import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
from Data_preparation_rej import X_train, X_val, y_train, y_val
from model import SubstrateClassifier
from sklearn.utils.class_weight import compute_class_weight
import torch.optim as optim



# Define the Focal Loss function
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

model = SubstrateClassifier(max(y_train)+1).cuda()
'''
# Compute class weights based on the frequency of classes
class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(np.array(y_train)), y = np.array(y_train))

class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

'''


unique_values, frequencies = np.unique(y_train, return_counts=True)
class_weights = [max(frequencies)/i for i in frequencies]
class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()
# Use Focal Loss instead of CrossEntropyLoss
loss_function = FocalLoss(gamma=2.0, alpha=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.000005)
num_epochs = 10

for epoch in range(1, num_epochs+1):
    all_loss = []
    start = time.time()
    for i in tqdm(range(len(X_train))):
        optimizer.zero_grad()

        sample = X_train[i]
        pred = model(sample).float()
        gold = torch.tensor([y_train[i]], dtype=torch.long).cuda()
        loss = loss_function(pred, gold)
        loss.backward()
        all_loss.append(loss.cpu().detach().numpy())
        optimizer.step()


# Save the model after the training loop
torch.save(model.state_dict(), "./Prot_bert_bfd_inorganic_up100_focal_loss_frequency")

