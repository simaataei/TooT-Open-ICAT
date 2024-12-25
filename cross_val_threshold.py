from Data_preparation_rej import  X_train, X_val, X_test, y_train, y_test, y_val
from pycm import *
import pandas as pd
import numpy as np
from model import SubstrateClassifier
import torch
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import time
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



activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.cpu().detach().tolist()
    return hook





val_sf = {}
for i in np.unique(y_train):
    val_sf[i] = []
val_pu = {}
for i in np.unique(y_train):
    val_pu[i] = []
X = X_train + X_val
y = y_train + y_val
kf = KFold(n_splits=3)
threshold_pu = {}
threshold_sf = {}
for j, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {j}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={val_index}")
    threshold_pu[j] = []
    threshold_sf[j] = []
    model = SubstrateClassifier(max(y_train)+1).cuda()

# Compute class weights based on the frequency of classes
    class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(np.array(y_train)), y = np.array(y_train))

    class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()
    model.classifier.register_forward_hook(get_activation('classifier'))
    # Use Focal Loss instead of CrossEntropyLoss
    loss_function = FocalLoss(gamma=2.0, alpha=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.000005)
    num_epochs = 10

    for epoch in range(1, num_epochs+1):
        all_loss = []
        start = time.time()
        for i in tqdm(train_index):
            optimizer.zero_grad()
            sample = X[i]
            pred = model(sample).float()
            gold = torch.tensor([y[i]], dtype=torch.long).cuda()
            loss = loss_function(pred, gold)
            loss.backward()
            all_loss.append(loss.cpu().detach().numpy())
            optimizer.step()
    with torch.no_grad():
         model.eval()
         all_gold = []
         all_pred = []
         for i in val_index:
             sf = model(X[i])
             all_gold.append(y[i])
             prediction = np.argmax(sf.cpu().detach().numpy())
             all_pred.append(prediction)
             if prediction == y[i]:
                 val_pu[prediction].append(activation['classifier'][prediction])
                 val_sf[prediction].append(sf.cpu().detach()[0].tolist()[prediction])
    del model
    for item in val_pu.keys():
        threshold_pu[j].append(min(val_pu[item]))
    for item in val_sf.keys():
        threshold_sf[j].append(min(val_sf[item]))



minimums_pu = [min(threshold_pu[key][i] for key in threshold_pu) for i in range(12)]

with open('./Results/threshold_pu_focal.txt', 'w') as f:
     for item in minimums_pu:
         f.write(str(item)+'\n')

minimums_sf = [min(threshold_sf[key][i] for key in threshold_sf) for i in range(12)]

with open('./Results/threshold_sf_focal.txt', 'w') as f:
     for item in minimums_sf:
         f.write(str(item)+'\n')

'''


with torch.no_grad():
    model.eval()
    all_gold = []
    all_pred = []
    for i in range(len(X_test)):
        pred = model(X_test[i])
        all_gold.append(y_test[i])
        pred = np.argmax(pred.cpu().detach().numpy())
        if activation['classifier'][pred] >thresholds[pred]:
           all_pred.append(pred)
        else:
           all_pred.append(1000)
'''

