from Data_preparation_rej import  X_train, X_val, X_test, y_train, y_test, y_val
from pycm import *
import pandas as pd
import numpy as np
from model import SubstrateClassifier
import torch


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.cpu().detach().tolist()
    return hook


model = SubstrateClassifier(max(y_train)+1).cuda()
model.load_state_dict(torch.load("./Prot_bert_bfd_inorganic_up100_main"))

model.classifier.register_forward_hook(get_activation('classifier'))

val_pu = {}
for i in np.unique(y_train):
    val_pu[i] = []
with torch.no_grad():
    model.eval()
    all_gold = []
    all_pred = []
    for i in range(len(X_val)):
        pred = model(X_val[i])
        all_gold.append(y_val[i])
        pred = np.argmax(pred.cpu().detach().numpy())
        all_pred.append(pred)
        if pred == y_val[i]:
           val_pu[pred].append(activation['classifier'][pred])


thresholds = []
for item in val_pu.keys():
    thresholds.append(min(val_pu[item]))


with open('./Results/threshold_pu.txt', 'w') as f:
     for item in thresholds:
         f.write(str(item)+'\n')



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

