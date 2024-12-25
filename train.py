
from torch import optim
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
from Data_preparation_other import X_train_other, X_val_other, y_train_other, y_val_other
from model import SubstrateClassifier3

import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define a function to visualize attention maps
def visualize_attention_map(attention_maps):
    i=0
    # Average attention scores across heads
    for attention_map in attention_maps:		
        avg_attention_map = torch.mean(attention_map, dim=1).squeeze(0).detach().cpu().numpy()

    # Plot the attention heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_attention_map, cmap='viridis', interpolation='nearest')
        plt.xlabel('Input tokens')
        plt.ylabel('Input tokens')
        plt.title('Attention Map')
        plt.colorbar()
        plt.savefig('Results/attention'+str(i)+'.png')
        i+=1



X_train = X_train_other
X_val = X_val_other
y_train = y_train_other
y_val = y_val_other

model = SubstrateClassifier3(max(y_train)+1).cuda()

#loss_function=nn.BCELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.00005)
loss_function =nn.CrossEntropyLoss()
num_epochs = 10 


pred_scores = []
gold_scores = []


all_loss_val = []

total_epochs = [5,10,50,100]
optimizers = [optim.Adam(model.parameters(), lr=0.00005),optim.SGD(model.parameters(), lr=0.00005)]
all_mcc = []
all_f1 = []
all_acc = []
all_rec = []
all_pre = []
loss_min = 100
for op in optimizers:
    for e in total_epochs:
        all_loss=[1000]
        for epoch in range(1,e+1):
            start = time.time()
            for i in tqdm(range(2)):
            #for i in tqdm(range(2)):
                op.zero_grad()
                sample = X_train[i]
                pred, attentions= model(sample)
                
                print(attentions)

                visualize_attention_map(attentions)  # Adjust the index as needed for the desired layer
                pred = torch.tensor([np.argmax(pred.cpu().detach().numpy())], dtype=torch.long).cuda()
                gold = torch.tensor([y_train[i]], dtype=torch.long).cuda()
                loss = loss_function(pred, gold)
                loss.backward()
                op.step()
'''
                if loss.cpu().detach().numpy() - all_loss[i-1] <= 0.001:
                   torch.save(model.state_dict(), "./Prot_bert_bfd_inorganic_up100_other_transporters_e"+str(e)+"_lr"+str(0.00005)+"op"+str(op))
                   break
                else:
                   all_loss.append(loss.cpu().detach().numpy())
            torch.save(model.state_dict(), "./Prot_bert_bfd_inorganic_up100_other_transporters_e"+str(e)+"_lr"+str(0.00005)+"op"+str(op))

    with torch.no_grad():
        model.eval()

        all_gold=list()
        all_pred=list()
        optimizer.zero_grad()
        for j in range(len(X_val)):
            pred = model(X_val[j])
            all_gold.append(y_val[j])
            gold = torch.tensor([y_val[j]],dtype=torch.long).cuda()
            loss = loss_function(pred,gold)
            all_loss_val.append(loss.cpu().detach().numpy())
            prediction = np.argmax(pred.cpu().detach().numpy())
            all_pred.append(prediction)
            if epoch == num_epochs:
                pred_scores.append(pred.cpu().detach().numpy())
                gold_scores.append(gold.cpu().detach().numpy())


    all_mcc.append(matthews_corrcoef(all_gold,all_pred))
    all_rec.append(sklearn.metrics.recall_score(all_gold,all_pred, average = 'macro'))
    all_pre.append(sklearn.metrics.precision_score(all_gold,all_pred, average = 'macro'))
    all_f1.append(sklearn.metrics.f1_score(all_gold,all_pred, average = 'macro'))
    all_acc.append(sklearn.metrics.accuracy_score(all_gold,all_pred))'''
    #torch.save(model.state_dict(), "./Prot_bert_bfd_inorganic_up100")


