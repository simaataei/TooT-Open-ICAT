from Bio import SeqIO
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import re
from model import SubstrateClassifier1
from Data_preparation_rej import X_train, X_test, y_test, y_train

# Load label dictionary
label_dict = {}
with open('./Dataset/Label_name_list_ICAT_uni_ident100_t10') as f:
    data = f.readlines()
    for d in data:
        d = d.split(',')
        label_dict[d[2].strip('\n')] = d[1]
label_dict['1000'] = 'other ICAT (reject)'
label_dict['5000'] = 'other transporters'

print(label_dict)

# Parse sequences and preprocess them
sequence_data = []
for record in SeqIO.parse("./Dataset/Arabidopsis-thalian.fasta", "fasta"):
    seq = str(record.seq)
    seq = re.sub(r'(\w)', r'\1 ', seq)
    seq = re.sub(r"[UZOB]", "X", seq)
    sequence_data.append({"id": record.id, "name": record.name, "sequence": seq,"description":record.description})

# Convert sequence data to a dataframe
df = pd.DataFrame(sequence_data)

# Initialize the first model and get predictions
model_1 = SubstrateClassifier1(2).cuda()
model_1.load_state_dict(torch.load("Prot_bert_bfd_binary_e12"))

with torch.no_grad():
    model_1.eval()
    df['Prediction_1'] = df['sequence'].apply(lambda x: np.argmax(model_1([x]).cpu().detach().numpy()))
# Filter sequences for the second model based on the first model's prediction

df['Pass_First_Model'] = df['Prediction_1'].apply(lambda pred: bool(pred == 1))
df_filtered = df[df['Pass_First_Model']].copy()

# Load the second model for more specific predictions
model_2 = SubstrateClassifier1(12).cuda()
model_2.load_state_dict(torch.load("Prot_bert_bfd_inorganic_up100_focal_loss"))

with torch.no_grad():
    model_2.eval()
    df_filtered['Prediction_2'] = df_filtered['sequence'].apply(lambda x: model_2([x]).cpu().detach().numpy())
# Determine final prediction based on thresholding
df_filtered['Final_Pred'] = df_filtered['Prediction_2'].apply(
    lambda x: np.argmax(x) if np.max(x) > 0.5 else 1000
)

# Combine predictions into the main dataframe
df['Final_Pred'] = df.apply(lambda row: df_filtered.loc[row.name, 'Final_Pred'] if row['Pass_First_Model'] else 5000, axis=1)

# Map predictions to substrate labels
df['Substrate'] = df['Final_Pred'].astype(str).map(label_dict)

# Save the dataframe to a LaTeX file
df_no_details = df.drop(columns=['sequence', 'description', 'name'])

# Save the modified dataframe to a LaTeX file
df_no_details.to_latex('./Results/At_double_all_seq.tex', index=True)
# Create a summary table by counting occurrences of each substrate
summary_df = df['Substrate'].value_counts().reset_index()
summary_df.columns = ['Substrate', 'Count']
summary_df.loc[len(summary_df.index)] = ['Total', len(df)]

# Save the summary table to a LaTeX file
summary_df.to_latex('./Results/At_double.tex', index=False)

