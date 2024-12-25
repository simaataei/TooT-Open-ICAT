
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel,AutoTokenizer, AutoConfig

from transformers import BertPreTrainedModel, BertModel, BertTokenizer, BertConfig
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
class SubstrateRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
        self.bert=AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
    def forward(self,input):
        input=self.tokenizer(input,return_tensors="pt", truncation = True, max_length=1024)   
        bert_rep=self.bert(input['input_ids'].cuda())   
        cls_rep=bert_rep.last_hidden_state[0][0]
        return cls_rep


class SubstrateClassifier2(BertPreTrainedModel):
    def __init__(self, num_classes, config_path="Rostlab/prot_bert_bfd"):
        config = BertConfig.from_pretrained(config_path)
        super(SubstrateClassifier2, self).__init__(config)
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained(config_path)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.custom_tokenizer = BertTokenizer.from_pretrained(config_path)

    def forward(self, input):
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        bert_rep = self.bert(input_ids, attention_mask=attention_mask.cuda())
        cls_rep = bert_rep.last_hidden_state[:, 0, :]
        class_scores = self.classifier(cls_rep)
        return F.softmax(class_scores, dim=1)

    def save_model(self, output_dir):
        # Save model weights
        self.save_pretrained(output_dir)

        # Save tokenizer
        self.custom_tokenizer.save_pretrained(output_dir)

    @classmethod
    def load_model(cls, model_path):
        # Load model weights
        model = cls.from_pretrained(model_path)

        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_path)

        # Set the custom tokenizer in the model
        model.custom_tokenizer = tokenizer

        return model

class SubstrateClassifier1(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.config=AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
        self.num_class=num_classes
        self.bert=AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
        self.classifier=nn.Linear(self.config.hidden_size,num_classes)
    def forward(self,input):
        input=self.tokenizer(input,return_tensors="pt", truncation = True, max_length=1024)
        bert_rep=self.bert(input['input_ids'].cuda())
        cls_rep=bert_rep.last_hidden_state[0][0]
        class_scores=self.classifier(cls_rep)
        return F.softmax(class_scores.view(-1, self.num_class), dim=1)



class SubstrateClassifier3(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.config=AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
        self.num_class=num_classes
        self.bert=AutoModel.from_pretrained("Rostlab/prot_bert_bfd", output_attentions=True)
        self.tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
        self.classifier=nn.Linear(self.config.hidden_size,num_classes)
    def forward(self,input):
        input=self.tokenizer(input,return_tensors="pt", truncation = True, max_length=1024)
        bert_rep=self.bert(input['input_ids'].cuda())
        #attention_maps = bert_rep.attentions
        output = self.bert(input['input_ids'].cuda(), return_dict=True, output_attentions=True)
        cls_rep = output.last_hidden_state[0][0]
        class_scores=self.classifier(cls_rep)
        probs = F.softmax(class_scores.view(-1, self.num_class), dim=1)
        return probs, output.attentions
        #cls_rep=bert_rep.last_hidden_state[0][0]
        #class_scores=self.classifier(cls_rep)
        #probs = F.softmax(class_scores.view(-1, self.num_class), dim=1)


class SubstrateClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.config = AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
        self.num_classes = num_classes
        self.bert = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask, labels=None):
        bert_rep = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = bert_rep.last_hidden_state[:, 0]  # Get CLS token representation
        class_scores = self.classifier(cls_rep)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(class_scores, labels)
            return loss
        else:
            return F.softmax(class_scores, dim=1)
class SubstrateRep(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
        self.bert=AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
    def forward(self,input):
        input=self.tokenizer(input,return_tensors="pt", truncation = True, max_length=1024)
        bert_rep=self.bert(input['input_ids'].cuda())
        cls_rep=bert_rep.last_hidden_state[0][0]
        return cls_rep

