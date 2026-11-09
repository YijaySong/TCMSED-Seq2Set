"""Symptom-token CNN; canonical token order is not a clinical sequence."""
import torch
from torch import nn


class SymptomCNN(nn.Module):
    def __init__(self, vocabulary, labels, embedding=128, channels=128, dropout=.15):
        super().__init__()
        self.embedding=nn.Embedding(vocabulary,embedding,padding_idx=0)
        self.convs=nn.ModuleList(nn.Conv1d(embedding,channels,k,padding=k//2) for k in (1,3,5))
        self.dropout=nn.Dropout(dropout)
        self.head=nn.Linear(3*channels,labels)

    def forward(self,tokens):
        mask=tokens!=0
        if not mask.any(1).all():
            raise ValueError('Each record needs at least one symptom token')
        embedded=self.embedding(tokens).transpose(1,2)
        features=[conv(embedded).relu().masked_fill(~mask[:,None,:],-torch.inf).amax(2)
                  for conv in self.convs]
        return self.head(self.dropout(torch.cat(features,1)))
