"""Position-free symptom attention with masked mean pooling."""
import torch
from torch import nn


class SymptomTransformer(nn.Module):
    def __init__(self,vocabulary,labels,embedding=128,heads=4,layers=2,feedforward=256,dropout=.15):
        super().__init__()
        self.embedding=nn.Embedding(vocabulary,embedding,padding_idx=0)
        self.layers=nn.ModuleList(nn.TransformerEncoderLayer(
            d_model=embedding,nhead=heads,dim_feedforward=feedforward,
            dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
            for _ in range(layers))
        self.norm=nn.LayerNorm(embedding)
        self.dropout=nn.Dropout(dropout)
        self.head=nn.Linear(embedding,labels)

    def forward(self,tokens):
        valid=tokens!=0
        if not valid.any(1).all():
            raise ValueError('Each record needs at least one symptom token')
        hidden=self.embedding(tokens)
        for layer in self.layers:
            hidden=layer(hidden,src_key_padding_mask=~valid)
        hidden=self.norm(hidden)
        pooled=(hidden*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True)
        return self.head(self.dropout(pooled))
