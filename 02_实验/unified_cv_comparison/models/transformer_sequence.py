"""Conditional label Transformer with a permutation-invariant symptom encoder."""
import torch
from torch import nn
from torch.nn import functional as F

from models.gru_sequence import random_targets


class SymptomTransformerSequence(nn.Module):
    def __init__(self,vocab,k,embedding=128,hidden=256,heads=4,layers=2,
                 feedforward=512,dropout=.15):
        super().__init__()
        self.k=k
        self.symptoms=nn.Embedding(vocab,embedding,padding_idx=0)
        self.context=nn.Sequential(nn.Linear(embedding,hidden),nn.Tanh())
        self.labels=nn.Embedding(k+2,embedding)
        self.label_projection=nn.Linear(embedding,hidden)
        self.positions=nn.Embedding(k+1,hidden)
        self.layers=nn.ModuleList([
            nn.TransformerDecoderLayer(hidden,heads,feedforward,dropout,
                                       activation='gelu',batch_first=True,norm_first=True)
            for _ in range(layers)])
        self.norm=nn.LayerNorm(hidden)
        self.dropout=nn.Dropout(dropout)
        self.head=nn.Linear(hidden,k+1)

    def encode(self,x):
        mask=x.ne(0).unsqueeze(-1)
        mean=(self.symptoms(x)*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.context(mean)

    def logits(self,context,prefix):
        """Each position predicts the next action using only its causal prefix."""
        length=prefix.shape[1]
        h=self.label_projection(self.labels(prefix))
        h=h+self.positions(torch.arange(length,device=prefix.device))[None]
        causal=torch.ones(length,length,dtype=torch.bool,device=prefix.device).triu(1)
        for layer in self.layers:
            h=layer(h,context.unsqueeze(1),tgt_mask=causal)
        logits=self.head(self.dropout(self.norm(h)))
        selected=F.one_hot(prefix,num_classes=self.k+2)[...,:self.k].cumsum(1).bool()
        illegal=torch.cat((selected,~selected.any(-1,keepdim=True)),dim=-1)
        return logits.masked_fill(illegal,float('-inf'))

    def loss(self,x,targets):
        bos=torch.full((len(x),1),self.k+1,dtype=torch.long,device=x.device)
        previous=targets[:,:-1].masked_fill(targets[:,:-1].eq(-100),self.k)
        prefix=torch.cat((bos,previous),dim=1)
        logits=self.logits(self.encode(x),prefix)
        ce=F.cross_entropy(logits.transpose(1,2),targets,ignore_index=-100,reduction='none')
        # A full label set has only STOP available, so its terminal loss is zero.
        full=F.one_hot(prefix,num_classes=self.k+2)[...,:self.k].cumsum(1).bool().all(-1)
        active=targets.ne(-100) & ~full
        return ((ce*active).sum(1)/active.sum(1).clamp_min(1)).mean()

    @torch.no_grad()
    def generate(self,x):
        context=self.encode(x)
        prefix=torch.full((len(x),1),self.k+1,dtype=torch.long,device=x.device)
        chosen=torch.zeros(len(x),self.k,dtype=torch.bool,device=x.device)
        done=torch.zeros(len(x),dtype=torch.bool,device=x.device)
        traces=[[] for _ in range(len(x))]
        for _ in range(self.k+1):
            action=self.logits(context,prefix)[:,-1].argmax(-1)
            for i in range(len(x)):
                if not done[i]:traces[i].append(int(action[i]))
            selected=~done & action.lt(self.k)
            chosen[selected,action[selected]]=True
            done=done | action.eq(self.k)
            if done.all():break
            action=torch.where(done,torch.full_like(action,self.k),action)
            prefix=torch.cat((prefix,action[:,None]),dim=1)
        return chosen.to(torch.int64),traces
