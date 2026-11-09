"""Symptom-set encoder and conditional autoregressive label decoder."""
import torch
from torch import nn


def random_targets(y, generator):
    rows=[]
    k=y.shape[1]
    for row in y:
        labels=torch.nonzero(row, as_tuple=False).flatten()
        if not len(labels):raise ValueError('Empty reference set')
        rows.append(torch.cat((labels[torch.randperm(len(labels),generator=generator)],torch.tensor([k]))))
    targets=torch.full((len(rows),max(map(len,rows))),-100,dtype=torch.long)
    for i,row in enumerate(rows):targets[i,:len(row)]=row
    return targets


class SymptomGRU(nn.Module):
    def __init__(self,vocab,k,embedding=128,hidden=256,dropout=.15):
        super().__init__()
        self.k=k
        self.symptoms=nn.Embedding(vocab,embedding,padding_idx=0)
        self.context=nn.Sequential(nn.Linear(embedding,hidden),nn.Tanh())
        self.labels=nn.Embedding(k+2,embedding)
        self.cell=nn.GRUCell(embedding+hidden,hidden)
        self.dropout=nn.Dropout(dropout)
        self.head=nn.Linear(hidden,k+1)

    def encode(self,x):
        mask=x.ne(0).unsqueeze(-1)
        mean=(self.symptoms(x)*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.context(mean)

    def step(self,previous,hidden,context,chosen):
        hidden=self.cell(torch.cat((self.labels(previous),context),dim=1),hidden)
        logits=self.head(self.dropout(hidden))
        illegal=torch.cat((chosen,~chosen.any(1,keepdim=True)),dim=1)
        return logits.masked_fill(illegal,float('-inf')),hidden

    def loss(self,x,targets):
        context=self.encode(x);hidden=context
        chosen=torch.zeros((len(x),self.k),dtype=torch.bool,device=x.device)
        previous=torch.full((len(x),),self.k+1,dtype=torch.long,device=x.device)
        sums=context.new_zeros(len(x));counts=context.new_zeros(len(x))
        for t in range(targets.shape[1]):
            logits,hidden=self.step(previous,hidden,context,chosen)
            target=targets[:,t]
            valid=target.ne(-100)
            active=valid & ~chosen.all(1)
            # Full sets terminate deterministically; count only learned decisions.
            ce=nn.functional.cross_entropy(logits,target,ignore_index=-100,reduction='none')
            sums=sums+ce*active;counts=counts+active
            selected=valid & target.lt(self.k)
            chosen=chosen.clone()
            chosen[selected,target[selected]]=True
            previous=torch.where(valid,target,torch.full_like(target,self.k))
        return (sums/counts.clamp_min(1)).mean()

    @torch.no_grad()
    def generate(self,x):
        context=self.encode(x);hidden=context
        chosen=torch.zeros((len(x),self.k),dtype=torch.bool,device=x.device)
        previous=torch.full((len(x),),self.k+1,dtype=torch.long,device=x.device)
        done=torch.zeros(len(x),dtype=torch.bool,device=x.device)
        traces=[[] for _ in range(len(x))]
        for _ in range(self.k+1):
            logits,hidden=self.step(previous,hidden,context,chosen)
            action=logits.argmax(1)
            for i in range(len(x)):
                if not done[i]:traces[i].append(int(action[i]))
            selected=~done & action.lt(self.k)
            chosen[selected,action[selected]]=True
            done=done | action.eq(self.k)
            previous=action
            if done.all():break
        return chosen.to(torch.int64),traces
