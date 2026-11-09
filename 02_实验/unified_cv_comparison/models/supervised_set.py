"""Exact set-likelihood training adapter for the existing concatenation policy."""
import torch
from torch import nn
from models.set_backbone import ConcatQNetwork
from models.set_likelihood import selected_subsets,set_nll_from_logits


class SupervisedSet(nn.Module):
    def __init__(self,vocab,k,hidden=256,hidden2=128,max_subsets=4096):
        super().__init__()
        self.k=k;self.vocab=vocab;self.max_subsets=max_subsets
        self.policy=ConcatQNetwork(vocab-1+k,k+1,hidden,hidden2,dropout=0.)

    def encode(self,x):
        # PAD is absent; UNK is one explicit symptom-presence coordinate.
        return torch.nn.functional.one_hot(x,num_classes=self.vocab).any(1)[:,1:].float()

    def forward(self,symptoms,selected):
        return self.policy(torch.cat((symptoms,selected),dim=1))

    def record_loss(self,symptoms,truth):
        target=truth.nonzero(as_tuple=True)[0].tolist()
        subsets=selected_subsets(target,self.k,dtype=symptoms.dtype,
                                 device=symptoms.device,max_subsets=self.max_subsets)
        logits=self(symptoms.expand(len(subsets),-1),subsets)
        return set_nll_from_logits(logits,target,normalization='target_steps',
                                   max_subsets=self.max_subsets)

    def backward_batch(self,x,y):
        symptoms=self.encode(x);total=0.
        # Release each subset graph before processing the next record.
        for s,truth in zip(symptoms,y):
            loss=self.record_loss(s,truth)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite set loss')
            (loss/len(x)).backward();total+=float(loss.detach())
        return total/len(x)

    @torch.no_grad()
    def generate(self,x):
        symptoms=self.encode(x)
        chosen=torch.zeros(len(x),self.k,device=x.device)
        done=torch.zeros(len(x),dtype=torch.bool,device=x.device)
        traces=[[] for _ in range(len(x))]
        for _ in range(self.k+1):
            illegal=torch.cat((chosen.bool(),~chosen.bool().any(1,keepdim=True)),dim=1)
            action=self(symptoms,chosen).masked_fill(illegal,float('-inf')).argmax(1)
            for i in range(len(x)):
                if not done[i]:traces[i].append(int(action[i]))
            selected=~done & action.lt(self.k)
            chosen[selected,action[selected]]=1
            done=done | action.eq(self.k)
            if done.all():break
        return chosen.to(torch.int64),traces
