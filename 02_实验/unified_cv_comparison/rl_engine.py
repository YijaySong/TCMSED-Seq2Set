import torch

def f1(pred, truth):
    return 2*(pred*truth).sum(1)/(pred.sum(1)+truth.sum(1)).clamp_min(1)

def rollout(model, x, y, rng, samples=4):
    symptoms=model.encode(x).repeat_interleave(samples,0); truth=y.repeat_interleave(samples,0)
    chosen=torch.zeros_like(truth); active=torch.ones(len(truth),dtype=torch.bool); owners=[]; logs=[]
    for _ in range(model.k+1):
        ix=active.nonzero().flatten()
        if not len(ix): break
        selected=chosen[ix]; illegal=torch.cat((selected.bool(),~selected.bool().any(1,keepdim=True)),1)
        lp=model(symptoms[ix],selected).masked_fill(illegal,-torch.inf).log_softmax(1)
        action=torch.multinomial(lp.exp(),1,generator=rng).squeeze(1); owners.append(ix); logs.append(lp.gather(1,action[:,None]).squeeze(1))
        add=action<model.k; chosen=chosen.clone(); chosen[ix[add],action[add]]=1; active[ix[~add]]=False
    if active.any(): raise RuntimeError('rollout did not terminate')
    owner=torch.cat(owners); reward=f1(chosen,truth)
    with torch.no_grad(): baseline=f1(model.generate(x)[0],y).repeat_interleave(samples)
    advantage=reward[owner]-baseline[owner]
    return -(torch.cat(logs)*advantage).sum()/len(truth)

@torch.no_grad()
def set_distribution(model, x):
    k=model.k
    if k>12: raise ValueError('exact set MAP is limited to K<=12')
    masks=torch.arange(1<<k); bits=((masks[:,None]>>torch.arange(k))&1).float(); symptom=model.encode(x)
    n,states=len(x),len(bits); logits=model(symptom[:,None,:].expand(-1,states,-1).reshape(n*states,-1),bits.repeat(n,1)).reshape(n,states,k+1)
    illegal=torch.cat((bits.bool(),~bits.bool().any(1,keepdim=True)),1); logp=logits.masked_fill(illegal[None],-torch.inf).log_softmax(-1).double()
    alpha=torch.full((n,states),-torch.inf,dtype=torch.float64); alpha[:,0]=0
    for depth in range(1,k+1):
        current=masks[bits.sum(1)==depth]; labels=bits[current].nonzero(as_tuple=True)[1].reshape(-1,depth); previous=current[:,None]^(1<<labels)
        alpha[:,current]=torch.logsumexp(alpha[:,previous]+logp[:,previous,labels],-1)
    return (alpha+logp[:,:,k]).exp(),bits
