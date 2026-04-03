#Helper functions to do the mirror descent.
import torch
from torch.distributions import Categorical, kl_divergence

def conjugate_gradient(Avp_fn, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)

    for _ in range(nsteps):
        Avp = Avp_fn(p)
        denom = p.dot(Avp)
        if abs(denom) <= 1e-8:
            break
        alpha = rdotr / denom
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x

def apply_update(actor, flat_params_delta):
    idx = 0
    for param in actor.parameters():
        size = param.numel()
        with torch.no_grad():
            param += flat_params_delta[idx:idx+size].view(param.shape)
        idx += size

def compute_kl(actor, old_dist, states):
    new_logits = actor(states)
    new_dist = Categorical(logits=new_logits)
    kl = kl_divergence(old_dist, new_dist)
    
    return kl.mean()

def compute_surrogate(actor, states, actions, advantages, old_log_probs, max_ratio=10.0):
    new_logits = actor(states)
    new_distribution = Categorical(logits=new_logits)
    new_logprobs = new_distribution.log_prob(actions)
    
    ratio = torch.exp(new_logprobs - old_log_probs)
    
    # Clip ratio for numerical stability
    ratio = torch.clamp(ratio, 0.0, max_ratio)
    
    surrogate = (ratio * advantages).mean()
    return surrogate