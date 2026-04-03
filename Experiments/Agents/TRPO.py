import torch
from torch import nn
from Experiments.Agents.nn_utils import build_network
from torch.distributions import Categorical
from Experiments.Agents.TRPOutils import compute_surrogate, compute_kl, conjugate_gradient, apply_update

class TRPOAgent(nn.Module):
    def __init__(self, env, num_filters = 64):
        super().__init__()
        self.name='TRPO'
        #Warning: For board games the head should be flattened before passing to fully connected linear layers.
        self.critic_head = [('conv2d', 1, {'std': 1}, {'kernel':1 }), 
                    ('flatten',), 
                    ('linear',20, {'std':1}), 
                    ('leaky_relu',), 
                    ('linear',1, {'std':1}), 
                    ('tanh',)]
        self.actor_head = [('conv2d', 1, {'std': 0.01}, {'kernel':1 }), 
                    ('flatten',), 
                    ('linear', env.single_action_space.n)]
        self.CriticSpec = [('conv2d', num_filters, {}, {'kernel': 8, 'stride':4}), 
                           ('conv2d', num_filters, {}, {'kernel': 4, 'stride': 2, 'padding': 1}),
                           ('residual', num_filters),
                           ('residual', num_filters),
                           ('residual', num_filters),
                           *self.critic_head]
        self.ActorSpec = [('conv2d', num_filters, {}, {'kernel': 8, 'stride':4}), 
                           ('conv2d', num_filters, {}, {'kernel': 4, 'stride': 2, 'padding': 1}),
                           ('residual', num_filters),
                           ('residual', num_filters),
                           *self.actor_head]
        self.critic = build_network(env.single_observation_space.shape, self.CriticSpec)
        self.actor = build_network(env.single_observation_space.shape, self.ActorSpec)
    
    def get_value(self, obs):
        return self.critic(obs)
    
    def get_valid_logits(self, obs):
        logits = self.actor(obs)
        return logits
    
    def get_action_value(self, obs, action=None):
        logits = self.get_valid_logits(obs)
        dist = Categorical(logits=logits)
        value = self.critic(obs)
        if action == None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def trpo_step(self, old_logits, states, actions, advantages, old_log_probs, max_kl=0.01, damping=0.1):
    
        orig_params = [p.detach().clone() for p in self.actor.parameters()]
        
        surrogate = compute_surrogate(self.get_valid_logits, states, actions, advantages, old_log_probs)
        grads = torch.autograd.grad(surrogate, self.actor.parameters(), retain_graph=False)
        flat_grad = torch.cat([g.view(-1) for g in grads]).detach()

        with torch.no_grad():
            old_dist = Categorical(logits=old_logits)
        
        def Fvp(v):
            kl = compute_kl(self.get_valid_logits, old_dist, states)
            
            grads_kl = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
            flat_kl_grad = torch.cat([g.contiguous().view(-1) for g in grads_kl])
            kl_v = (flat_kl_grad * v).sum()

            grads_kl_v = torch.autograd.grad(kl_v, self.actor.parameters())
            flat_kl_grad_grad = torch.cat([g.contiguous().view(-1) for g in grads_kl_v]).detach()
            return flat_kl_grad_grad + damping * v  # damping term

        step_dir = conjugate_gradient(Fvp, flat_grad, nsteps=10)

        
        fvp_step = Fvp(step_dir)
        shs = 0.5 * (step_dir * fvp_step).sum()
        
        if shs <= 0:
            print(f"Warning: shs <= 0 ({shs}), skipping update")
            return None, None
        
        step_size = torch.sqrt(2.0 * max_kl / (shs + 1e-8))
        full_step = step_size * step_dir

        surrogate_before = surrogate.detach().item()
        expected_improve = (flat_grad * full_step).sum().item()
        
        if expected_improve <= 0:
            print(f"Warning: expected_improve <= 0 ({expected_improve}), skipping update")
            return None, None
        
        accept = False
        best_ratio = 0
        best_kl = float('inf')

        
        # Try different step size fractions
        for i in range(15):  # More backtracking steps
            fraction = 0.5 ** i
            
            # Reset to original parameters
            for p, orig in zip(self.actor.parameters(), orig_params):
                with torch.no_grad():
                    p.copy_(orig)
            
            # Apply fraction of the full step
            apply_update(self.actor, fraction * full_step)
            
            # Evaluate new policy
            with torch.no_grad():
                new_kl = compute_kl(self.get_valid_logits, old_dist, states).item()
                new_surr = compute_surrogate(self.get_valid_logits, states, actions, advantages, old_log_probs).item()
            
            improve = new_surr - surrogate_before
            
            # Check KL constraint and improvement
            if new_kl <= max_kl * 1.1:  # Allow 10% slack
                if improve > 0:
                    # Accept if improvement is reasonable
                    ratio = improve / (expected_improve * fraction + 1e-8)
                    if ratio > 0.1:
                        accept = True
                        best_ratio = ratio
                        best_kl = new_kl
                        print(f"Accepted step {i}: fraction={fraction:.4f}, KL={new_kl:.6f}, improve={improve:.6f}, ratio={ratio:.3f}")
                        break
                elif i == 0:
                    # If first step doesn't improve, try smaller steps anyway
                    continue
                else:
                    # Accept even small negative improvements if KL is very small
                    if new_kl < max_kl * 0.5 and improve > -0.01 * abs(surrogate_before):
                        accept = True
                        best_ratio = 0.0
                        best_kl = new_kl
                        print(f"Accepted small negative step {i}: fraction={fraction:.4f}, KL={new_kl:.6f}, improve={improve:.6f}")
                        break
        
        # 7. Final cleanup
        if not accept:
            # Revert to original parameters
            for p, orig in zip(self.actor.parameters(), orig_params):
                with torch.no_grad():
                    p.copy_(orig)
            print(f"No acceptable step found. Best KL would be: {best_kl:.6f}")
            return None, None
        
        # Verify final KL constraint
        with torch.no_grad():
            final_kl = compute_kl(self.get_valid_logits, old_dist, states).item()
            if final_kl > max_kl * 1.2:
                print(f"Warning: Final KL ({final_kl:.6f}) exceeds max_kl ({max_kl:.6f})")
        
        return best_ratio, best_kl