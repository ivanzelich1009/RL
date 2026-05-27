import math
import random
import wandb
from tqdm import tqdm
from os.path import join
from os import makedirs
import numpy as np
import torch
from collections import deque

def get_curr_lr(n_update, lr_decay, warmup, max_lr, min_lr, total_updates):
    """
    Calculates the current learning rate based on the update step, learning rate decay schedule,
    warmup period, and other parameters.

    Parameters:
    n_update (int): The current update step (1-indexed).
    lr_decay (str): The type of learning rate decay to apply ("linear" or "cosine").
    warmup (float): The fraction of total updates to be used for the learning rate warmup.
    max_lr (float): The maximum learning rate.
    min_lr (float): The minimum learning rate.
    total_updates (int): The total number of updates.

    Returns:
    float: The current learning rate.

    Raises:
    NotImplementedError: If an unsupported lr_decay type is provided.
    """
    # Convert to 0-indexed for internal calculations
    n_update -= 1
    total_updates -= 1

    # Calculate the end of the warmup period
    warmup_period_end = total_updates * warmup

    if warmup_period_end > 0 and n_update <= warmup_period_end:
        lrnow = max_lr * n_update / warmup_period_end
    else:
        if lr_decay == "linear":
            slope = (max_lr - min_lr) / (warmup_period_end - total_updates)
            intercept = max_lr - slope * warmup_period_end
            lrnow = slope * n_update + intercept

        elif lr_decay == "cosine":
            cosine_arg = (
                (n_update - warmup_period_end)
                / (total_updates - warmup_period_end)
                * math.pi
            )
            lrnow = min_lr + (max_lr - min_lr) * (1 + math.cos(cosine_arg)) / 2

        else:
            raise NotImplementedError(
                "Only 'linear' and 'cosine' lr-schedules are available."
            )

    return lrnow

def training_loop(
        envs,
        args,
        device,
        optimizer,
        agent
):
    obs = torch.zeros((args['num_steps'], args['num_envs']) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args['num_steps'], args['num_envs']) + envs.single_action_space.shape).to(device)
    dones = torch.zeros((args['num_steps'], args['num_envs'])).to(device) 
    rewards = torch.zeros((args['num_steps'], args['num_envs'])).to(device)
    in_rewards = torch.zeros((args['num_steps'], args['num_envs'])).to(device)
    values = torch.zeros((args['num_steps'], args['num_envs'])).to(device)
    logprobs = torch.zeros((args['num_steps'], args['num_envs'])).to(device)
    episodic_returns = np.zeros((args['num_envs'],))
    episodic_length = np.zeros((args['num_envs'],))
    returns_queue = deque([0], maxlen=25)
    lengths_queue = deque([0], maxlen=25)
    if agent.name=='TRPO':
        distributions=torch.zeros((args['num_steps'], args['num_envs']) + envs.single_action_space.shape).to(device)

    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args['num_envs']).to(device)
    num_updates = args['total_timesteps'] // args['batch_size'] #batch_size=num_envs*num_steps, so total_timesteps=num_envs*num_steps*num_updates
    beta = None if args['is_loss_clip'] else args['beta']

    out_dir = f'out/{args["run_name"]}'
    makedirs(out_dir, exist_ok=True)
    
    for update in tqdm(range(1,num_updates+1), desc='Training_progress', total=num_updates):

        random.seed(args['seed'] + update)
        np.random.seed(args['seed'] + update)
        torch.manual_seed(args['seed'] + update)

        if args['anneal_lr']:
            lrnow = get_curr_lr(
                n_update=update,
                lr_decay=args['lr_decay'],
                warmup=args['warmup_period'],
                max_lr=args['learning_rate'],
                min_lr=args['learning_rate'] * args['min_lr_frac'],
                total_updates=num_updates,
            )
            optimizer.param_groups[0]["lr"] = lrnow

        for step in tqdm(range(args['num_steps']), desc=f'Training Rollout at {update}', leave=False):

            #Make a step and record the values
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action,prob,_, value = agent.get_action_value(next_obs)
                if agent.name=='TRPO':
                    old_logits = agent.get_valid_logits(next_obs)
                    distributions[step] = old_logits
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = prob
                if agent.name == 'PPOdistill':
                    in_reward = ((agent.target_reward(next_obs)-agent.pred_reward(next_obs))**2).reshape(-1,)
                    in_rewards[step] = in_reward

            next_obs, ex_reward, done, truncated, _ = envs.step(action.cpu().numpy()) #tensor is taken to cpu and numpy and then step is computed in cpu.
            rewards[step] = torch.as_tensor(ex_reward, dtype=torch.float32, device=device)
            episodic_returns += ex_reward
            episodic_length += 1

            '''Here we will record done of not, and then manually reset the envs. Crucially,
            if a state is reset, it will still be classified as done, but the next_obs will be the reset state. 
            This way we can calculate the advantages correctly, since if an env is done, the next value will not be used in the calculation, 
            and if it's not done, it will be used.
            '''
            _record_info = np.array(
                [
                    True if done[i] or truncated[i] else False for i in range(args['num_envs'])
                ]
            )
            if _record_info.any():
                for i,el in enumerate(_record_info):
                    if el:
                        next_obs[i],_ = envs.envs[i].reset()
                        returns_queue.append(episodic_returns[i])
                        lengths_queue.append(episodic_length[i])
                        episodic_returns[i], episodic_length[i] = 0,0


            next_obs, next_done = torch.as_tensor(next_obs, dtype=torch.float32, device=device), torch.as_tensor(done, dtype=torch.float32, device=device)
            next_obs = torch.nan_to_num(next_obs, nan=0.0)

        returns_array=np.array(returns_queue)
        lengths_array=np.array(lengths_queue)
        
        '''
        We calculate the advantages. Since truncation occurs, we do boostrap if not done.
        If we use distill, normalize in_rewards
        '''

        with torch.no_grad():
            if agent.name == 'PPOdistill':
                agent.obs_rms.update(in_rewards.cpu().numpy())
                reward_std = torch.tensor(agent.obs_rms.var**0.5, dtype=torch.float32, device=device)
                in_rewards = (in_rewards)/(reward_std + 1e-8)
                in_rewards = torch.clamp(in_rewards, 0, 5)
                rewards = rewards + args['distill_coef']*in_rewards
            next_value = agent.get_value(next_obs).reshape(1,-1)
            advantages = torch.zeros_like(rewards)
            last_adv = 0
            for t in range(args['num_steps']-1, -1, -1):
                if t == args['num_steps']-1:
                    nextnonterm = 1.0-next_done #if we're done then ignore next value in calculation, else use it.
                    next_values = next_value
                else:
                    nextnonterm = 1.0-dones[t+1]
                    next_values = values[t+1]
                delta=(rewards[t] + args['gamma']*next_values*nextnonterm - values[t])
                advantages[t] = last_adv = (delta + args['gaelambda'] * args['gamma'] * nextnonterm * last_adv)
            returns = advantages + values
            
        
        #Prepare tensors for batching by flattening.
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_values = values.reshape((-1,))
        b_returns = returns.reshape((-1,))
        b_logprobs = logprobs.reshape((-1,))
        b_advantages = advantages.reshape((-1,))
        b_distributions = distributions.reshape((-1, envs.single_action_space[0])) if agent.name=='TRPO' else None

        b_inds = np.arange(args['batch_size'])

        if agent.name == 'TRPO':

            #Batch value loss
            newvalues = agent.get_value(b_obs)
            if args['clip_vloss']:
                    v_loss_unclipped = (newvalues-b_returns)**2
                    v_clipped = b_values + torch.clamp(newvalues-b_values, -args['clip_coef'], args['clip_coef'])
                    v_loss_clipped = (v_clipped-b_returns)**2
                    v_loss_clipped = torch.max(v_loss_unclipped,v_loss_clipped)
                    v_loss = 0.5*v_loss_clipped.mean()
            else:
                v_loss = ((newvalues-b_returns)**2).mean()
            

            #Train TRPO Policy
            if args['norm_adv']:
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )
            ratio, new_kl = agent.trpo_step(b_distributions, b_obs, b_actions, b_advantages, b_logprobs, max_kl=0.01, damping=0.5)
            optimizer.zero_grad()
            v_loss.backward()
            optimizer.step()
            wandb.log(
                {
                    "losses/improved ratio": ratio.item(),
                    "losses/trpo_kl": new_kl.item(),
                }
            )

        else:
            if agent.name == 'PPO' or agent.name=='PPOdistill':
                clipfracs=[]
            for epoch in range(args['update_epochs']):
                np.random.shuffle(b_inds)
                #Approximate gradient via minibatching
                for start in range(0, args['batch_size'],args['minibatch_size']):
                    end=start + args['minibatch_size']
                    mb_inds = b_inds[start:end]
                    _, newlogprobs, entropy, newvalues = agent.get_action_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds])
                    newvalues.reshape((-1,))

                    #Value loss
                    if args['clip_vloss']:
                        v_loss_unclipped = (newvalues-b_returns[mb_inds])**2
                        v_clipped = b_values[mb_inds] + torch.clamp(newvalues-b_values[mb_inds], -args['clip_coef'], args['clip_coef'])
                        v_loss_clipped = (v_clipped-b_returns[mb_inds])**2
                        v_loss_clipped = torch.max(v_loss_unclipped,v_loss_clipped)
                        v_loss = 0.5*v_loss_clipped.mean()
                    else:
                        v_loss = ((newvalues-b_returns[mb_inds])**2).mean()
                    
                    #Entropy loss
                    entropy_loss = entropy.mean()
                    mb_advantages = b_advantages[mb_inds]
                    if args['norm_adv']:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    if agent.name == 'A2C':
                        pg_loss = (-mb_advantages * newlogprobs).mean()

                    if agent.name == 'PPO' or agent.name=='PPOdistill':
                        logratio = newlogprobs - b_logprobs[mb_inds]
                        ratio = logratio.exp()
                        kl_var = ratio-1-logratio
                        with torch.no_grad():
                            approx_kl = kl_var.mean()
                            #Keep track of the mean of the clipping ratios that exceeded the
                            clipfracs += [((ratio - 1.0).abs() > args['clip_coef']).float().mean().item()]

                        pg1_loss = -mb_advantages * ratio
                        if args['clip_ploss']:
                            pg2_loss = -mb_advantages * torch.clamp(ratio, 1-args['clip_coef'], 1+args['clip_coef'])
                            pg_loss = torch.max(pg2_loss, pg1_loss).mean()
                        else:
                            pg2_loss = beta * kl_var
                            pg_loss = (pg1_loss + pg2_loss).mean()

                    loss = pg_loss - args['entropy_coef']*entropy_loss + args['value_coef']*v_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                '''Beta here is used if we don't use clipped policy loss, and instead use KL-penalty. 
                We update it adaptively based on how the KL is doing compared to the target KL (a hyperparameter).
                If the KL is too high, we increase the penalty, if it's too low, we decrease the penalty. If it's just right, we keep it the same.
                '''
                if agent.name == 'PPO' or agent.name=='PPOdistill':
                    if args['clip_ploss']:  # if clip loss and approx_kl > target kl, break
                        if args['target_kl'] is not None and approx_kl > args['target_kl']:
                            break
                    else:  # if KL-penalty loss, update beta
                        beta = (beta / 2 if approx_kl < args['target_kl'] / 1.5 
                                else (beta * 2 if approx_kl > args['target_kl'] * 1.5 else beta)
                    )
                        
            #Compute the RND exploration loss
            if agent.name == 'PPOdistill':
                np.random.shuffle(b_inds)
                for start in range(0, args['batch_size'],args['minibatch_size']):
                    end=start + args['minibatch_size']
                    mb_inds = b_inds[start:end]
                    exploration_loss = ((agent.target_rewards(b_obs[mb_inds])-agent.pred_rewards(b_obs[mb_inds]))**2).reshape(-1,).mean()
                    agent.rnd_optimizer.zero_grad()
                    exploration_loss.backward()
                    agent.rnd_optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        wandb.log(
                {
                    "rewards/mean_episodic_reward": returns_array.mean(),
                    "rewards/mean_episodic_length": lengths_array.mean(),
                    "losses/value_loss": v_loss.item(),
                    "losses/explained_variance": explained_var,
                    "debug/advantages_mean": b_advantages.mean(),
                    "debug/advantages_std": b_advantages.std(),
                }
            )
        if agent.name == 'A2C':
            wandb.log(
                    {
                        'losses/policy_loss': pg_loss.item(), 
                        'losses/entropy_loss': entropy_loss.item(),
                    }
                )
        if agent.name == 'PPO':
            wandb.log(
                    {
                        'losses/policy_loss': pg_loss.item(), 
                        'losses/entropy_loss': entropy_loss.item(),
                        'losses/approx_kl': approx_kl.item(), 
                        'losses/clipfrac': np.mean(clipfracs),
                    }
                )
        if agent.name == 'PPOdistill':
            wandb.log(
                    {
                        'losses/policy_loss': pg_loss.item(), 
                        'losses/entropy_loss': entropy_loss.item(),
                        'losses/exploration_loss': exploration_loss.item(),
                        'losses/approx_kl': approx_kl.item(), 
                        'losses/clipfrac': np.mean(clipfracs),
                    }
                )
            
        if update > 0 and update%100==0:

            checkpoint = {
                "critic": agent.critic.state_dict(),
                "actor": agent.actor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "config": args,
                "value_loss": v_loss.item(),
            }
            if agent.name == 'A2C':
                checkpoint['policy_loss'] = pg_loss.item()
                checkpoint['entropy_loss'] = entropy_loss.item()
            if agent.name == 'PPO':
                checkpoint['policy_loss'] = pg_loss.item()
                checkpoint['entropy_loss'] = entropy_loss.item()  
                checkpoint['approx_kl'] = approx_kl.item()
                checkpoint['clipfrac'] = np.mean(clipfracs)
            if agent.name=='PPOdistill':
                checkpoint['target_reward_nn'] = agent.target_reward.state_dict()
                checkpoint['pred_reward_nn'] = agent.pred_reward.state_dict()
                checkpoint['RND_optimizer'] = agent.rnd_optimizer.state_dict()
                checkpoint['policy_loss'] = pg_loss.item()
                checkpoint['entropy_loss'] = entropy_loss.item()  
                checkpoint['exploration_loss'] = exploration_loss.item()
                checkpoint['approx_kl'] = approx_kl.item()
                checkpoint['clipfrac'] = np.mean(clipfracs)

            print(f'Saving checkpoint to {out_dir}')
            torch.save(checkpoint, join(out_dir, 'ckpt.pt'))
    return