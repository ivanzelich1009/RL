import copy
import random
from os import makedirs
import wandb
import collections
from tqdm import tqdm
from os.path import join
import Experiments.Agents.MCTS as MCTS
import numpy as np
import torch.nn.functional as F

import torch


def evaluate(agent, envs, net2, rounds: int, device: torch.device):
    n1_win, n2_win = 0, 0
    mcts_stores = [MCTS.MCTS(), MCTS.MCTS()]

    for _ in range(rounds):
        r, _ = agent.play_game(envs, 
                               mcts_stores=mcts_stores, 
                               replay_buffer=None, 
                               net1=agent, 
                               net2=net2,
                               steps_before_tau_0=0, 
                               mcts_searches=20, 
                               mcts_batch_size=320,
                               device=device)
        if r < -0.5:
            n2_win += 1
        elif r > 0.5:
            n1_win += 1
    return n1_win / (n1_win + n2_win)

def training_loop(
        envs,
        args,
        device,
        optimizer,
        agent
):

    replay_buffer = collections.deque(maxlen=args['replay_buffer'])
    mcts_store = MCTS.MCTS(args['c_puct'])
    best_idx = 0

    best_agent = copy.deepcopy(agent)
    best_agent.eval()  # ensures deterministic inference
    for p in best_agent.parameters():
        p.requires_grad = False  # ensures no gradients

    num_updates = args['total_timesteps'] // args['batch_size']
    out_dir = f'out/{args["run_name"]}'
    makedirs(out_dir, exist_ok=True)

    for update in tqdm(range(1,num_updates+1),desc=f'Rollout phase', total=num_updates):

        game_steps = 0
        prev_nodes=len(mcts_store)
        for _ in range(args['play_episodes']):
            _,steps=agent.play_game(envs,
                            mcts_store, 
                            replay_buffer, 
                            best_agent, 
                            best_agent,
                            steps_before_tau_0 = args['steps_before_tau_0'], 
                            mcts_searches = args['mcts_minibatches'],
                            mcts_batch_size = args['mcts_batch_size'], 
                            device=device)
            game_steps += steps
        game_nodes= len(mcts_store) - prev_nodes
        wandb.log({'charts/steps': game_steps, 'charts/nodes': game_nodes})
        if len(replay_buffer) < args['min_replay_to_train']:
            continue

        for _ in range(args['update_epochs']):
            batch = random.sample(replay_buffer, args['batch_size'])
            batch_states, batch_probs, batch_values = zip(*batch)

            optimizer.zero_grad()
            states_v = torch.as_tensor(np.stack(batch_states), dtype=torch.float32).to(device)
            probs_v = torch.as_tensor(batch_probs, dtype=torch.float32).to(device)
            values_v = torch.as_tensor(batch_values, dtype=torch.float32).to(device)
            out_values_v = agent.critic(states_v)
            out_probs_v = F.log_softmax(agent.get_valid_logits(states_v), dim=-1)

            v_loss = ((out_values_v.squeeze(-1) - values_v)**2).mean()
            pg_loss = (-out_probs_v * probs_v).sum(dim=-1).mean()

            loss = pg_loss + v_loss
            loss.backward()
            optimizer.step()
            

        wandb.log({'losses/value_loss': v_loss.item(), 'losses/policy_loss': pg_loss.item()}, step=update)

        if update >0 and update%100==0:
            win_ratio = evaluate(agent,
                                 envs,
                                 best_agent,
                                 rounds=args['evaluation_rounds'], 
                                 device=device)
            wandb.log({'charts/winratio':win_ratio}, step=update)

            if win_ratio > args['best_win_ratio']:
                    best_agent.load_state_dict(agent.state_dict())
                    best_idx += 1
                    mcts_store.clear()
                    checkpoint={"critic": agent.critic.state_dict(),
                                "actor": agent.actor.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "win_ratio": win_ratio,
                                "value_loss": v_loss,
                                "policy_loss": pg_loss,
                                "update": update,
                                "config": args,
                                "best_id": best_idx
                    }
                    print(f'Saving checkpoint to {out_dir}')
                    torch.save(checkpoint, join(out_dir, f'ckpt_{best_idx}.pt'))
    return