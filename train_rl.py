#!/usr/bin/env python3
"""
Luanti RL 训练 — 纯 PyTorch 实现 PPO & DQN

依赖: torch, numpy, gymnasium  (无需 stable-baselines3)

使用方法:
  1. 启动 Luanti, 加载含 agent_bridge mod 的世界
  2. 训练 PPO:  python3 train_rl.py --algo ppo --total-steps 50000
  3. 训练 DQN:  python3 train_rl.py --algo dqn --total-steps 50000
  4. 评估:      python3 train_rl.py --eval --resume models/ppo_final.pt
  5. 随机基线:   python3 train_rl.py --algo random --total-steps 2000

模型保存在 models/ , 训练日志在 logs/rl/
"""

import argparse
import csv
import logging
import math
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger("TrainRL")


# ════════════════════════════════════════════════
#  网络
# ════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """PPO Actor-Critic (共享特征层)"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

        # 正交初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        feat = self.shared(x)
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def act(self, obs_t):
        """单步采样: 返回 (action, log_prob, value)"""
        with torch.no_grad():
            logits, value = self(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs_t, actions_t):
        """批量评估: 返回 (log_probs, entropy, values)"""
        logits, values = self(obs_t)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions_t), dist.entropy(), values


class QNetwork(nn.Module):
    """DQN Q-Network (Dueling 可选)"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════
#  经验缓冲区
# ════════════════════════════════════════════════

class RolloutBuffer:
    """PPO 在线回合缓冲"""

    def __init__(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.values, self.log_probs = [], [], []

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_gae(self, last_value, gamma, lam):
        """计算 GAE 优势估计 + 回报"""
        n = len(self.rewards)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            nv = last_value if t == n - 1 else self.values[t + 1]
            nt = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * nv * nt - self.values[t]
            gae = delta + gamma * lam * nt * gae
            adv[t] = gae
        returns = adv + np.array(self.values, dtype=np.float32)
        return adv, returns

    def tensors(self):
        return (
            torch.FloatTensor(np.array(self.obs)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.log_probs),
        )

    def clear(self):
        self.obs.clear(); self.actions.clear(); self.rewards.clear()
        self.dones.clear(); self.values.clear(); self.log_probs.clear()

    def __len__(self):
        return len(self.obs)


class ReplayBuffer:
    """DQN 经验回放"""

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        o, a, r, o2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(o)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(o2)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


# ════════════════════════════════════════════════
#  CSV 日志
# ════════════════════════════════════════════════

class CSVLogger:
    def __init__(self, path, fields):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()

    def log(self, **kw):
        self.w.writerow(kw)
        self.f.flush()

    def close(self):
        self.f.close()


# ════════════════════════════════════════════════
#  PPO 训练
# ════════════════════════════════════════════════

def train_ppo(env, args):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = ActorCritic(obs_dim, act_dim, args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    buf = RolloutBuffer()

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"恢复模型: {args.resume}")

    log = CSVLogger(
        os.path.join(args.log_dir, "ppo_log.csv"),
        ["step", "episode", "reward", "length", "policy_loss",
         "value_loss", "entropy", "fps"],
    )

    obs, _ = env.reset()
    total_steps = 0
    ep_reward, ep_steps, ep_count = 0.0, 0, 0
    best_reward = -float("inf")

    print(f"\n{'='*55}")
    print(f"  PPO 训练")
    print(f"  观测: {obs_dim}  动作: {act_dim}  隐藏: {args.hidden}")
    print(f"  步数: {args.total_steps:,}  Rollout: {args.rollout_steps}")
    print(f"  学习率: {args.lr}  Clip: {args.clip_range}")
    print(f"{'='*55}\n")
    print("等待 Luanti 连接…\n")

    t0 = time.time()

    while total_steps < args.total_steps:
        # ── 收集 rollout ──
        buf.clear()
        rollout_start = time.time()

        for _ in range(args.rollout_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action, lp, val = model.act(obs_t)

            next_obs, reward, done, trunc, info = env.step(action)
            buf.add(obs, action, reward, done or trunc, val, lp)

            obs = next_obs
            total_steps += 1
            ep_reward += reward
            ep_steps += 1

            if done or trunc:
                ep_count += 1
                fps = ep_steps / max(time.time() - rollout_start, 0.01)
                print(f"  [Step {total_steps:>6}] Ep {ep_count}: "
                      f"R={ep_reward:+.1f}  len={ep_steps}  "
                      f"fps={fps:.2f}")
                log.log(step=total_steps, episode=ep_count,
                        reward=round(ep_reward, 2), length=ep_steps,
                        policy_loss="", value_loss="", entropy="",
                        fps=round(fps, 2))

                if ep_reward > best_reward:
                    best_reward = ep_reward
                    torch.save(model.state_dict(),
                               os.path.join(args.save_dir, "ppo_best.pt"))

                ep_reward, ep_steps = 0.0, 0
                obs, _ = env.reset()

        # ── 计算 GAE ──
        with torch.no_grad():
            _, last_val = model(torch.FloatTensor(obs).unsqueeze(0))
            last_val = last_val.item()

        adv, rets = buf.compute_gae(last_val, args.gamma, args.gae_lambda)
        obs_t, act_t, old_lp_t = buf.tensors()
        adv_t = torch.FloatTensor(adv)
        ret_t = torch.FloatTensor(rets)

        # 优势归一化
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ── PPO 更新 ──
        pl_sum, vl_sum, ent_sum, n_updates = 0, 0, 0, 0

        for _ in range(args.n_epochs):
            idx = torch.randperm(len(buf))
            for start in range(0, len(buf), args.batch_size):
                end = min(start + args.batch_size, len(buf))
                b = idx[start:end]

                new_lp, entropy, values = model.evaluate(obs_t[b], act_t[b])
                ratio = torch.exp(new_lp - old_lp_t[b])

                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1 - args.clip_range,
                                    1 + args.clip_range) * adv_t[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret_t[b])
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + args.vf_coef * value_loss
                        + args.ent_coef * entropy_loss)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()

                pl_sum += policy_loss.item()
                vl_sum += value_loss.item()
                ent_sum += entropy.mean().item()
                n_updates += 1

        if n_updates > 0:
            logger.info(
                f"  PPO update: ploss={pl_sum/n_updates:.4f} "
                f"vloss={vl_sum/n_updates:.4f} "
                f"ent={ent_sum/n_updates:.4f}")

        # 定期保存
        if total_steps % args.save_freq < args.rollout_steps:
            p = os.path.join(args.save_dir, f"ppo_{total_steps}.pt")
            torch.save(model.state_dict(), p)
            print(f"  💾 Checkpoint: {p}")

    # 最终保存
    p = os.path.join(args.save_dir, "ppo_final.pt")
    torch.save(model.state_dict(), p)
    elapsed = time.time() - t0
    print(f"\n训练完成: {total_steps} steps, {ep_count} episodes, "
          f"{elapsed/60:.1f} min")
    print(f"模型: {p}")
    log.close()


# ════════════════════════════════════════════════
#  DQN 训练
# ════════════════════════════════════════════════

def train_dqn(env, args):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    q_net = QNetwork(obs_dim, act_dim, args.hidden)
    target_net = QNetwork(obs_dim, act_dim, args.hidden)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    opt = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size)

    if args.resume:
        q_net.load_state_dict(torch.load(args.resume, map_location="cpu"))
        target_net.load_state_dict(q_net.state_dict())
        print(f"恢复模型: {args.resume}")

    log = CSVLogger(
        os.path.join(args.log_dir, "dqn_log.csv"),
        ["step", "episode", "reward", "length", "loss", "epsilon", "fps"],
    )

    obs, _ = env.reset()
    total_steps = 0
    ep_reward, ep_steps, ep_count = 0.0, 0, 0
    best_reward = -float("inf")

    print(f"\n{'='*55}")
    print(f"  DQN 训练")
    print(f"  观测: {obs_dim}  动作: {act_dim}  隐藏: {args.hidden}")
    print(f"  步数: {args.total_steps:,}  Buffer: {args.buffer_size}")
    print(f"  学习率: {args.lr}  Target更新: {args.target_update}")
    print(f"{'='*55}\n")
    print("等待 Luanti 连接…\n")

    t0 = time.time()

    while total_steps < args.total_steps:
        # ε-greedy
        eps = args.eps_end + (args.eps_start - args.eps_end) * \
              max(0, 1 - total_steps / args.eps_decay)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = q_net(torch.FloatTensor(obs).unsqueeze(0))
                action = q.argmax(1).item()

        next_obs, reward, done, trunc, info = env.step(action)
        replay.add(obs, action, reward, next_obs, float(done or trunc))

        obs = next_obs
        total_steps += 1
        ep_reward += reward
        ep_steps += 1

        if done or trunc:
            ep_count += 1
            print(f"  [Step {total_steps:>6}] Ep {ep_count}: "
                  f"R={ep_reward:+.1f}  len={ep_steps}  ε={eps:.3f}")
            log.log(step=total_steps, episode=ep_count,
                    reward=round(ep_reward, 2), length=ep_steps,
                    loss="", epsilon=round(eps, 3), fps="")

            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save(q_net.state_dict(),
                           os.path.join(args.save_dir, "dqn_best.pt"))

            ep_reward, ep_steps = 0.0, 0
            obs, _ = env.reset()

        # ── DQN 更新 ──
        if len(replay) >= args.learning_starts:
            o, a, r, o2, d = replay.sample(args.batch_size)

            with torch.no_grad():
                next_q = target_net(o2).max(1)[0]
                target_q = r + args.gamma * next_q * (1 - d)

            current_q = q_net(o).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(current_q, target_q)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
            opt.step()

            # Target 网络更新
            if total_steps % args.target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

        # 定期保存
        if total_steps % args.save_freq == 0:
            p = os.path.join(args.save_dir, f"dqn_{total_steps}.pt")
            torch.save(q_net.state_dict(), p)
            print(f"  💾 Checkpoint: {p}")

    p = os.path.join(args.save_dir, "dqn_final.pt")
    torch.save(q_net.state_dict(), p)
    elapsed = time.time() - t0
    print(f"\n训练完成: {total_steps} steps, {ep_count} episodes, "
          f"{elapsed/60:.1f} min")
    print(f"模型: {p}")
    log.close()


# ════════════════════════════════════════════════
#  随机基线
# ════════════════════════════════════════════════

def train_random(env, args):
    print("随机基线测试…")
    obs, _ = env.reset()
    total_steps, ep_count, ep_reward, ep_steps = 0, 0, 0.0, 0
    rewards = []

    while total_steps < args.total_steps:
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_steps += 1
        ep_reward += reward
        ep_steps += 1

        if done or trunc:
            ep_count += 1
            rewards.append(ep_reward)
            print(f"  Ep {ep_count}: R={ep_reward:+.1f}  len={ep_steps}")
            ep_reward, ep_steps = 0.0, 0
            obs, _ = env.reset()

    if rewards:
        print(f"\n随机基线: {len(rewards)} eps, "
              f"均值={np.mean(rewards):.1f}, "
              f"最大={np.max(rewards):.1f}, "
              f"最小={np.min(rewards):.1f}")


# ════════════════════════════════════════════════
#  评估
# ════════════════════════════════════════════════

def evaluate(env, args):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    if args.algo == "ppo":
        model = ActorCritic(obs_dim, act_dim, args.hidden)
    else:
        model = QNetwork(obs_dim, act_dim, args.hidden)

    model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    model.eval()
    print(f"评估模型: {args.resume}  ({args.algo.upper()})")

    obs, _ = env.reset()
    ep_count, ep_reward = 0, 0.0
    all_rewards = []

    try:
        while True:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                if args.algo == "ppo":
                    logits, _ = model(obs_t)
                    action = logits.argmax(1).item()
                else:
                    action = model(obs_t).argmax(1).item()

            act_name = env.ACTION_NAMES[action]
            obs, reward, done, trunc, info = env.step(action)
            ep_reward += reward

            if done or trunc:
                ep_count += 1
                all_rewards.append(ep_reward)
                print(f"  Ep {ep_count}: R={ep_reward:+.1f}  "
                      f"action={act_name}")
                ep_reward = 0.0
                obs, _ = env.reset()

    except KeyboardInterrupt:
        if all_rewards:
            print(f"\n评估: {len(all_rewards)} eps, "
                  f"均值={np.mean(all_rewards):.1f}")


# ════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Luanti RL 训练 (纯 PyTorch PPO/DQN)")

    p.add_argument("--algo", choices=["ppo", "dqn", "random"], default="ppo")
    p.add_argument("--total-steps", type=int, default=50_000)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="localhost")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--save-dir", default="models")
    p.add_argument("--log-dir", default="logs/rl")
    p.add_argument("--save-freq", type=int, default=2000)
    p.add_argument("--resume", default=None)
    p.add_argument("--eval", action="store_true")

    # PPO
    p.add_argument("--rollout-steps", type=int, default=64,
                   help="每次更新收集的步数 (PPO)")
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # DQN
    p.add_argument("--buffer-size", type=int, default=10_000)
    p.add_argument("--learning-starts", type=int, default=100)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=5000)
    p.add_argument("--target-update", type=int, default=200)

    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    from gym_env import LuantiGymEnv
    env = LuantiGymEnv(
        host=args.host, port=args.port, max_steps=args.max_episode_steps)

    if args.eval:
        if not args.resume:
            print("评估模式需要 --resume 模型路径")
            sys.exit(1)
        evaluate(env, args)
    elif args.algo == "ppo":
        train_ppo(env, args)
    elif args.algo == "dqn":
        train_dqn(env, args)
    else:
        train_random(env, args)

    env.close()


if __name__ == "__main__":
    main()
