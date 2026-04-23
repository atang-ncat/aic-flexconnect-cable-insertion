"""Quick recompute of baselines on the 201-ep v6 split.

Not meant to stay -- temporary scratch for the v6 review.
"""
import random, numpy as np, pyarrow.parquet as pq
from pathlib import Path

parquets = sorted(Path('teleop-dataset/data').rglob('*.parquet'))
action_rows, state_rows, ep_rows = [], [], []
for p in parquets:
    t = pq.read_table(p)
    action_rows.append(np.array([r.as_py() for r in t['action']], dtype=np.float32))
    state_rows.append(np.array([r.as_py() for r in t['observation.state']], dtype=np.float32))
    ep_rows.append(np.array(t['episode_index'].to_pylist(), dtype=np.int64))
action = np.concatenate(action_rows)
state = np.concatenate(state_rows)
ep_idx = np.concatenate(ep_rows)
print(f'Total frames: {len(action)}  episodes: {ep_idx.max()+1}')

rng = random.Random(42)
eps = list(range(int(ep_idx.max())+1))
rng.shuffle(eps)
val_eps = set(eps[:30])
train_mask = np.array([e not in val_eps for e in ep_idx])
val_mask   = ~train_mask
print(f'train frames={train_mask.sum()}  val frames={val_mask.sum()}')

def metrics(pred, tgt):
    err = pred - tgt
    l1 = float(np.mean(np.abs(err)))
    tgt_cls = np.clip(np.round(tgt/0.1),-1,1).astype(int)+1
    pred_cls = np.clip(np.round(pred/0.1),-1,1).astype(int)+1
    acc = float((pred_cls==tgt_cls).mean())
    mmask = tgt_cls!=1
    macc = float((pred_cls==tgt_cls)[mmask].mean()) if mmask.any() else 0.0
    return l1, acc, macc

def ols(X, Y):
    Xb = np.concatenate([X, np.ones((len(X),1))], axis=1)
    W,*_ = np.linalg.lstsq(Xb, Y, rcond=None)
    return W

def olspred(X, W):
    Xb = np.concatenate([X, np.ones((len(X),1))], axis=1)
    return Xb @ W

print()
print('=== 201-ep action distribution ===')
for i,n in enumerate(['lin.x','lin.y','lin.z','ang.x','ang.y','ang.z']):
    u = np.unique(action[:,i])
    pct_zero = (np.abs(action[:,i])<1e-6).mean()
    print(f'  {n:<6s} std={action[:,i].std():.5f}  %zero={pct_zero:.2%}  unique={u.tolist()}')
pct_all_zero = (np.linalg.norm(action,axis=1)<1e-6).mean()
print(f'  %all-zero frames (idle): {pct_all_zero:.2%}')

print()
print('=== Baselines on 30-ep val split ===')
pz = np.zeros_like(action[val_mask])
print(f'zero-pred:       l1={metrics(pz, action[val_mask])[0]:.5f}  acc={metrics(pz, action[val_mask])[1]:.4f}  motion_acc={metrics(pz, action[val_mask])[2]:.4f}')

Wl = ols(state[train_mask], action[train_mask])
pl = olspred(state[val_mask], Wl)
r = metrics(pl, action[val_mask])
print(f'OLS leaky (26D): l1={r[0]:.5f}  acc={r[1]:.4f}  motion_acc={r[2]:.4f}')

clean_idx = [i for i in range(26) if not (7 <= i < 19)]
Xc = state[:, clean_idx]
Wc = ols(Xc[train_mask], action[train_mask])
pc = olspred(Xc[val_mask], Wc)
r = metrics(pc, action[val_mask])
print(f'OLS clean (14D): l1={r[0]:.5f}  acc={r[1]:.4f}  motion_acc={r[2]:.4f}')

prev_a = np.concatenate([np.zeros((1,6), np.float32), action[:-1]], axis=0)
prev_a[np.concatenate([[True], ep_idx[1:]!=ep_idx[:-1]])] = 0.0
Xp = np.concatenate([Xc, prev_a], axis=1)
Wp = ols(Xp[train_mask], action[train_mask])
pp = olspred(Xp[val_mask], Wp)
r = metrics(pp, action[val_mask])
print(f'OLS clean+prev_a: l1={r[0]:.5f}  acc={r[1]:.4f}  motion_acc={r[2]:.4f}')

# prev_action alone
Wa = ols(prev_a[train_mask], action[train_mask])
pa = olspred(prev_a[val_mask], Wa)
r = metrics(pa, action[val_mask])
print(f'OLS prev_a only:  l1={r[0]:.5f}  acc={r[1]:.4f}  motion_acc={r[2]:.4f}')

# majority class per dim
p_maj = np.zeros_like(action[val_mask])  # 0 is majority for every dim
r = metrics(p_maj, action[val_mask])
print(f'majority-class:  l1={r[0]:.5f}  acc={r[1]:.4f}  motion_acc={r[2]:.4f}')
