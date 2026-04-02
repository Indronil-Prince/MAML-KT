# MAML
# -----------------------------------------------------------
# MAML for cold-start KT with a GRU base model
# robust support/query split and PAD handling
# -----------------------------------------------------------

import io, copy, random
import numpy as np
from typing import List
import contextlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import higher

def sdp_kernel(Q, K, V, mask=None):
    """
    Simple scaled dot-product attention kernel.
    Q: [B, T, D]
    K: [B, T, D]
    V: [B, T, D]
    mask: [B, T, T] or None
    Returns: [B, T, D]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5  # [B, T, T]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

# ----------------------------
# Parse 3-line blocks
# ----------------------------
def parse_student_data(data_string: str):
    """
    Each student is 3 lines:
        line1: n
        line2: qids (comma/space separated)
        line3: answers (comma/space separated, 0/1)
    Returns list of dicts with 'question_ids', 'answers'
    """
    lines = [ln.strip() for ln in io.StringIO(data_string) if ln.strip()]
    out = []
    i = 0
    while i + 2 < len(lines):
        try:
            n = int(lines[i])
            q = [int(x) for x in lines[i+1].replace(",", " ").split()]
            a = [int(x) for x in lines[i+2].replace(",", " ").split()]
            q, a = q[:n], a[:n]
            if len(q) >= 2 and len(a) >= 2:
                out.append({"question_ids": q, "answers": a})
        except Exception:
            pass
        i += 3
    return out

# ----------------------------
# Dataset with offset IDs (0=PAD)
# ----------------------------
class MetaKTDataset(Dataset):
    def __init__(self, students: List[dict], num_skills: int, min_len: int = 2):
        self.num_skills = num_skills
        self.data = [s for s in students if len(s["question_ids"]) >= min_len]

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        q_ids = s["question_ids"]
        a     = s["answers"]

        # Reserve 0 for PAD; shift questions to 1..num_skills
        q_off = [q + 1 for q in q_ids]

        # Ensure all indices are within valid range for embeddings
        V = (self.num_skills + 1)   # vocab per correctness channel (includes PAD slot)
        max_inter_idx = 2 * (self.num_skills + 1) - 1
        max_exer_idx = self.num_skills

        # Clamp indices to valid range for exercise embedding (1..num_skills)
        q_off = [min(max(q, 1), self.num_skills) for q in q_off]
        # Clamp indices to valid range for interaction embedding (0..max_inter_idx)
        interactions = torch.tensor(
            [min(max(q + a_i * V, 0), max_inter_idx) for q, a_i in zip(q_off, a)],
            dtype=torch.long
        )

        # Next-step setup: predict a[t] from history <= t-1 and q[t]
        data      = interactions[:-1]                         # [T-1]
        targets   = torch.tensor(a[1:], dtype=torch.float)    # [T-1]
        exercises = torch.tensor(q_off[1:], dtype=torch.long) # [T-1]
        return data, targets, exercises

def meta_collate_fn(batch):
    """
    Pads variable-length sequences in batch.
    Returns: [B,T] data, [B,T] targets, [B,T] exercises, [B] lengths
    """
    batch = [b for b in batch if b is not None]
    if not batch: 
        return None, None, None, None
    lens = [len(b[0]) for b in batch]
    T = max(lens)
    pad_data, pad_targ, pad_exer = [], [], []
    for d, t, e in batch:
        padL = T - len(d)
        if padL > 0:
            d = torch.cat([d, torch.zeros(padL, dtype=torch.long)])
            t = torch.cat([t, torch.zeros(padL, dtype=torch.float)])
            e = torch.cat([e, torch.zeros(padL, dtype=torch.long)])
        pad_data.append(d); pad_targ.append(t); pad_exer.append(e)
    return torch.stack(pad_data), torch.stack(pad_targ), torch.stack(pad_exer), torch.tensor(lens, dtype=torch.long)

# ----------------------------
# GRU DKT base model (no SDPA)
# ----------------------------
class DKTGRU(nn.Module):
    def __init__(self, num_skills: int, embed_dim: int = 64, hidden: int = 128):
        """
        Embeddings:
        - interactions vocab: 0..(2*(num_skills+1)-1), 0 is PAD
        - exercises vocab:    0..(num_skills), 0 is PAD; actual questions are 1..num_skills
        """
        super().__init__()
        self.V_inter = 2 * (num_skills + 1)   # includes PAD slot
        self.V_exer  = (num_skills + 1)       # includes PAD slot

        self.inter_embed = nn.Embedding(self.V_inter, embed_dim, padding_idx=0)
        self.exer_embed  = nn.Embedding(self.V_exer,  embed_dim, padding_idx=0)

        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)

        # project exercise embedding (E) to GRU hidden size (H) so we can add
        self.tgt_proj = nn.Linear(embed_dim, hidden)

        self.fc  = nn.Linear(hidden, 1)

    def forward(self, interactions: torch.Tensor, target_exercises: torch.Tensor, lengths: torch.Tensor = None):
        """
        interactions: [B,T] long (0=PAD)
        target_exercises: [B,T] long (0=PAD)
        lengths: [B] true lengths (optional, for pack/pad)
        """
        x   = self.inter_embed(interactions)        # [B,T,E]
        tgt = self.exer_embed(target_exercises)     # [B,T,E]
        tgt_h = self.tgt_proj(tgt)                  # [B,T,H]

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        else:
            h, _ = self.gru(x)                      # [B,T,H]

        logits = self.fc(h + tgt_h).squeeze(-1)     # [B,T]
        return logits

# ----------------------------
# Support/Query split
# ----------------------------
def split_task_one(d, t, e, true_len: int, k_support: int):
    """
    Split single student's sequence into support/query by true length.
    Auto-shrinks k to ensure non-empty query.
    """
    T = int(true_len)
    if T <= 1: return None
    k = min(k_support, T - 1)
    if k <= 0 or (T - k) <= 0: return None
    return (d[:k], t[:k], e[:k]), (d[k:T], t[k:T], e[k:T])

# ----------------------------
# True MAML train epoch (higher)
# ----------------------------
def train_epoch_maml(
    model, train_ds, meta_opt, criterion,
    support_shots=3, meta_batch=8, fast_lr=0.05, device=torch.device("cpu"), inner_steps=2
):
    model.train()
    loader = DataLoader(train_ds, batch_size=meta_batch, shuffle=True, collate_fn=meta_collate_fn)

    for bd, bt, be, blen in loader:
        if bd is None: 
            continue
        bd, bt, be, blen = bd.to(device), bt.to(device), be.to(device), blen.to(device)

        meta_opt.zero_grad()
        meta_loss = 0.0
        valid = 0

        for i in range(bd.size(0)):
            d_i, t_i, e_i, L_i = bd[i], bt[i], be[i], int(blen[i].item())
            parts = split_task_one(d_i, t_i, e_i, L_i, support_shots)
            if not parts: 
                continue
            (sd, st, se), (qd, qt, qe) = parts

            inner_opt = optim.SGD(model.parameters(), lr=fast_lr)
            with higher.innerloop_ctx(
                model, inner_opt, copy_initial_weights=False, track_higher_grads=True
            ) as (fmodel, diffopt):
                fmodel.train()
                for _ in range(inner_steps):
                    s_logits = fmodel(sd.unsqueeze(0), se.unsqueeze(0), torch.tensor([len(sd)], device=device))
                    s_loss = criterion(s_logits, st.unsqueeze(0))
                    diffopt.step(s_loss)

                q_logits = fmodel(qd.unsqueeze(0), qe.unsqueeze(0), torch.tensor([len(qd)], device=device))
                q_loss = criterion(q_logits, qt.unsqueeze(0))
                meta_loss = meta_loss + q_loss
                valid += 1

        if valid == 0: 
            continue
        meta_loss = meta_loss / valid
        meta_loss.backward()
        meta_opt.step()

# ----------------------------
# Eval on unseen students (adapt K, test on rest)
# ----------------------------
def evaluate_cold_start(model, test_ds, criterion, support_shots=3, fast_lr=0.05, device=torch.device("cpu"), inner_steps=2):
    model.eval()
    all_probs, all_true = [], []

    for idx in range(len(test_ds)):
        d, t, e = test_ds[idx]
        if len(d) <= support_shots: 
            continue

        sd, st, se = d[:support_shots], t[:support_shots], e[:support_shots]
        qd, qt, qe = d[support_shots:], t[support_shots:], e[support_shots:]

        # clone and ADAPT with grads enabled
        fmodel = copy.deepcopy(model).to(device)
        inner_opt = optim.SGD(fmodel.parameters(), lr=fast_lr)

        fmodel.train()
        with torch.enable_grad():
            for _ in range(inner_steps):
                s_logits = fmodel(
                    sd.unsqueeze(0).to(device),
                    se.unsqueeze(0).to(device),
                    torch.tensor([len(sd)], device=device),
                )
                s_loss = criterion(s_logits, st.unsqueeze(0).to(device))
                inner_opt.zero_grad(); s_loss.backward(); inner_opt.step()

        # INFER on the query set without grads
        fmodel.eval()
        with torch.no_grad():
            q_logits = fmodel(
                qd.unsqueeze(0).to(device),
                qe.unsqueeze(0).to(device),
                torch.tensor([len(qd)], device=device),
            )
            probs  = torch.sigmoid(q_logits).squeeze(0).detach().cpu().numpy().tolist()
            labels = qt.detach().cpu().numpy().tolist()

        all_probs.extend(probs)
        all_true.extend(labels)

    if not all_probs:
        return 0.5, 0.5

    auc = roc_auc_score(all_true, all_probs)
    acc = accuracy_score(all_true, np.rint(all_probs))
    return float(auc), float(acc)

# ----------------------------
# Main
# ----------------------------
stu = 20  # Number of students
que = 20  # Number of questions

# Ensure all question_ids are within range
def filter_invalid(students, num_skills):
        return [
            s for s in students
            if all(0 <= q < num_skills for q in s["question_ids"])
        ]

if __name__ == "__main__":
 for setid in range(1,6):
  dts = 2015
  print(f"------ASSIST{dts} Dataset------")
  print(f"---- RUN for {setid} Student Set----")
  train_file = 'C:\\Users\\ibpri\\Downloads\\Knowledge Tracing\\DKVMN-No-ID\\DKVMN-main\\dataset\\assist'+str(dts)+'\\Test - '+str(stu)+'\\Set'+str(setid)+'\\assist'+str(dts)+'_train_new_'+str(stu)+'_set'+str(setid)+'.txt'
#     train_file = 'C:\\Users\\ibpri\\Downloads\\Knowledge Tracing\\DKVMN-No-ID\\DKVMN-main\\dataset\\assist2017\\assist2017_train.txt'
  # Load raw files
  with open(train_file, "r", encoding="utf-8") as f: 
        raw_train = f.read()
  train_students = parse_student_data(raw_train)

    # vocab size across both sets (safe)
  all_q = set()
  for s in train_students: all_q.update(s["question_ids"])
  NUM_SKILLS = max(all_q) + 1 if all_q else 1  # questions are 0..max -> +1 distinct skills

  train_students = filter_invalid(train_students, NUM_SKILLS)

    # Need at least SUPPORT+1 steps to make non-empty query
  SUPPORT = 2 # + int(i/4) - 1
  MINLEN  = SUPPORT + 1
  train_students = [s for s in train_students if len(s["question_ids"]) >= MINLEN]
    
  print(f"Meta-train students: {len(train_students)}")
    # Datasets
  train_ds = MetaKTDataset(train_students, NUM_SKILLS, min_len=MINLEN)
   
    # Model/optim/loss
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = DKTGRU(NUM_SKILLS, embed_dim=128, hidden=256).to(device)
  meta_opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
  criterion = nn.BCEWithLogitsLoss()


  # Training configuration
  EPOCHS = 10
  META_BATCH = 16
  FAST_LR = 0.005
  INNER_STEPS = 2

  for ep in range(1, EPOCHS+1):
        train_epoch_maml(
            model, train_ds, meta_opt, criterion,
            support_shots=SUPPORT, meta_batch=META_BATCH,
            fast_lr=FAST_LR, device=device, inner_steps=INNER_STEPS
        )
        if ep % 2 == 0:
            print(f"Completed Meta-Train Epoch {ep:02d}")

  for i in range(4,que+1):
    print(f"---- RUN for {i} questions----")
    # Your exact paths on Windows:
    # train_file = 'C:\\Users\\ibpri\\Downloads\\Knowledge Tracing\\DKVMN-No-ID\\DKVMN-main\\dataset\\assist2009\\Test - '+str(stu)+'\\Set'+str(setid)+'\\assist2009_train_new_'+str(stu)+'_set'+str(setid)+'.txt'
    test_file  = 'C:\\Users\\ibpri\\Downloads\\Knowledge Tracing\\DKVMN-No-ID\\DKVMN-main\\dataset\\assist'+str(dts)+'\\Test - '+str(stu)+'\\Set'+str(setid)+'\\assist'+str(dts)+'_test_new_'+str(i)+'_set'+str(setid)+'.txt'
    # test_file  = 'C:\\Users\\ibpri\\Downloads\\Knowledge Tracing\\DKVMN-No-ID\\DKVMN-main\\dataset\\assist2017\\Test\\assist2017_test_new_'+str(i)+'.txt'
                        
                        
    # Load raw files
    # with open(train_file, "r", encoding="utf-8") as f: 
    #     raw_train = f.read()
    with open(test_file, "r", encoding="utf-8") as f: 
        raw_test  = f.read()

    # train_students = parse_student_data(raw_train)
    test_students  = parse_student_data(raw_test)

    # vocab size across both sets (safe)
    # all_q = set()
    # for s in train_students: all_q.update(s["question_ids"])
    for s in test_students:  all_q.update(s["question_ids"])
    NUM_SKILLS = max(all_q) + 1 if all_q else 1  # questions are 0..max -> +1 distinct skills
    print(f"NUM_SKILLS: {NUM_SKILLS}")

    # # Ensure all question_ids are within range
    # def filter_invalid(students, num_skills):
    #     return [
    #         s for s in students
    #         if all(0 <= q < num_skills for q in s["question_ids"])
    #     ]
    # train_students = filter_invalid(train_students, NUM_SKILLS)
    test_students  = filter_invalid(test_students, NUM_SKILLS)

    # Need at least SUPPORT+1 steps to make non-empty query
    SUPPORT = 2 # + int(i/4) - 1
    MINLEN  = SUPPORT + 1
    # train_students = [s for s in train_students if len(s["question_ids"]) >= MINLEN]
    test_students  = [s for s in test_students  if len(s["question_ids"]) >= MINLEN]
    # print(f"Meta-train students: {len(train_students)}")
    print(f"Meta-test  students: {len(test_students)}")

    # Datasets
    # train_ds = MetaKTDataset(train_students, NUM_SKILLS, min_len=MINLEN)
    test_ds  = MetaKTDataset(test_students,  NUM_SKILLS, min_len=MINLEN)


    # Training configuration
    EPOCHS = 1
    META_BATCH = 16
    FAST_LR = 0.005
    INNER_STEPS = 2

    for ep in range(1, EPOCHS+1):
        # train_epoch_maml(
        #     model, train_ds, meta_opt, criterion,
        #     support_shots=SUPPORT, meta_batch=META_BATCH,
        #     fast_lr=FAST_LR, device=device, inner_steps=INNER_STEPS
        # )
        auc, acc = evaluate_cold_start(
            model, test_ds, criterion,
            support_shots=SUPPORT, fast_lr=FAST_LR, device=device, inner_steps=INNER_STEPS
        )
        print(f"Epoch {ep:02d} | Cold-start AUC {auc:.4f} | ACC {acc:.4f}")
