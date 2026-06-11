# MAML-KT: Model-Agnostic Meta-Learning for Cold-Start Knowledge Tracing

MAML-KT is a meta-learning framework for cold-start knowledge tracing, designed to improve prediction for new students with only a few initial interactions. Instead of training a single global KT model and applying it unchanged to every learner, MAML-KT learns a parameter initialization that can be rapidly adapted to each student with one or a few gradient updates.

This repository contains the code, data processing pipeline, training scripts, and evaluation setup used for our paper on applying Model-Agnostic Meta-Learning (MAML) to Knowledge Tracing (KT) under strict new-student cold-start conditions.

##  Overview

Knowledge Tracing models such as DKT, DKVMN, and SAKT are typically trained with standard empirical risk minimization (ERM). While effective with long student histories, they often struggle in the early phase when a brand-new student first enters the system.

* MAML-KT addresses this by treating each student sequence as a task: *

- a short prefix of interactions is used as the support set
- the remaining interactions form the query set
- the model is adapted on the support set
- performance is optimized on the query set

This allows the learned initialization to specialize quickly to a new student.

## Key Features
- MAML-based few-shot adaptation for KT
- GRU/DKT-style backbone
- strict causal support → query split
- evaluation under new-student cold-start
- experiments on ASSIST2009, ASSIST2015, and ASSIST2017
- support for 10, 20, and 50 held-out new-student cohorts
- critical and moderate cold-start evaluation windows

## Problem Setting

We focus on new-student cold start.

Given a student interaction sequence:

- the first few interactions are used for fast adaptation
- the model then predicts the student’s later responses
- no interaction from held-out test students is seen during training

We evaluate performance in two early regimes:

Critical cold start: Questions 3–10 \
Moderate cold start: Questions 11–15

Repository Structure
```
MAML-KT/
│ 
├── dataset/               # processed and raw dataset files 
├── models/                # model definitions (GRU/DKT backbone, meta-learning modules) 
├── figures/               # final figures used in paper 
├── README.md 
└── requirements.txt
```

## Data Format

This project supports student-sequence data in the common KT format.

3-line format

Each student is represented by 3 lines:

sequence length \
question IDs \
correctness labels

Example:

5 \
12, 15, 15, 20, 21 \
1, 0, 1, 1, 0

## Method

For each student task:

- build a support set from the first K interactions
- build a query set from the remaining interactions
- initialize fast parameters from the shared meta-parameters
- perform one or more inner-loop gradient steps on support loss
- compute query loss using the adapted parameters
- backpropagate through the adaptation step to update the shared initialization

This follows the MAML objective, adapted to sequence-based KT tasks rather than iid tasks.

## Citation

If you use this repository, please cite:

```
@inproceedings{bhattacharjee2026mamlkt,
  title={MAML-KT: Addressing Cold Start in Knowledge Tracing via Few-Shot Meta-Learning},
  author={Bhattacharjee, Indronil and Wayllace, Christabel},
  booktitle={Proceedings of AIED 2026},
  year={2026}
}
```