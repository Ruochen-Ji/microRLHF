# Reward Model Training: Findings

## Setup

- **Model**: GPT-2 small (124M params) + scalar reward head (768 params)
- **Dataset**: Anthropic HH-RLHF, 160,800 train / 8,552 test preference pairs
- **Training**: 3 epochs, batch_size=4, lr=1e-5, AdamW with weight_decay=0.01
- **LR schedule**: Linear warmup (3% of steps) then cosine decay to 0
- **Loss**: Bradley-Terry pairwise ranking loss: `-log(sigmoid(r_chosen - r_rejected))`

## Results

| Metric | Best Value | At Step |
|--------|-----------|---------|
| Best eval accuracy | **65.75%** | 29,500 (mid epoch 1) |
| Best eval loss | 0.6359 | 29,000 |
| Final train accuracy | 80% | 108,500 (end epoch 3) |
| Final train loss | 0.47 | 108,500 |

## Key Observations

### 1. Most learning happens in epoch 1
The model went from 50% (random) to ~60% train accuracy within the first epoch.
Best eval accuracy (65.75%) was achieved at step 29,500 — before epoch 1 even finished.

### 2. Overfitting after epoch 1
The overfitting gap plot shows this clearly:
- **Epoch 1**: Train and eval accuracy track closely (gap near 0)
- **Epoch 2**: Gap grows to ~5% as train acc rises but eval stagnates
- **Epoch 3**: Gap explodes to ~18% — train acc hits 80% while eval drops to 62%

The eval loss plot confirms this: eval loss starts *increasing* after epoch 1 while train
loss continues to decrease — the classic overfitting signature.

### 3. LR schedule worked well for epoch 1
The cosine schedule kept a high learning rate through most of epoch 1 (where learning
actually happened), then decayed. However, the extra epochs with decaying LR didn't help —
they only memorized training data without improving generalization.

### 4. 65-66% is likely the ceiling for GPT-2 small
Human preferences are inherently noisy — annotators disagree with each other ~30% of the time.
A 124M parameter model with 512 token context is limited in how much preference signal it
can extract. Larger models (GPT-2 medium 350M+) and longer context would likely improve this.

## Recommendations for future runs

1. **1 epoch is sufficient** for GPT-2 small on this dataset. Extra epochs overfit.
2. **Larger batch size** (via gradient accumulation) could reduce the noisy gradients
   we observed (grad norms swinging from 0.5 to 60+).
3. **Larger model** (GPT-2 medium) is the most likely path to higher accuracy.
4. **Early stopping** based on eval accuracy would save compute — stop when eval
   hasn't improved for N eval intervals.

## Files

- `training_progress.png` — 4-panel training visualization
- `plot_training.py` — script to regenerate plots from CSV
- `../reward_model_log.csv` — raw training metrics
- `../reward_model.pt` — best checkpoint (65.75% eval acc, step 29,500)
