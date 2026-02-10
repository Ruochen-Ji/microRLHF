"""
Reward Model

Learns to predict human preferences using the Bradley-Terry model.

Key concepts:
- Architecture: GPT-2 + scalar output head
- Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
- The reward model simulates human judgment

Phase 4 Implementation:
- RewardModel architecture (GPT backbone + value head)
- Preference pair training
- Evaluation on held-out preferences
"""

import torch
import torch.nn as nn

# TODO: Phase 4 Implementation


class RewardModel(nn.Module):
    """
    Reward model that predicts scalar rewards for text sequences.

    Architecture:
        GPT-2 backbone (frozen or fine-tuned) + scalar output head
    """

    def __init__(self, gpt_model):
        """
        Args:
            gpt_model: Pre-trained GPT model (from model.py).
                       We take its transformer blocks but discard lm_head.
        """
        super().__init__()
        # Keep the transformer backbone (embeddings + blocks + final layernorm)
        self.transformer = gpt_model.transformer
        n_embd = gpt_model.config.n_embd  # 768 for GPT-2 small

        # New scalar head: maps 768-dim hidden state → 1 scalar reward
        self.reward_head = nn.Linear(n_embd, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        """
        Compute reward for input sequence.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding

        Returns:
            rewards: Scalar rewards [batch]
        """
        device = input_ids.device
        b, t = input_ids.size()

        # Run through transformer backbone to get hidden states
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(input_ids)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)         # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # (b, t, n_embd)

        # --- Step 2: Find last real token per example ---
        if attention_mask is not None:
            last_token_idx = attention_mask.sum(dim=1) - 1  # (b,)
        else:
            last_token_idx = torch.full((b,), t - 1, dtype=torch.long, device=device)

        # Index into x: for each example in the batch, grab its last real hidden state
        last_hidden = x[torch.arange(b, device=device), last_token_idx]  # (b, n_embd)

        # --- Step 3: Scalar reward ---
        rewards = self.reward_head(last_hidden).squeeze(-1)  # (b,)
        return rewards


class RewardTrainer:
    """Trainer for reward model using Bradley-Terry loss."""

    def __init__(self, model, lr=1e-5, device="cuda"):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def compute_loss(self, chosen_rewards, rejected_rewards):
        """
        Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))

        Args:
            chosen_rewards: Rewards for chosen responses [batch]
            rejected_rewards: Rewards for rejected responses [batch]

        Returns:
            loss: Scalar loss value
            accuracy: Fraction where chosen > rejected
        """
        diff = chosen_rewards - rejected_rewards
        # F.logsigmoid is numerically stable (avoids computing log(0))
        loss = -nn.functional.logsigmoid(diff).mean()
        accuracy = (diff > 0).float().mean()
        return loss, accuracy

    def train(self, train_dataset, eval_dataset=None, batch_size=4,
              num_epochs=1, log_interval=50, eval_interval=500,
              save_path="rlhf/reward_model.pt"):
        """
        Train reward model on preference pairs.

        LR schedule: linear warmup (3% of steps) then cosine decay to 0.
        - Warmup: LR ramps 0 → peak. At step 0 the reward head is randomly
          initialized, so gradients are huge and noisy. Starting from 0
          prevents those early wild gradients from causing bad updates.
        - Cosine decay: LR follows a half-cosine from peak → 0. This spends
          more time near the peak (where most learning happens) and near zero
          (where it polishes), compared to linear decay which reduces too
          aggressively in the middle. With multiple epochs, earlier epochs
          train with high LR (big learning), later epochs with low LR
          (gentle refinement), so each epoch plays a different role.

        Args:
            train_dataset: PreferenceDataset for training
            eval_dataset: Optional PreferenceDataset for evaluation
            batch_size: Examples per batch
            num_epochs: Passes over the full dataset
            log_interval: Print metrics every N steps
            eval_interval: Run evaluation every N steps
            save_path: Where to save best checkpoint
        """
        from torch.utils.data import DataLoader
        import time
        import csv
        import os

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2)

        # LR schedule: linear warmup (3%) then cosine decay to 0
        import math
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(0.03 * total_steps)
        print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps  # linear 0 → 1
            # cosine decay from 1 → 0
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # CSV logging — save metrics alongside the checkpoint
        log_path = save_path.replace(".pt", "_log.csv")
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["step", "loss", "avg_loss", "acc", "avg_acc",
                             "grad_norm", "lr", "eval_loss", "eval_acc"])
        print(f"Logging metrics to {log_path}")

        self.model.train()
        global_step = 0
        best_eval_acc = 0.0

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_acc = 0
            epoch_steps = 0
            t0 = time.time()

            for batch in train_loader:
                # --- Move to device ---
                chosen_ids = batch["chosen_ids"].to(self.device)
                chosen_mask = batch["chosen_mask"].to(self.device)
                rejected_ids = batch["rejected_ids"].to(self.device)
                rejected_mask = batch["rejected_mask"].to(self.device)

                # --- Forward: score both responses ---
                chosen_rewards = self.model(chosen_ids, chosen_mask)
                rejected_rewards = self.model(rejected_ids, rejected_mask)

                # --- Loss ---
                loss, acc = self.compute_loss(chosen_rewards, rejected_rewards)

                # --- Backward ---
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                scheduler.step()

                # --- Track metrics ---
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_steps += 1
                global_step += 1

                # --- Log ---
                if global_step % log_interval == 0:
                    avg_loss = epoch_loss / epoch_steps
                    avg_acc = epoch_acc / epoch_steps
                    current_lr = scheduler.get_last_lr()[0]
                    dt = time.time() - t0
                    print(
                        f"step {global_step:>6d} | "
                        f"loss {loss.item():.4f} (avg {avg_loss:.4f}) | "
                        f"acc {acc.item():.0%} (avg {avg_acc:.0%}) | "
                        f"grad_norm {grad_norm:.2f} | "
                        f"lr {current_lr:.2e} | "
                        f"{epoch_steps/dt:.1f} steps/s"
                    )
                    log_writer.writerow([
                        global_step, f"{loss.item():.4f}", f"{avg_loss:.4f}",
                        f"{acc.item():.4f}", f"{avg_acc:.4f}",
                        f"{grad_norm:.4f}", f"{current_lr:.6e}", "", ""
                    ])
                    log_file.flush()

                # --- Evaluate and checkpoint ---
                if eval_dataset is not None and global_step % eval_interval == 0:
                    eval_loss, eval_acc = self.evaluate(eval_dataset, batch_size)
                    print(f"  >>> EVAL: loss {eval_loss:.4f}, acc {eval_acc:.0%}")
                    log_writer.writerow([
                        global_step, "", "", "", "",
                        "", "", f"{eval_loss:.4f}", f"{eval_acc:.4f}"
                    ])
                    log_file.flush()
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        torch.save(self.model.state_dict(), save_path)
                        print(f"  >>> Saved best model (acc {eval_acc:.0%}) to {save_path}")

            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            print(f"  Avg loss: {epoch_loss / epoch_steps:.4f}")
            print(f"  Avg accuracy: {epoch_acc / epoch_steps:.0%}\n")

        log_file.close()
        print(f"Metrics saved to {log_path}")

    @torch.no_grad()
    def evaluate(self, dataset, batch_size=4, max_batches=100):
        """Evaluate on a dataset. Returns (avg_loss, avg_accuracy)."""
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        total_loss = 0
        total_acc = 0
        n = 0

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            chosen_ids = batch["chosen_ids"].to(self.device)
            chosen_mask = batch["chosen_mask"].to(self.device)
            rejected_ids = batch["rejected_ids"].to(self.device)
            rejected_mask = batch["rejected_mask"].to(self.device)

            chosen_rewards = self.model(chosen_ids, chosen_mask)
            rejected_rewards = self.model(rejected_ids, rejected_mask)
            loss, acc = self.compute_loss(chosen_rewards, rejected_rewards)

            total_loss += loss.item()
            total_acc += acc.item()
            n += 1

        self.model.train()
        return total_loss / n, total_acc / n


class TrainedRewardModel:
    """
    Wraps a trained RewardModel to match the naive reward function interface.

    The naive reward functions (LengthReward, BrevityReward, etc.) use:
        reward = compute(generated_ids, prompt_length, max_new_tokens, eos_token_id)

    This wrapper adapts the trained RewardModel (which takes input_ids + attention_mask)
    to the same interface, handling attention mask construction for left-padded
    sequences and post-EOS masking.
    """

    def __init__(self, reward_model):
        self.reward_model = reward_model

    def compute(self, generated_ids, prompt_length, max_new_tokens, eos_token_id):
        """
        Compute rewards using the trained reward model.

        Builds an attention mask that:
        1. Masks left-padding (leading eos tokens before the real prompt)
        2. Masks tokens after the first EOS in the response

        Args:
            generated_ids: (batch_size, seq_len) - prompt + generated response
            prompt_length: int - number of tokens in the (padded) prompt
            max_new_tokens: int - maximum response length (unused, kept for interface compat)
            eos_token_id: int - end-of-sequence token ID

        Returns:
            rewards: (batch_size,) - scalar reward per example
        """
        batch_size, seq_len = generated_ids.shape
        attention_mask = torch.ones_like(generated_ids)

        for i in range(batch_size):
            # Mask left-padding (leading eos tokens before the real prompt)
            for j in range(prompt_length):
                if generated_ids[i, j].item() == eos_token_id:
                    attention_mask[i, j] = 0
                else:
                    break

            # Mask tokens after first EOS in the response
            for j in range(prompt_length, seq_len):
                if generated_ids[i, j].item() == eos_token_id:
                    attention_mask[i, j + 1:] = 0
                    break

        with torch.no_grad():
            rewards = self.reward_model(generated_ids, attention_mask)
        return rewards
