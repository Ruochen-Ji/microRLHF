"""
LoRA (Low-Rank Adaptation) Implementation for nanoGPT

This file implements LoRA, a parameter-efficient finetuning technique.

=== THE KEY INSIGHT ===

When finetuning large models, researchers discovered that the weight updates
have a low "intrinsic dimensionality" - meaning they can be well-approximated
by low-rank matrices.

=== HOW IT WORKS ===

For a pretrained weight matrix W ∈ R^(d×k), instead of learning a full
update ΔW ∈ R^(d×k), LoRA decomposes it as:

    W_new = W + ΔW = W + B @ A

where:
    - A ∈ R^(r×k)  (the "down projection")
    - B ∈ R^(d×r)  (the "up projection")
    - r << min(d, k) is the "rank" (typically 4, 8, or 16)

=== WHY IS THIS EFFICIENT? ===

Example with GPT-2's attention projection (768 → 768):
    - Full finetuning: 768 × 768 = 589,824 parameters
    - LoRA (r=8): 768 × 8 + 8 × 768 = 12,288 parameters
    - That's ~48x fewer parameters!

=== INTUITION ===

Think of it like this: instead of allowing the model to change in ANY direction
in weight space (very high dimensional), we constrain it to only change along
a few important directions (the low-rank subspace). Surprisingly, this works
really well for adaptation tasks!

References:
- LoRA paper: https://arxiv.org/abs/2106.09685
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    A Linear layer augmented with LoRA (Low-Rank Adaptation).
    
    This wraps an existing nn.Linear and adds trainable low-rank matrices.
    The original weights are frozen; only the LoRA matrices are trained.
    
    Forward pass: y = x @ W^T + x @ (B @ A)^T
                    = x @ W^T + x @ A^T @ B^T
    
    We scale the LoRA output by (alpha / r) to control the magnitude of
    the adaptation relative to the pretrained weights.
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            original_linear: The pretrained nn.Linear layer to adapt
            rank: The rank of the low-rank decomposition (r in the paper)
            alpha: Scaling factor. The LoRA output is scaled by alpha/rank.
                   Higher alpha = stronger adaptation effect.
            dropout: Dropout probability applied to input before LoRA
        """
        super().__init__()
        
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # This controls the "strength" of LoRA
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # Freeze the original weights because we don't want to update the orignal weights. 
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        
        # LoRA matrices
        # A: "down projection" - projects input to low-rank space
        # B: "up projection" - projects back to output space
        #
        # Why this initialization?
        # - A is initialized with small random values (Kaiming/He init)
        # - B is initialized to zeros
        # This means at the START of training, B @ A = 0, so the model
        # behaves exactly like the pretrained model. The adaptation
        # "grows" from zero during training.
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with Kaiming uniform (good for linear layers)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero - this is crucial! It means we start with no change.
        nn.init.zeros_(self.lora_B)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combines original linear + LoRA adaptation.
        
        Mathematically: output = W @ x + (B @ A) @ x * scaling
        """
        # Original pretrained transformation (frozen)
        result = self.original_linear(x)
        
        # LoRA adaptation (trainable)
        # First apply dropout, then project down (A), then project up (B)
        lora_output = self.dropout(x)
        lora_output = lora_output @ self.lora_A.T  # Project to rank-r space
        lora_output = lora_output @ self.lora_B.T  # Project back to output space
        lora_output = lora_output * self.scaling   # Scale the contribution
        
        return result + lora_output
    
    def get_lora_params(self) -> int:
        """Returns the number of trainable LoRA parameters."""
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list = None,
) -> nn.Module:
    """
    Apply LoRA to specific linear layers in a model.
    
    Args:
        model: The model to modify
        rank: LoRA rank (lower = fewer params, higher = more expressive)
        alpha: Scaling factor for LoRA contributions
        dropout: Dropout rate for LoRA layers
        target_modules: List of module name patterns to apply LoRA to.
                       If None, applies to attention projections (c_attn, c_proj)
    
    Returns:
        The modified model with LoRA layers
    
    For GPT-2, the key linear layers are:
        - c_attn: Combined Q, K, V projection (most important for adaptation)
        - c_proj: Attention output projection
        - c_fc: MLP first linear layer
        - c_proj (in MLP): MLP second linear layer
    
    The paper found that applying LoRA to attention layers (especially Q and V)
    gives the best results with fewest parameters.
    """
    if target_modules is None:
        # Default: apply to attention projections only
        # These are the most impactful for style/task adaptation
        # c_atten: Q, K, V 
        # c_proj: Attention output -> Residual, MLP hidden -> output
        target_modules = ['c_attn', 'c_proj']
    
    # First, freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Track statistics
    total_lora_params = 0
    modified_layers = []
    
    # Walk through all named modules and replace matching Linear layers
    for name, module in model.named_modules():
        # Check if this module's name matches any of our target patterns
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get the parent module so we can replace the child
                # e.g., for "transformer.h.0.attn.c_attn", parent is the attn module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                # Create LoRA-wrapped version of this linear layer
                lora_layer = LoRALinear(
                    original_linear=module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                
                # Replace the original layer with the LoRA version
                setattr(parent, child_name, lora_layer)
                
                total_lora_params += lora_layer.get_lora_params()
                modified_layers.append(name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("LoRA Applied Successfully!")
    print(f"{'='*60}")
    print(f"Rank (r): {rank}")
    print(f"Alpha: {alpha}")
    print(f"Scaling (alpha/r): {alpha/rank:.2f}")
    print(f"Target modules: {target_modules}")
    print(f"Modified layers: {len(modified_layers)}")
    print(f"Total LoRA parameters: {total_lora_params:,}")
    
    # Calculate percentage of original model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    print(f"{'='*60}\n")
    
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Extract only the LoRA parameters from a model.
    
    This is useful for saving checkpoints - you only need to save
    the tiny LoRA weights, not the entire pretrained model!
    
    A LoRA checkpoint might be only ~1MB vs ~500MB for full model.
    """
    lora_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Save both A and B matrices for this layer
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data.clone()
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data.clone()
    
    return lora_state_dict


def load_lora_state_dict(model: nn.Module, lora_state_dict: dict):
    """
    Load LoRA weights into a model that already has LoRA applied.
    
    This allows you to:
    1. Load pretrained base model
    2. Apply LoRA structure
    3. Load saved LoRA weights
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            
            if a_key in lora_state_dict:
                module.lora_A.data.copy_(lora_state_dict[a_key])
            if b_key in lora_state_dict:
                module.lora_B.data.copy_(lora_state_dict[b_key])


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights back into the original linear layers.
    
    After training, you can "bake in" the LoRA adaptation:
        W_merged = W_original + (B @ A) * scaling
    
    This gives you a standard model without LoRA overhead at inference time!
    The merged model is exactly equivalent but faster (no extra computation).
    
    Note: This is destructive - you lose the ability to separate the adaptation.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Compute the LoRA contribution: B @ A * scaling
            lora_weight = (module.lora_B @ module.lora_A) * module.scaling
            
            # Add it to the original weight
            module.original_linear.weight.data += lora_weight
            
            # Get parent to replace LoRALinear with merged Linear
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            # Replace LoRALinear with the merged original linear
            setattr(parent, child_name, module.original_linear)
    
    print("LoRA weights merged into base model.")
    return model


# =============================================================================
# EDUCATIONAL EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Quick demonstration of LoRA's parameter efficiency.
    """
    print("LoRA Parameter Efficiency Demo")
    print("=" * 50)
    
    # Simulate a weight matrix like in GPT-2 attention
    d, k = 768, 768  # typical GPT-2 dimensions
    
    # Full finetuning: update all d*k parameters
    full_params = d * k
    print(f"\nFull finetuning parameters: {full_params:,}")
    
    # LoRA with different ranks
    for r in [4, 8, 16, 32]:
        lora_params = r * k + d * r  # A is (r×k), B is (d×r)
        reduction = full_params / lora_params
        print(f"LoRA rank={r}: {lora_params:,} params ({reduction:.1f}x reduction)")
    
    print("\n" + "=" * 50)
    print("Key insight: Even with r=32, we use 24x fewer parameters!")
    print("And rank=8 typically works well for most tasks.")
