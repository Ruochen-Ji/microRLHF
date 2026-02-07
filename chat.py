"""
Gradio chat UI for nanoGPT — talk to your trained models in a browser.

Usage:
    pip install gradio
    python chat.py
"""

import os
import glob
import pickle
from contextlib import nullcontext

import torch
import tiktoken
import gradio as gr

from torch.nn import functional as F

from model import GPTConfig, GPT
from lora import apply_lora_to_model

# ---------------------------------------------------------------------------
# globals
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

model = None
encode = None
decode = None
eos_token = None
current_ckpt = None

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def discover_checkpoints():
    """Scan for out*/ckpt.pt files."""
    paths = sorted(glob.glob("out*/ckpt.pt"))
    return paths


def load_checkpoint(ckpt_path):
    """Load a checkpoint and set up the tokenizer (mirrors sample.py logic)."""
    global model, encode, decode, eos_token, current_ckpt

    # free previous model
    if model is not None:
        del model
        torch.cuda.empty_cache()

    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    new_model = GPT(gptconf)

    # handle LoRA checkpoints
    if "lora_config" in checkpoint:
        lora_cfg = checkpoint["lora_config"]
        new_model = apply_lora_to_model(
            new_model,
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,
            target_modules=lora_cfg["target_modules"],
        )

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    new_model.load_state_dict(state_dict)

    new_model.eval()
    new_model.to(device)
    model = new_model

    # tokenizer: check for meta.pkl (char-level), else tiktoken gpt-2
    load_meta = False
    if "config" in checkpoint and "dataset" in checkpoint["config"]:
        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        load_meta = os.path.exists(meta_path)

    if load_meta:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
        eos_token = None  # char-level models typically have no EOS
    else:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        eos_token = enc.eot_token  # 50256

    current_ckpt = ckpt_path


def generate_tokens(prompt, temperature, top_k, max_new_tokens):
    """Yield one decoded token at a time, stopping at EOS."""
    ids = encode(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    block_size = model.config.block_size
    with torch.no_grad():
        with ctx:
            for _ in range(int(max_new_tokens)):
                idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token_id = idx_next.item()
                if eos_token is not None and token_id == eos_token:
                    break
                idx = torch.cat((idx, idx_next), dim=1)
                yield decode([token_id])

# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def on_select_checkpoint(ckpt_path):
    """Called when user picks a new checkpoint from the dropdown."""
    if not ckpt_path:
        return [], gr.update(interactive=False)
    load_checkpoint(ckpt_path)
    return [], gr.update(interactive=True)


def on_user_message(user_message, history, temperature, top_k, max_new_tokens):
    """Streaming generator — yields partial history as tokens arrive."""
    if model is None:
        history = history + [{"role": "user", "content": user_message},
                             {"role": "assistant", "content": "Please select a checkpoint first."}]
        yield history, ""
        return

    # build prompt from history
    prompt = ""
    for msg in history:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        else:
            prompt += f"Assistant: {msg['content']}\n"
    prompt += f"User: {user_message}\nAssistant:"

    history = history + [{"role": "user", "content": user_message},
                         {"role": "assistant", "content": ""}]

    response = ""
    for token in generate_tokens(prompt, temperature, top_k, max_new_tokens):
        response += token
        # stop at "User:" turn boundary
        if "User:" in response:
            response = response[:response.index("User:")]
            history[-1]["content"] = response.strip()
            yield history, ""
            return
        history[-1]["content"] = response.strip()
        yield history, ""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    checkpoints = discover_checkpoints()

    with gr.Blocks(title="Chat with your model") as demo:
        gr.Markdown("### Chat with your model")

        ckpt_dropdown = gr.Dropdown(
            choices=checkpoints,
            label="Checkpoint",
            value=None,
            interactive=True,
        )

        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Type a message…", label="Message", interactive=False)

        with gr.Accordion("Generation settings", open=False):
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            top_k = gr.Slider(1, 500, value=200, step=1, label="Top-k")
            max_new_tokens = gr.Slider(1, 2048, value=1024, step=1, label="Max new tokens")

        clear_btn = gr.Button("Clear chat")

        # events
        ckpt_dropdown.change(
            on_select_checkpoint, inputs=[ckpt_dropdown], outputs=[chatbot, msg]
        )
        msg.submit(
            on_user_message,
            inputs=[msg, chatbot, temperature, top_k, max_new_tokens],
            outputs=[chatbot, msg],
        )
        clear_btn.click(lambda: [], outputs=[chatbot])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
