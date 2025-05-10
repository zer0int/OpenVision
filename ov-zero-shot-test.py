import torch
from torchvision import transforms
from PIL import Image
import os, json
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import nn
import argparse

import warnings # stop spam
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import open_clip
from open_clip.tokenizer import HFTokenizer
from open_clip import create_model
from open_clip.model import CLIP
from transformers import AutoTokenizer

# Example using https://huggingface.co/UCSC-VLAA/openvision-vit-large-patch14-224/tree/main

def parse_arguments():
    parser = argparse.ArgumentParser(description='OpenVision Text-Image Test')
    parser.add_argument('--use_model', type=str, default="F:/openvision-vit-large-patch14-224", help="Path to an OpenVision model + config")
    parser.add_argument('--image_dir', type=str, default="testcat", help="Path to image directory")
    return parser.parse_args()

args = parse_arguments()

# === Configs ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = args.use_model
config_path = f"{model_path}/open_clip_config.json"
image_dir = args.image_dir

# === Load config ===
with open(config_path, "r") as f:
    cfg = json.load(f)

mean = cfg["preprocess_cfg"]["mean"]
std = cfg["preprocess_cfg"]["std"]
context_len = cfg["model_cfg"]["text_cfg"]["context_length"]
tokenizer_name = cfg["model_cfg"]["text_cfg"]["hf_tokenizer_name"]
model_cfg = cfg["model_cfg"]
vision_cfg = cfg["model_cfg"]["vision_cfg"]
text_cfg = cfg["model_cfg"]["text_cfg"]
clip_args = {k: v for k, v in model_cfg.items() if k not in ("vision_cfg", "text_cfg")}

# === Instantiate / Load Model ===  
model = CLIP(vision_cfg=vision_cfg, text_cfg=text_cfg, **clip_args)  
state_dict = torch.load(f"{model_path}/open_clip_pytorch_model.bin", map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


print("\nVisual Config Used:")
print(f"  Pool type:             {model.visual.pool_type}")
print(f"  Final LN after pool:   {model.visual.final_ln_after_pool}")
print(f"  Attentional pool:      {model.visual.attn_pool is not None}")
print(f"  Projection shape:      {model.visual.proj.shape}")
print(f"  Positional embedding:  {model.visual.positional_embedding.shape}")
print(f"  CLS token:             {model.visual.class_embedding.shape}")
print("\n----------------------------------------\n")
#print({k: getattr(model.visual, k) for k in dir(model.visual)
#       if not k.startswith("_") and not callable(getattr(model.visual, k))})


# === Preprocess ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# === Tokenize ===
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bat", "a photo of a text", "cat", "dog", "bat", "hey", "text"]
tokenizer = HFTokenizer(tokenizer_name, context_length=context_len)
text_tokens = tokenizer(texts).to(device)

def text_global_pool(
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        pool_type: str = 'last',
) -> torch.Tensor:
    if pool_type == 'first':
        pooled = x[:, 0]
    elif pool_type == 'last':
        pooled = x[:, -1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    else:
        pooled = x

    return pooled

# You can also use model.encode_image(), but here's the exposed forward pass if you need it:
def encode_image_fixed(model, image):
    # Patchify
    x = model.visual.conv1(image)                                 # [B, C, H, W] → [B, width, gh, gw]
    x = x.reshape(x.shape[0], x.shape[1], -1)                     # [B, width, N]
    x = x.permute(0, 2, 1)                                        # [B, N, width]

    # Add class token + positional embedding
    cls_token = model.visual.class_embedding.expand(x.shape[0], 1, -1)  # [B, 1, D]
    x = torch.cat([cls_token, x], dim=1)                          # [B, N+1, D]
    x = x + model.visual.positional_embedding.to(x.dtype)        # [B, N+1, D]

    # Dropout + LN
    #x = model.visual.patch_dropout(x)
    x = model.visual.ln_pre(x)

    # Transformer
    x = model.visual.transformer(x)

    # Final pooling: mean of patch tokens
    pooled = x[:, 1:].mean(dim=1)                                 # [B, D]
    pooled = model.visual.ln_post(pooled)
    pooled = pooled @ model.visual.proj                           # [B, output_dim]

    return pooled

# You can also use model.encode_text(), but here's the exposed forward pass if you need it:
def encode_text_fixed(model, text_tokens):
    # Determine dtype used by the transformer's parameters
    cast_dtype = model.transformer.get_cast_dtype()

    # Token embedding + positional embedding
    x = model.token_embedding(text_tokens).to(cast_dtype)  # [B, T, D]
    x = x + model.positional_embedding[:x.size(1)].to(cast_dtype)

    # Transformer with full self-attention mask
    x = model.transformer(x, attn_mask=model.attn_mask)

    # Final layer norm
    x = model.ln_final(x)  # [B, T, D]

    # Pooling: determine based on model.text_pool_type
    x = text_global_pool(x, text_tokens, model.text_pool_type)
    #x = text_global_pool(x, text_tokens, 'last') # Correct model config, same as model.text_pool_type
    #x = text_global_pool(x, text_tokens, 'argmax') # ruined!

    # Projection
    if model.text_projection is not None:
        if isinstance(model.text_projection, torch.nn.Linear):
            x = model.text_projection(x)
        else:
            x = x @ model.text_projection

    return x


# Texts
with torch.no_grad():
    text_features = encode_text_fixed(model, text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)


# Process all Images
results = []
print("\n=== Cosine Similarities and Predictions ===")
for filename in os.listdir(image_dir):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue

    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = encode_image_fixed(model, image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        cosine = (image_features @ text_features.T)[0]  # shape: (len(texts),)
        logits = model.logit_scale.exp() * cosine
        probs = logits.softmax(dim=-1)

    sorted_indices = cosine.argsort(descending=True)

    print(f"\n--- {filename} ---")
    for idx in sorted_indices:
        label = texts[idx]
        cos = cosine[idx].item()
        prob = probs[idx].item()
        print(f"{label:<25} cosine: {cos:+.4f}  prob: {prob:.4%}")

    best_idx = probs.argmax().item()
    best_label = texts[best_idx]
    best_score = probs[best_idx].item()
    results.append((filename, best_label, best_score, probs.cpu().tolist()))

# === Per-Text Best Image ===
print("\n=== Best Image Per Text ===")
num_labels = len(texts)
best_images = [(None, -float("inf")) for _ in range(num_labels)]  # (filename, prob)

for filename, _, _, prob_list in results:
    for i, p in enumerate(prob_list):
        if p > best_images[i][1]:
            best_images[i] = (filename, p)

for i, (fname, p) in enumerate(best_images):
    print(f"{texts[i]:<20} ← {fname:>25} ({p:.4%})")
