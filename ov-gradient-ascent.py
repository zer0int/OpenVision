"""
Uses a heavily modified version of Original CLIP Gradient Ascent Script: by Twitter / X: @advadnoun

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GRADIENT ASCENT FOR >> OpenVision << MODELS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/UCSC-VLAA/OpenVision

"""

import argparse
import os
import kornia.augmentation as kaugs
import kornia
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from colorama import Fore, Style
import copy
from torch.cuda.amp import autocast, GradScaler
from safetensors.torch import load_file
import json
from typing import Any, Dict, List, Optional, Tuple, Union


import warnings # Stop the spam
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from cliptoolsoptimized import fix_random_seed # For determinism!

import open_clip
from open_clip.tokenizer import HFTokenizer
from open_clip import create_model
from open_clip.model import CLIP
from transformers import AutoTokenizer

# Example using https://huggingface.co/UCSC-VLAA/openvision-vit-large-patch14-224/tree/main

# Argument Parsing; rarely needed, but just CTRL+F for: Additional CONFIGURATION
def parse_arguments():
    parser = argparse.ArgumentParser(description='OpenVision gradient ascent')
    parser.add_argument('--batch_size', default=13, type=int)
    parser.add_argument('--use_model', default="F:/openvision-vit-large-patch14-224", help="Path to an OpenVision model")
    parser.add_argument('--use_image', type=str, default="testcat/catdog.png", help="Path to image")
    parser.add_argument('--img_folder', type=str, default="None", help="Path to image folder (batch process)")
    parser.add_argument("--deterministic", action='store_true', help="Use deterministic behavior (CUDA backends, torch, numpy)")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model
args.model_name = args.use_model

scaler = GradScaler()

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Expect mean and std as lists of 3 elements.
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

# Image Loader
def load_image(img_path, sideX, sideY):
    im = torch.tensor(np.array(Image.open(img_path).convert("RGB"))).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255
    im = F.interpolate(im, (sideX, sideY))
    return im

# Image Augmentation Pipeline
def augment(into, augs):
    return augs(into)


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

# Text Encode forward pass
def clip_encode_text(model, text, many_tokens, attn_mask, prompt, pool_type="last", normalize=True):

    # Token embedding + positional embedding
    #x = model.token_embedding(text)  # [B, T, D]
    x = text @ model.token_embedding.weight
    
    x = x + model.positional_embedding[:x.size(1)]

    # Transformer with full self-attention mask
    x = model.transformer(x)# x, attn_mask=model.attn_mask

    # Final layer norm
    x = model.ln_final(x)  # [B, T, D]

    # Pooling: determine based on model.text_pool_type
    x = text_global_pool(x, model.text_pool_type)# default (for model) is 'last'.

    # Projection
    if model.text_projection is not None:
        if isinstance(model.text_projection, torch.nn.Linear):
            x = model.text_projection(x)
        else:
            x = x @ model.text_projection

    return x

# Image Encode forward pass
def clip_encode_image(model, image):
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


# Entertain user by printing CLIP's 'opinion' rants about image to console
def checkin(loss, tx, lll, tok, tok_kwargs, bests, imagename):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist(), **tok_kwargs) for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist(), **tok_kwargs)
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}" + Fore.RESET)
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace(',', '')
        tokens = j.split()
        unique_tokens.update(tokens)
    os.makedirs("opinion-tokens", exist_ok=True)
    with open(f"opinion-tokens/tokens_{imagename}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

# Softmax
class Pars(nn.Module):
    def __init__(self, batch_size, many_tokens, prompt, vocab_size=32000,
                 bos_token_id=101, eos_token_id=102, pad_token_id=0, gumbel_temp=1000):
        super().__init__()

        self.batch_size = batch_size
        self.many_tokens = many_tokens
        self.vocab_size = vocab_size
        self.gumbel_temp = gumbel_temp
        self.prompt = prompt

        # The learned distribution / region
        st = torch.zeros(batch_size, many_tokens, vocab_size).normal_()
        self.normu = nn.Parameter(st.cuda())

        # start token TODO find out if this matters or is disruptive
        self.start = torch.zeros(batch_size, 1, vocab_size).cuda()
        self.start[:, :, bos_token_id] = 1

        ptt = prompt

        self.prompt = torch.zeros(batch_size, len(ptt), vocab_size).cuda()
        for jk, pt in enumerate(ptt):
            self.prompt[:, jk, pt] = 1 

        # If you use self.start in forward, make it +1 here:
        #self.pad = torch.zeros(batch_size, 80 - (self.many_tokens + len(ptt) + 1), vocab_size).cuda()
        self.pad = torch.zeros(batch_size, 80 - (self.many_tokens + len(ptt) + 0), vocab_size).cuda()
        self.pad[:, :, pad_token_id] = 1

    def forward(self):
        # Differentiable hard token selection
        soft = F.gumbel_softmax(self.normu, tau=self.gumbel_temp, dim=-1, hard=True)
        
        # Though it only makes a small difference, less coherent if start token is prepended:
        #tokens = torch.cat([self.start, self.prompt, soft, self.pad], dim=1)
        tokens = torch.cat([self.prompt, self.pad, soft], dim=1)

        ids = tokens.argmax(dim=-1)  # [B, 80]
        #print("Last tokens per batch:", ids[:, -1]) # debug

        return tokens


# Gradient Ascent
def ascend_txt(image, model, lats, many_tokens, prompt, nom, attn_mask, augment):
    iii = nom(augment(image[:,:3,:,:].expand(lats.normu.shape[0], -1, -1, -1)))
    iii = clip_encode_image(model, iii).detach()
    lll = lats()
    tx = clip_encode_text(model, lll, many_tokens, attn_mask, prompt)
    loss = -100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, lats.normu.shape[0]).T.mean(1)
    return loss, tx, lll


# Loop with AMP
def train(image, model, lats, many_tokens, prompt, optimizer, nom, attn_mask, augment):
    with autocast():
        loss1, tx, lll = ascend_txt(image, model, lats, many_tokens, prompt, nom, attn_mask, augment)
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward(retain_graph=True)
    scaler.step(optimizer)
    scaler.update()
    return loss1, tx, lll



def generate_target_text_embeddings(img_path, model, lats, optimizer, training_iterations, checkin_step, many_tokens, prompt, nom, attn_mask, image_size, augment, tok, tok_kwargs, bests, args):

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = load_image(img_path, image_size, image_size)
    print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)

    scaler = GradScaler()

    best_loss = float('inf')  # Initialize the best loss as infinity
    best_text_embeddings = None  # Placeholder for the best text embeddings

    for j in range(training_iterations):
        # Adjust active tokens dynamically at specific steps
                
        # Training step
        loss, tx, lll = train(img, model, lats, many_tokens, prompt, optimizer, nom, attn_mask, augment)
        current_loss = loss.mean().item()

        # Update best embeddings if current loss is better
        if current_loss < best_loss:
            best_loss = current_loss
            best_text_embeddings = copy.deepcopy(tx.detach())
            print(Fore.RED + Style.BRIGHT + f"New best loss: {best_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, tok_kwargs, bests, img_name)
            print(Fore.RED + Style.BRIGHT + "-------------------" + Fore.RESET)

        # Print learning rate for monitoring
        if j % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(Fore.GREEN + f"Iteration {j}: Average Loss: {current_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, tok_kwargs, bests, img_name)

    # Save the best embeddings to disk
    os.makedirs("txtembeds", exist_ok=True)
    torch.save(best_text_embeddings, f"txtembeds/{img_name}_text_embedding.pt")
    print(Fore.MAGENTA + Style.BRIGHT + "\nBest text embedding saved to 'txtembeds'.\nTokens (CLIP 'opinion') saved to 'opinion-tokens' folder.\n" + Fore.RESET)
    del optimizer, lats, scaler, prompt
    return img, best_text_embeddings, img_path




# Main loop
def main():
    args = parse_arguments()
    if args.deterministic:
        fix_random_seed()
    device="cuda" if torch.cuda.is_available() else "cpu"

    model_path = model_name_or_path
    config_path = f"{model_path}/open_clip_config.json"

    # === Load JSON config ===
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_cfg = cfg["model_cfg"]
    mean = cfg["preprocess_cfg"]["mean"]
    std = cfg["preprocess_cfg"]["std"]
    context_len = model_cfg["text_cfg"]["context_length"]
    tokenizer_name = model_cfg["text_cfg"]["hf_tokenizer_name"]
    image_size = cfg["model_cfg"]["vision_cfg"].get("image_size")
    vision_cfg = cfg["model_cfg"]["vision_cfg"]
    text_cfg = cfg["model_cfg"]["text_cfg"]
    clip_args = {k: v for k, v in model_cfg.items() if k not in ("vision_cfg", "text_cfg")}

    # Hypothetically needed for masking, though mask not needed for *this* model.
    head_width = model_cfg["vision_cfg"]["head_width"]
    width = model_cfg["vision_cfg"]["width"]
    num_heads = width // head_width

    # === Instantiate / Load Model ===  
    model = CLIP(vision_cfg=vision_cfg, text_cfg=text_cfg, **clip_args)  
    
    state_dict = torch.load(f"{model_path}/open_clip_pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).float()

    # === Preprocessing ===
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # specific values from config; OpenVision values != CLIP values
    #normalizer = Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cuda()
    normalizer = Normalization(mean=mean, std=std).cuda()

    # === Tokenizer ===
    tokenizer = HFTokenizer(tokenizer_name, context_length=context_len)    
    
    # For decode(), as isn't exposed with sub-imported HF-tokenizer:
    tok = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=context_len, use_fast=True)
    tok_kwargs = dict(skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
  
    # === Init for use in softmax / sampling tokens ===
    bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None'}   
    prompt = tokenizer('''''').numpy().tolist()[0]
    prompt = [i for i in prompt if i != 0 and i != 101 and i != 102 and i != 100 and i != 103]
    #{'bos_token_id': 101, 'eos_token_id': 102, 'pad_token_id': 0, 'unk_token_id': 100, 'mask_token_id': 103}
    
    attn_mask = None # Here: unnecessary placeholder, but would be needed if model uses mask

    # ----------------------------
    # Additional CONFIGURATION
    # ----------------------------
    
    checkin_step = 10 # How often to print results (token ramble to cli)
    tokinit = 4 # How many tokens to sample @ softmax. Too many = marginally related words + memory use UP!. Add "bests" above if you decide to try with many many tokens anyway!
    iterations = 340 # Number of optimization steps

    # ----------------------------

    lats = Pars(args.batch_size, tokinit, prompt).cuda() # softmax
    augs = torch.nn.Sequential(kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda()).cuda() # augs of input image
    optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}]) # optimizer

    if args.img_folder != "None":
        image_folder = args.img_folder
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                       if f.lower().endswith(valid_extensions)]
        
        for img_path in image_files:
            lats = Pars(args.batch_size, tokinit, prompt).cuda()
            tokinit = tokinit
            iterations = iterations
            optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
            bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None'}
            prompt = tokenizer('''''').numpy().tolist()[0]
            prompt = [i for i in prompt if i != 0 and i != 101 and i != 102 and i != 100 and i != 103]
            img, target_text_embedding, img_path = generate_target_text_embeddings(img_path, model, lats, optimizer, iterations, checkin_step, tokinit, prompt, normalizer, attn_mask, image_size, augs, tok, tok_kwargs, bests, args)
            print(f"Done processing image: {img_path}")

    else:
        img, target_text_embedding, img_path = generate_target_text_embeddings(args.use_image, model, lats, optimizer, iterations, checkin_step, tokinit, prompt, normalizer, attn_mask, image_size, augs, tok, tok_kwargs, bests, args)
        print(f"Done processing image: {img_path}")


if __name__ == "__main__":
    main()
