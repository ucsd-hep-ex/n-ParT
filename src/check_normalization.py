import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import copy


use_nGPT = 1
learning_rate = 15e-4
n_layer = 12
n_head = 16
n_embd = 1024
block_size = 1024
base_scale = 1.0 
bias = False 
dropout = 0.1  
meta_vocab_size = 50304  


out_dir = "./checkpoints"
init_from = "scratch" 

def justnorm(x, idim=-1):
    """
    Normalize tensor `x` along the specified dimension using L2 norm.
    """
    dtype = x.dtype
    x = x.float()
    res = (x / x.norm(p=2, dim=idim, keepdim=True)).to(dtype=dtype)
    return res

def normalize_matrices(model):
    """
    Normalize all weight matrices in the model.
    """
    transformer = model.transformer
    module = model  

    # Normalize input embeddings
    transformer.wte.weight.data.copy_(justnorm(transformer.wte.weight.data, 1))
    module.lm_head.weight.data.copy_(justnorm(module.lm_head.weight.data, 1))

    config = model.config  

    for layer_idx in range(config.n_layer):
        block = transformer["h"][layer_idx] 

        block.query.weight.data.copy_(justnorm(block.query.weight.data, 1))             # n_proj, n_embd
        block.key.weight.data.copy_(justnorm(block.key.weight.data, 1))                 # n_proj, n_embd
        block.value.weight.data.copy_(justnorm(block.value.weight.data, 1))             # n_proj, n_embd
        block.att_c_proj.weight.data.copy_(justnorm(block.att_c_proj.weight.data, 0))   # n_embd, n_proj

        block.c_fc.weight.data.copy_(justnorm(block.c_fc.weight.data, 1))               # n_proj, n_embd
        block.mlp_c_proj.weight.data.copy_(justnorm(block.mlp_c_proj.weight.data, 0))   # n_embd, n_proj

def compute_weight_norms(model):
    """
    Computes the L2 norm of input, output, and internal weight matrices.

    Returns:
        dict: Contains sorted norms for embeddings and internal layers.
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module  # Unwrap DDP model if necessary

    norms = {}

    input_embeds = model.transformer.wte.weight
    input_norms = torch.norm(input_embeds.float(), p=2, dim=1).detach().cpu().numpy()

    output_embeds = model.lm_head.weight
    output_norms = torch.norm(output_embeds.float(), p=2, dim=1).detach().cpu().numpy()

    norms["input_embeddings"] = np.sort(input_norms)[::-1]
    norms["output_embeddings"] = np.sort(output_norms)[::-1]

    transformer = model.transformer
    config = model.config
    internal_norms = []

    for layer_idx in range(config.n_layer):
        block = transformer.h[layer_idx]
        layer_norms = [
            torch.norm(block.query.weight.float(), p=2, dim=1).detach().cpu().numpy(),
            torch.norm(block.key.weight.float(), p=2, dim=1).detach().cpu().numpy(),
            torch.norm(block.value.weight.float(), p=2, dim=1).detach().cpu().numpy(),
            torch.norm(block.att_c_proj.weight.float(), p=2, dim=0).detach().cpu().numpy(),
            torch.norm(block.c_fc.weight.float(), p=2, dim=1).detach().cpu().numpy(),
            torch.norm(block.mlp_c_proj.weight.float(), p=2, dim=0).detach().cpu().numpy(),
        ]
        
        internal_norms.append(np.mean([np.mean(norm) for norm in layer_norms]))

    norms["internal_layers"] = np.array(internal_norms)

    return norms

def check_normalization(model, tolerance=1e-2):
    """
    Checks if all weight matrices are properly normalized.
    """
    norms = compute_weight_norms(model)

    input_mean_norm = np.mean(norms["input_embeddings"])
    output_mean_norm = np.mean(norms["output_embeddings"])
    internal_mean_norm = np.mean(norms["internal_layers"])

    print(f"Mean Norm (Input Embeddings): {input_mean_norm:.5f}")
    print(f"Mean Norm (Output Embeddings): {output_mean_norm:.5f}")
    print(f"Mean Norm (Internal Layers): {internal_mean_norm:.5f}")

    issues = []

    if abs(input_mean_norm - 1.0) > tolerance:
        issues.append("Input Embeddings")
    if abs(output_mean_norm - 1.0) > tolerance:
        issues.append("Output Embeddings")
    if abs(internal_mean_norm - 1.0) > tolerance:
        issues.append("Internal Layers")

    if issues:
        print(f"WARNING: The following components are NOT properly normalized: {', '.join(issues)}")
    else:
        print("All model weights are properly normalized.")



def plot_embedding_norms(models, model_labels, save_path_prefix="embedding_norms"):
    """
    Plots the distribution of norms of input embeddings, output embeddings, and internal layers.

    Args:
        models (list): List of trained models.
        model_labels (list): Corresponding labels for the models.
        save_path_prefix (str): Prefix for saving the figure files.
    """
    
    #Input Embedding Norms
    plt.figure(figsize=(6, 5))
    for model, label in zip(models, model_labels):
        input_embeds = model.transformer.wte.weight
        input_norms = torch.norm(input_embeds.float(), p=2, dim=1).detach().cpu().numpy()

        sorted_norms = np.sort(input_norms)[::-1]
        normalized_rank = np.linspace(0, 1, len(sorted_norms))  # Normalize rank from 0 to 1

        plt.plot(normalized_rank, sorted_norms, label=label)

    plt.axhline(y=1.0, color='black', linestyle='-', label="Expected Norm")
    plt.xlabel("Normalized rank")
    plt.ylabel("Norm value")
    plt.title("Distribution of norms of Input Embeddings")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path_prefix}_input.png")
    plt.show()

    #Output Embedding Norms
    plt.figure(figsize=(6, 5))
    for model, label in zip(models, model_labels):
        output_embeds = model.lm_head.weight
        output_norms = torch.norm(output_embeds.float(), p=2, dim=1).detach().cpu().numpy()

        sorted_norms = np.sort(output_norms)[::-1]
        normalized_rank = np.linspace(0, 1, len(sorted_norms))

        plt.plot(normalized_rank, sorted_norms, label=label)

    plt.axhline(y=1.0, color='black', linestyle='-', label="Expected Norm")
    plt.xlabel("Normalized rank")
    plt.ylabel("Norm value")
    plt.title("Distribution of norms of Output Embeddings")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path_prefix}_output.png")
    plt.show()

    #Internal Layer Norms
    plt.figure(figsize=(6, 5))
    for model, label in zip(models, model_labels):
        transformer = model.transformer
        config = model.config
        internal_layer_norms = []

        for layer_idx in range(config.n_layer):
            block = transformer.h[layer_idx]
            layer_norms = [
                torch.norm(block.query.weight.float(), p=2, dim=1).detach().cpu().numpy(),
                torch.norm(block.key.weight.float(), p=2, dim=1).detach().cpu().numpy(),
                torch.norm(block.value.weight.float(), p=2, dim=1).detach().cpu().numpy(),
                torch.norm(block.att_c_proj.weight.float(), p=2, dim=0).detach().cpu().numpy(),
                torch.norm(block.c_fc.weight.float(), p=2, dim=1).detach().cpu().numpy(),
                torch.norm(block.mlp_c_proj.weight.float(), p=2, dim=0).detach().cpu().numpy(),
            ]
            internal_layer_norms.append(np.mean([np.mean(norm) for norm in layer_norms]))

        sorted_norms = np.sort(np.array(internal_layer_norms))[::-1]
        normalized_rank = np.linspace(0, 1, len(sorted_norms))

        plt.plot(normalized_rank, sorted_norms, label=label)

    plt.axhline(y=1.0, color='black', linestyle='-', label="Expected Norm")
    plt.xlabel("Normalized rank")
    plt.ylabel("Norm value")
    plt.title("Distribution of norms of Internal Layers")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path_prefix}_internal.png")
    plt.show()


if __name__ == "__main__":
   
    from model import GPTConfig, GPT 
    
    # Load existing model or initialize new one
    model_args = dict(use_nGPT=use_nGPT, n_layer=n_layer, n_head=n_head, 
                      n_embd=n_embd, block_size=block_size, base_scale=base_scale, 
                      bias=bias, vocab_size=None, dropout=dropout)

    if init_from == "resume":
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cpu")
        model_args.update(checkpoint["model_args"])
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.load_state_dict(checkpoint["model"])
        iter_num = checkpoint["iter_num"]
    else:
        print("Initializing a new model from scratch...")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    original_model = copy.deepcopy(model)

    if use_nGPT == 1:
        print("Applying initial normalization to weight matrices...")
        normalize_matrices(model)

    print("Checking weight normalization at start of training...")
    check_normalization(model)

    print("Model loaded and checked. Ready for training!")
    models = [original_model, model]  
    model_labels = ["GPT","nGPT"]
    
    plot_embedding_norms(models, model_labels)