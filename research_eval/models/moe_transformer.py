import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import EvaluableModel
from ..utils import calculate_cv # Relative import for utils

class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k=1, expert_hidden_dim=None, dropout=0.1, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.capacity_factor = capacity_factor # For load balancing, not used in this simplified version's loss
        
        expert_hidden_dim = expert_hidden_dim if expert_hidden_dim is not None else 2 * input_dim # Smaller FFN for MoE experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(expert_hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)
        self.aux_loss = torch.tensor(0.0)
        self.expert_counts = torch.zeros(num_experts)


    def forward(self, x): # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        x_reshaped = x.reshape(-1, self.input_dim) # (N, input_dim), N = batch_size * seq_len
        
        gate_logits = self.gate(x_reshaped) # (N, num_experts)
        
        # Top-k gating
        weights, indices = torch.topk(gate_logits, self.top_k, dim=1) # weights: (N, top_k), indices: (N, top_k)
        weights = F.softmax(weights, dim=1) # Softmax over top_k experts
        
        # Load balancing loss (simplified version: CV of router assignments)
        # A more standard one involves P_i * f_i (router prob * fraction of tokens)
        # Here, we use counts of tokens assigned to each expert post-routing.
        # This is a proxy for load.
        self.expert_counts.zero_().to(x.device) # Reset counts
        
        # For each token, which experts were chosen (top_k)
        # indices_one_hot = F.one_hot(indices, num_classes=self.num_experts).float() # (N, top_k, num_experts)
        # Sum over top_k to get a multi-hot encoding if top_k > 1
        # chosen_experts_per_token = indices_one_hot.sum(dim=1) # (N, num_experts)
        # tokens_per_expert = chosen_experts_per_token.sum(dim=0) # (num_experts)
        
        # Simplified: count based on top-1 choice for load balancing metric
        top1_indices = indices[:, 0]
        expert_load_counts = torch.bincount(top1_indices, minlength=self.num_experts).float()
        self.expert_counts = expert_load_counts.detach() # For metrics
        
        # Calculate load balancing loss based on router probabilities (pre-topk)
        # This encourages the router to distribute its probability mass.
        router_probs = F.softmax(gate_logits, dim=1) # (N, num_experts)
        tokens_per_expert_prob_sum = router_probs.sum(dim=0) # (num_experts)
        # Loss = CV^2 of tokens_per_expert_prob_sum. CV = std/mean.
        # This is a common load balancing loss for MoEs.
        # It measures the imbalance in the router's assignment probabilities.
        load_balance_loss = calculate_cv(tokens_per_expert_prob_sum).pow(2)
        self.aux_loss = load_balance_loss
        
        # Dispatch to experts and combine
        output = torch.zeros_like(x_reshaped.unsqueeze(1).repeat(1,self.top_k,1).transpose(1,2).contiguous().view(-1, self.output_dim)) # Placeholder
        
        # This part is tricky for batching with top_k > 1 and varying experts per token.
        # A common approach is to use a capacity factor and drop tokens if capacity is exceeded.
        # For simplicity, we iterate, which is slow but correct for small top_k.
        # If top_k = 1, it's much simpler.
        
        final_output_flat = torch.zeros_like(x_reshaped) # (N, output_dim)
        for i in range(x_reshaped.size(0)): # Iterate over each token (N tokens)
            token_output_sum = torch.zeros(self.output_dim, device=x.device)
            for k_idx in range(self.top_k):
                expert_idx = indices[i, k_idx]
                weight = weights[i, k_idx]
                token_output_sum += weight * self.experts[expert_idx](x_reshaped[i])
            final_output_flat[i] = token_output_sum
        
        return final_output_flat.reshape(batch_size, seq_len, self.output_dim)

class MoETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_experts, dim_feedforward=None, dropout=0.1, top_k_experts=1):
        super().__init__()
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 2 * d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe_ffn = MoELayer(d_model, d_model, num_experts, top_k=top_k_experts, expert_hidden_dim=dim_feedforward, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.aux_loss = torch.tensor(0.0)
        self.expert_counts = torch.zeros(num_experts)


    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Norm first for stability
        normed_src = self.norm1(src)
        src2 = self.self_attn(normed_src, normed_src, normed_src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False)[0]
        src = src + self.dropout1(src2)
        
        normed_src = self.norm2(src)
        src2 = self.moe_ffn(normed_src)
        src = src + self.dropout2(src2)
        
        self.aux_loss = self.moe_ffn.aux_loss
        self.expert_counts = self.moe_ffn.expert_counts.detach()
        return src

class MoETransformerClassifier(EvaluableModel):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_experts, 
                 dim_feedforward, num_classes, dropout=0.1, padding_idx=0, top_k_experts=1):
        super().__init__()
        self.model_name = "MoETransformer"
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.padding_idx)
        
        self.encoder_layers = nn.ModuleList([
            MoETransformerEncoderLayer(embed_dim, num_heads, num_experts, dim_feedforward, dropout, top_k_experts)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim) # Final layer norm

        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout_embed = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        src_key_padding_mask = (input_ids == self.padding_idx) if attention_mask is None else (attention_mask == 0)
        
        x = self.dropout_embed(self.embedding(input_ids))
        
        current_aux_loss = 0.0
        all_layer_expert_counts = []
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            current_aux_loss += layer.aux_loss
            all_layer_expert_counts.append(layer.expert_counts)
        
        self.aux_loss_value = current_aux_loss / len(self.encoder_layers) if self.encoder_layers else 0.0
        self.expert_counts_per_layer = all_layer_expert_counts # For metrics

        x = self.final_norm(x)
        pooled_output = x.mean(dim=1)
        return self.fc(pooled_output)

    def get_parameter_count(self) -> int: return self.get_trainable_parameter_count()
    def get_model_name(self) -> str: return self.model_name
    def get_auxiliary_loss(self): return self.aux_loss_value
    
    def get_model_specific_metrics(self):
        metrics = {}
        if hasattr(self, 'expert_counts_per_layer') and self.expert_counts_per_layer:
            for i, counts in enumerate(self.expert_counts_per_layer):
                metrics[f'moe_layer_{i}_expert_load_cv'] = calculate_cv(counts).item()
        return metrics
