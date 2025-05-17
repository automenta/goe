import math
import collections
import warnings # Import the warnings module
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming these utilities are available in the framework's utils
from ..utils import calculate_cv, gumbel_softmax, calculate_entropy

# --- Model Components (Adapted from goe_original.py) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, embed_dim)
        # pe: (1, max_len, embed_dim)
        # Ensure sequence length doesn't exceed max_len
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
             # Handle sequences longer than max_len, e.g., truncate or pad PE
             # Using available PE up to seq_len
             pe_to_use = self.pe[:, :self.pe.size(1)]
             # Pad x if needed? No, PE should match x's seq_len.
             # Let's just use the available PE up to min(seq_len, max_len)
             pe_to_use = self.pe[:, :min(seq_len, self.pe.size(1))]
             # If seq_len > max_len, PE won't cover the whole sequence. This is a limitation.
             # A proper fix would involve handling longer sequences, but for now,
             # let's assume max_len is set appropriately or accept this limitation.
             # The original code used max(512, max_lm_seq_len_param * 2), suggesting max_len is dynamic.
             # For a fixed classifier, a fixed max_len like 512 or 1024 is typical.
             # Let's keep the fixed 5000 for now as in the original snippet provided.
             # The slicing `self.pe[:, :x.size(1)]` is correct if x.size(1) <= max_len.
             # If x.size(1) > max_len, this slice will be `self.pe[:, :max_len]`, which is wrong.
             # Correct slicing: `self.pe[:, :seq_len]` if seq_len <= max_len, else handle error/warning.
             # Let's assume max_len is sufficient for now.
             pe_to_use = self.pe[:, :seq_len] # This is correct if seq_len <= max_len

        # x: (batch_size, seq_len, embed_dim)
        # pe: (1, max_len, embed_dim)
        # Add positional encoding to the input tensor.
        # The PE buffer is already on the correct device because the model is.
        # Slice PE to match the sequence length of the input tensor x.
        # If seq_len > max_len, PE will be truncated.
        return self.dropout(x + self.pe[:, :seq_len])


class GatedAttention(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True, dropout=dropout)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, seq_len, embed_dim)
        # key_padding_mask: (batch_size, seq_len) - True for padding
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask) # Pass mask to attention
        # Concatenate x and attn_out along the last dimension
        gated_attn = torch.sigmoid(self.gate(torch.cat([x, attn_out], dim=-1))) * attn_out
        return self.norm(x + gated_attn)


class Expert(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dim_ff: int, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({
            'attn': GatedAttention(embed_dim, nhead, dropout),
            'ffn': nn.Sequential(nn.Linear(embed_dim, dim_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_ff, embed_dim)),
            'norm1': nn.LayerNorm(embed_dim), 'norm2': nn.LayerNorm(embed_dim), 'dropout': nn.Dropout(dropout)
        }) for _ in range(num_layers)])
        # Specialization tag is kept, might help experts specialize
        self.specialization_tag = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.01)

        # Adaptive Gate (Enhancement 1 from original)
        # Gate linear layer takes mean of processed output, outputs a scalar gate value per sequence
        self.gate_linear = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, seq_len, embed_dim)
        # src_key_padding_mask: (batch_size, seq_len) - True for padding
        x_input = x  # Save original input for skip connection in adaptive gate

        # Core expert processing
        for mod in self.layers:
            # Apply attention with residual and norm, passing the mask
            x = x + mod['dropout'](mod['attn'](mod['norm1'](x), key_padding_mask=src_key_padding_mask))
            # Apply FFN with residual and norm
            x = x + mod['dropout'](mod['ffn'](mod['norm2'](x)))

        # Apply adaptive gate
        # Gate value based on the mean of the processed sequence features
        # x.mean(dim=1, keepdim=True) -> (batch_size, 1, embed_dim)
        # self.gate_linear(...) -> (batch_size, 1, 1)
        gate_val = torch.sigmoid(self.gate_linear(x.mean(dim=1, keepdim=True)))

        # Modulate output: gate * processed_output + (1-gate) * original_input
        # gate_val is (batch_size, 1, 1), x and x_input are (batch_size, seq_len, embed_dim)
        # Broadcasting handles this correctly.
        x_gated = gate_val * x + (1 - gate_val) * x_input

        # Add learnable tag after gating
        # specialization_tag is (1, 1, embed_dim), broadcasts to (batch_size, seq_len, embed_dim)
        return x_gated + self.specialization_tag


class RoutingController(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        # Router predicts logits for each expert + termination
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts + 1)  # +1 for termination action
        self.norm = nn.LayerNorm(num_experts + 1) # Normalize logits before softmax/gumbel

        # Q-values are removed as we are simplifying the RL aspect for framework integration
        # self.q_values = nn.Parameter(torch.zeros(num_experts + 1))

    # visit_counts and max_visits are kept for masking
    def forward(self, x_summary: torch.Tensor, visit_counts: torch.Tensor, max_visits: int) -> torch.Tensor:
        # x_summary: (batch_size, input_dim)
        # visit_counts: (batch_size, num_experts)
        # max_visits: scalar int
        x = F.gelu(self.fc1(x_summary))
        logits = self.fc2(x) # (batch_size, num_experts + 1)

        # Masking based on visit counts
        # Create a mask for experts that have been visited max_visits times
        mask = visit_counts >= max_visits # (batch_size, num_experts)
        # Termination action (last logit) is not masked by visit counts
        # Extend mask to include the termination logit column (all False for termination)
        extended_mask = torch.cat([mask, torch.zeros_like(mask[:, :1], dtype=torch.bool, device=mask.device)], dim=-1) # (batch_size, num_experts + 1)

        # Apply masking: set logits of masked actions to -inf
        final_logits = self.norm(logits).masked_fill(extended_mask, -float('inf'))

        return final_logits


# RunningStat is kept for potential metric normalization if needed, but not for reward normalization here
class RunningStat:
    def __init__(self):
        self.mean, self.m2, self.count = 0.0, 0.0, 0

    def update(self, x_tensor: torch.Tensor):
        if x_tensor.numel() == 0: return
        # Process elements one by one for Welford's algorithm stability
        for x in x_tensor.detach().cpu().flatten().tolist():
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (x - self.mean)

    @property
    def var(self) -> float:
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Avoid division by zero if std is 0
        return (x - self.mean) / (self.std + 1e-8)


# --- Main Model (Adapted for Classification) ---
# Inherit from EvaluableModel
from .base_model import EvaluableModel

class GoEOriginalClassifier(EvaluableModel):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 dim_feedforward: int, num_classes: int, # num_classes for classification
                 num_total_experts: int, max_path_len: int, router_hidden_dim: int,
                 expert_layers: int, # num_layers for experts
                 dropout: float = 0.1, padding_idx: int = 0, gumbel_tau: float = 1.0,
                 path_penalty_coef: float = 0.005, # Path length penalty as a loss term
                 diversity_loss_coef: float = 0.02, # Diversity loss coefficient
                 contrastive_loss_coef: float = 0.05, # Contrastive loss coefficient
                 max_visits_per_expert: int = 2 # Added max_visits_per_expert HP
                 ):
        super().__init__()
        self.model_name = "GoEOriginalClassifier"
        self.padding_idx = padding_idx
        self.vocab_size, self.embed_dim, self.num_classes = vocab_size, embed_dim, num_classes
        self.num_total_experts = num_total_experts
        self.max_path_len = max_path_len
        self.max_visits_per_expert = max_visits_per_expert # Store new HP
        self.gumbel_tau = gumbel_tau
        self.path_penalty_coef = path_penalty_coef # Renamed for clarity as it's a loss term
        self.diversity_coef = diversity_loss_coef
        self.contrastive_coef = contrastive_loss_coef

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.padding_idx)
        # Max length for positional encoding should be sufficient for input sequences
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout, max_len=512) # Use a reasonable default max_len

        # Experts use the adapted Expert module
        self.experts = nn.ModuleList([
            Expert(embed_dim, num_heads, dim_feedforward, expert_layers, dropout=dropout)
            for _ in range(self.num_total_experts)
        ])

        # Router input dimension: mean + max pooling (Enhancement 2 from original)
        router_input_dim = embed_dim * 2
        self.router = RoutingController(router_input_dim, router_hidden_dim, self.num_total_experts)

        # Final classification head
        self.fc = nn.Linear(embed_dim, num_classes)

        self.input_norm = nn.LayerNorm(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout_embed = nn.Dropout(dropout)

        # Metrics and auxiliary loss storage
        self._aux_loss_value = torch.tensor(0.0)
        self._avg_router_entropy = torch.tensor(0.0)
        self._avg_path_length = torch.tensor(0.0)
        self._expert_usage_cv = torch.tensor(0.0)
        self._expert_usage_counts = torch.zeros(self.num_total_experts)


    def _compute_diversity_loss(self, router_probs_list: List[torch.Tensor], expert_usage_counts: torch.Tensor) -> torch.Tensor:
        # router_probs_list: List of (N_active, num_experts + 1) tensors from each step
        # expert_usage_counts: (num_experts) - total counts across batch and steps
        device = expert_usage_counts.device
        if not router_probs_list: return torch.tensor(0.0, device=device)

        # Calculate average router probabilities over all steps and active samples
        all_router_probs = torch.cat([p[:, :self.num_total_experts] for p in router_probs_list], dim=0) # (Total_N_decisions, num_experts)
        if all_router_probs.numel() == 0: return torch.tensor(0.0, device=device)

        avg_probs = all_router_probs.mean(dim=0) # (num_experts)
        # Entropy of average probabilities
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))

        # Load balancing loss based on actual expert usage counts
        load_balance_cv = calculate_cv(expert_usage_counts.float())

        # Combine entropy and load balancing (CV^2)
        return -entropy + load_balance_cv.pow(2)


    def _compute_contrastive_loss(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        # expert_outputs: List of tensors, each (batch_size_subset, seq_len, embed_dim)
        model_dev = self.embedding.weight.device
        if not expert_outputs: return torch.tensor(0.0, device=model_dev)

        # Calculate mean pooling for each collected expert output tensor
        all_pooled_outputs = torch.cat([o.mean(dim=1) for o in expert_outputs], dim=0) # (Total_N, embed_dim)
        if all_pooled_outputs.size(0) < 2: return torch.tensor(0.0, device=model_dev)

        # Normalize the pooled outputs
        normed = F.normalize(all_pooled_outputs, p=2, dim=1) # (Total_N, embed_dim)

        # Compute pairwise similarity matrix
        sim = torch.matmul(normed, normed.t()) # (Total_N, Total_N)

        # Target is identity matrix (each output should be orthogonal to others)
        target = torch.eye(sim.size(0), device=sim.device) # (Total_N, Total_N)

        # Mean squared error between similarity matrix and identity matrix
        return (sim - target).pow(2).mean()


    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len) - 1 for real token, 0 for padding

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create key padding mask from attention_mask or padding_idx
        if attention_mask is None:
            src_key_padding_mask = (input_ids == self.padding_idx) # (batch_size, seq_len) - True for padding
        else:
            src_key_padding_mask = (attention_mask == 0) # (batch_size, seq_len) - True for padding

        # Embedding and Positional Encoding
        # Scale embedding by sqrt(embed_dim) as in original Transformer/goe_original
        x = self.dropout_embed(self.embedding(input_ids) * math.sqrt(self.embed_dim))
        # Apply PE and input norm
        x = self.input_norm(self.pos_encoder(x))


        current_data_repr = x.clone() # Representation that gets updated through the path

        # Track state for auxiliary losses and metrics
        router_probs_list: List[torch.Tensor] = [] # Collect router probabilities per step
        expert_usage_counts_total = torch.zeros(self.num_total_experts, dtype=torch.long, device=device)
        path_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        visits = torch.zeros((batch_size, self.num_total_experts), dtype=torch.long, device=device)
        expert_outputs_collected: List[torch.Tensor] = [] # Collect outputs for contrastive loss

        # --- Path Traversal ---
        # Keep track of which samples are still active.
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        active_indices = torch.arange(batch_size, device=device)

        for step_idx in range(self.max_path_len):
            if not active_indices.numel(): break # Stop if no samples are active

            # Select active samples' data and router state
            current_active_repr = current_data_repr[active_indices] # (N_active, seq_len, embed_dim)
            visits_active = visits[active_indices] # (N_active, num_experts)
            src_key_padding_mask_active = src_key_padding_mask[active_indices] # (N_active, seq_len)

            # Enhancement 2: Improved summary for router (mean + max pooling)
            mean_summary = current_active_repr.mean(dim=1) # (N_active, embed_dim)
            max_summary, _ = current_active_repr.max(dim=1) # (N_active, embed_dim)
            summary_for_router = torch.cat((mean_summary, max_summary), dim=-1) # (N_active, embed_dim * 2)

            # Get router logits for active samples (Router is stateless)
            # Pass max_visits_per_expert to router forward for masking
            expert_logits_active = self.router(
                summary_for_router, visits_active, self.max_visits_per_expert
            ) # expert_logits_active: (N_active, num_experts + 1)

            # Store router probabilities for diversity loss calculation later
            router_probs_list.append(F.softmax(expert_logits_active, dim=-1))

            # Gumbel-Softmax for routing decision
            routing_decision_one_hot_active = gumbel_softmax(
                expert_logits_active, tau=self.gumbel_tau, hard=self.training, dim=-1
            ) # (N_active, num_experts + 1)

            # Separate termination decision from expert selection
            term_decision_active = routing_decision_one_hot_active[:, self.num_total_experts] # (N_active)
            expert_sel_active = routing_decision_one_hot_active[:, :self.num_total_experts] # (N_active, num_experts)

            # --- Process Termination ---
            # Samples where term_decision_active is 1 will stop traversing the path.
            # Their representation is finalized as their current_active_repr.
            # They are removed from the set of active samples for the next step.
            terminate_mask_step = (term_decision_active == 1) # (N_active)
            # Update the global active_mask based on which original indices terminated
            active_mask[active_indices[terminate_mask_step]] = False

            # --- Process Expert Selection ---
            # Samples where term_decision_active is 0 continue.
            # They select an expert based on expert_sel_active.
            continue_mask_step = ~terminate_mask_step # (N_active)
            continue_indices_in_active = torch.where(continue_mask_step)[0] # Indices within the active_indices tensor
            continue_orig_indices = active_indices[continue_indices_in_active] # Original indices that continue

            if continue_orig_indices.numel() > 0:
                # Get the one-hot expert selection for continuing samples
                expert_sel_continue = expert_sel_active[continue_indices_in_active] # (N_continue, num_experts)
                # Get the chosen expert index for each continuing sample
                chosen_expert_idx_continue = expert_sel_continue.argmax(dim=1) # (N_continue)

                # Update path lengths and visit counts for continuing samples
                path_lengths[continue_orig_indices] += 1
                # Use scatter_add to update visit counts efficiently
                visits[continue_orig_indices].scatter_add_(1, chosen_expert_idx_continue.unsqueeze(-1), torch.ones_like(chosen_expert_idx_continue.unsqueeze(-1)))

                # Update total expert usage counts for diversity loss metric
                expert_usage_counts_total.scatter_add_(0, chosen_expert_idx_continue, torch.ones_like(chosen_expert_idx_continue))

                # Process data through the chosen experts
                # Update current_data_repr directly using original indices
                for exp_idx in range(self.num_total_experts):
                    # Find samples that chose this expert
                    expert_chosen_mask = (chosen_expert_idx_continue == exp_idx) # (N_continue)
                    expert_chosen_orig_indices = continue_orig_indices[expert_chosen_mask]

                    if expert_chosen_orig_indices.numel() > 0:
                        # Get the data representation for these samples
                        data_for_expert = current_data_repr[expert_chosen_orig_indices] # (N_expert_subset, seq_len, embed_dim)
                        # Get the padding mask for these samples
                        mask_for_expert = src_key_padding_mask[expert_chosen_orig_indices] # (N_expert_subset, seq_len)

                        # Process through the expert, passing the mask
                        expert_out = self.experts[exp_idx](data_for_expert, src_key_padding_mask=mask_for_expert) # (N_expert_subset, seq_len, embed_dim)

                        # Update representation for continuing samples
                        current_data_repr[expert_chosen_orig_indices] = expert_out

                        # Collect expert outputs for contrastive loss
                        expert_outputs_collected.append(expert_out)

                active_indices = continue_orig_indices # Update active indices for next step
            else:
                active_indices = torch.tensor([], dtype=torch.long, device=device)
                break # All samples terminated

        # --- Final Processing ---
        # The final representation for each sample is its last updated representation in current_data_repr.
        # Samples that terminated early retain their representation from the step they terminated.
        # Samples that reached max_path_len use their representation after the last expert step.

        final_repr = self.final_norm(current_data_repr) # Apply final norm

        # Global average pooling over sequence length for classification
        pooled_output = final_repr.mean(dim=1) # (batch_size, embed_dim)

        # Classification logits
        logits = self.fc(pooled_output) # (batch_size, num_classes)

        # --- Calculate Final Auxiliary Loss and Metrics ---
        # Average router entropy over all routing decisions made
        # Calculate average entropy per step and then average over steps
        avg_entropies_per_step = [calculate_entropy(p[:, :self.num_total_experts], eps=1e-10) for p in router_probs_list]
        self._avg_router_entropy = torch.mean(torch.stack(avg_entropies_per_step)) if avg_entropies_per_step else torch.tensor(0.0, device=device)


        # Diversity loss based on total expert usage counts across all steps and samples
        diversity_loss = self._compute_diversity_loss(router_probs_list, expert_usage_counts_total.float())
        self._expert_usage_cv = calculate_cv(expert_usage_counts_total.float())


        # Contrastive loss on collected expert outputs
        contrastive_loss = self._compute_contrastive_loss(expert_outputs_collected)

        # Path length penalty: penalize longer paths
        # Average path length over the batch
        avg_path_len = path_lengths.float().mean()
        path_length_penalty = avg_path_len * self.path_penalty_coef
        self._avg_path_length = avg_path_len.detach()

        # Total auxiliary loss
        self._aux_loss_value = (
            self.diversity_coef * diversity_loss +
            self.contrastive_coef * contrastive_loss +
            path_length_penalty
        )

        # Store expert usage counts for metrics
        self._expert_usage_counts = expert_usage_counts_total.detach()


        return logits # Return only logits for the framework's criterion


    def get_parameter_count(self) -> int:
        return self.get_trainable_parameter_count()

    def get_model_name(self) -> str:
        return self.model_name

    def get_auxiliary_loss(self):
        # Return the calculated auxiliary loss value
        return self._aux_loss_value

    def get_model_specific_metrics(self):
        # Return stored metrics
        metrics = {
            'goe_original_avg_router_entropy': self._avg_router_entropy.item(),
            'goe_original_avg_path_length': self._avg_path_length.item(),
            'goe_original_expert_usage_cv': self._expert_usage_cv.item(),
        }
        # Add expert usage counts as a list
        metrics['goe_original_expert_usage_counts'] = self._expert_usage_counts.tolist()
        return metrics
