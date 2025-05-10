import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import EvaluableModel
from ..utils import gumbel_softmax, calculate_entropy # Relative imports

class ExpertModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout):
        super().__init__()
        # Using norm_first for stability
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True, norm_first=True 
        )
    def forward(self, x, src_key_padding_mask=None):
        return self.transformer_layer(x, src_key_padding_mask=src_key_padding_mask)

class Router(nn.Module):
    def __init__(self, embed_dim, router_hidden_dim, num_total_experts):
        super().__init__()
        # Using GRUCell as it's simpler than LSTMCell and often works well
        self.rnn_cell = nn.GRUCell(embed_dim, router_hidden_dim)
        self.fc_expert_logits = nn.Linear(router_hidden_dim, num_total_experts)
        self.router_hidden_dim = router_hidden_dim

    def forward(self, summary_input, h_prev): # summary_input: (batch, embed_dim), h_prev: (batch, router_hidden_dim)
        h_next = self.rnn_cell(summary_input, h_prev)
        expert_logits = self.fc_expert_logits(h_next) # (batch, num_total_experts)
        return expert_logits, h_next

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.router_hidden_dim, device=device)

class GoEClassifier(EvaluableModel):
    def __init__(self, vocab_size, embed_dim, num_heads, dim_feedforward, num_classes,
                 num_total_experts, max_path_len, router_hidden_dim, dropout=0.1, padding_idx=0, gumbel_tau=1.0):
        super().__init__()
        self.model_name = "GoEClassifier"
        self.padding_idx = padding_idx
        self.embed_dim = embed_dim
        self.max_path_len = max_path_len
        self.num_total_experts = num_total_experts
        self.gumbel_tau = gumbel_tau

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.padding_idx)
        self.experts = nn.ModuleList([
            ExpertModule(embed_dim, num_heads, dim_feedforward, dropout) 
            for _ in range(num_total_experts)
        ])
        self.router = Router(embed_dim, router_hidden_dim, num_total_experts)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout_embed = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(embed_dim)

        self.aux_loss_value = torch.tensor(0.0)
        self.avg_router_entropy = torch.tensor(0.0)
        self.expert_selections_over_path = []


    def forward(self, input_ids, attention_mask=None):
        batch_size, _ = input_ids.shape
        device = input_ids.device
        src_key_padding_mask = (input_ids == self.padding_idx) if attention_mask is None else (attention_mask == 0)

        current_data_repr = self.dropout_embed(self.embedding(input_ids))
        
        h_router = self.router.init_state(batch_size, device)
        
        total_router_entropy = 0.0
        path_expert_selections = [] # List of tensors, each (batch_size, num_experts) one-hot

        for _ in range(self.max_path_len):
            summary_for_router = current_data_repr.mean(dim=1) # Global average pooling for router input
            expert_logits, h_router = self.router(summary_for_router, h_router)
            
            # Router entropy regularization (auxiliary loss)
            router_probs = F.softmax(expert_logits, dim=-1)
            total_router_entropy += calculate_entropy(router_probs)
            
            # Hard Gumbel-Softmax for selecting ONE expert per item in batch
            routing_decision_one_hot = gumbel_softmax(expert_logits, tau=self.gumbel_tau, hard=True, dim=-1)
            path_expert_selections.append(routing_decision_one_hot.detach().cpu()) # For metrics

            # Weighted sum of expert outputs using one-hot weights (effectively selects one expert)
            # This is a batch-friendly way to implement hard routing.
            # All experts process the full `current_data_repr`, then outputs are masked and summed.
            # This is less efficient than scatter-gather but simpler.
            step_output_accumulator = torch.zeros_like(current_data_repr)
            for i in range(self.num_total_experts):
                # expert_out: (batch, seq_len, embed_dim)
                expert_out = self.experts[i](current_data_repr, src_key_padding_mask=src_key_padding_mask)
                # routing_decision_one_hot[:, i] is (batch_size), need (batch_size, 1, 1) for broadcast
                selection_mask = routing_decision_one_hot[:, i].unsqueeze(-1).unsqueeze(-1)
                step_output_accumulator += expert_out * selection_mask
            current_data_repr = step_output_accumulator

        self.aux_loss_value = -total_router_entropy / self.max_path_len # Maximize entropy = Minimize negative entropy
        self.avg_router_entropy = (total_router_entropy / self.max_path_len).detach()
        self.expert_selections_over_path = path_expert_selections # Store for metrics

        current_data_repr = self.final_norm(current_data_repr)
        pooled_output = current_data_repr.mean(dim=1)
        return self.fc(pooled_output)

    def get_parameter_count(self) -> int: return self.get_trainable_parameter_count()
    def get_model_name(self) -> str: return self.model_name
    def get_auxiliary_loss(self): return self.aux_loss_value

    def get_model_specific_metrics(self):
        metrics = {'goe_avg_router_entropy': self.avg_router_entropy.item()}
        if self.expert_selections_over_path:
            # Calculate expert utilization at each step
            for step_idx, selections_one_hot in enumerate(self.expert_selections_over_path):
                # selections_one_hot is (batch, num_experts)
                expert_counts_at_step = selections_one_hot.sum(dim=0) # (num_experts)
                metrics[f'goe_step_{step_idx}_expert_util_cv'] = calculate_cv(expert_counts_at_step.float()).item()
        self.expert_selections_over_path = [] # Clear after fetching
        return metrics
