import torch.nn as nn
from .base_model import EvaluableModel

class DenseTransformerClassifier(EvaluableModel):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dim_feedforward, num_classes, dropout=0.1, padding_idx=0):
        super().__init__()
        self.model_name = "DenseTransformer"
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.padding_idx)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True) # norm_first for stability
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        src_key_padding_mask = (input_ids == self.padding_idx) if attention_mask is None else (attention_mask == 0)
        
        embedded = self.dropout(self.embedding(input_ids))
        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        
        pooled_output = transformer_output.mean(dim=1) # CLS token pooling could also be an option
        return self.fc(pooled_output)

    def get_parameter_count(self) -> int: return self.get_trainable_parameter_count()
    def get_model_name(self) -> str: return self.model_name
