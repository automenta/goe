import math
import sys
import time
import random
import collections
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QSizePolicy, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF

# --- Constants and Configuration ---

# Language Model Data
WIKITEXT_SAMPLE = """
= Valkyria Chronicles III = 

Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria Chronicles series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the original Valkyria Chronicles and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who are assigned to sensitive operations and missions deemed too dangerous for the regular army . 
The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's theme is " despair " , with the plot exploring the Gallian military 's persecution of those deemed undesirable and the Nameless 's struggle for redemption . 
Upon release , the game received positive reviews and was praised by both Japanese and western critics . After release , it received downloadable content , as well as manga and drama CD adaptations . Despite the positive reception , Valkyria Chronicles III was not localized for Western territories . The decision was attributed to the poor sales of Valkyria Chronicles II , a troubled history with the PlayStation Portable , and the popularity of mobile gaming in the West . Various English fan translation projects were initiated , but were left unfinished due to the complexity of the game 's script and coding . A full English fan translation was eventually released in 2014 .
"""
MAX_VOCAB_SIZE = 2000
MIN_FREQ = 2
MAX_LM_SEQ_LEN = 10  # Length of input sequence to predict the next word

# Model HPs
EMBED_DIM_DEFAULT = 64
NUM_EXPERTS_DEFAULT = 3
EXPERT_NHEAD_DEFAULT = 2  # For TransformerEncoderLayer in Expert
EXPERT_DIM_FEEDFORWARD_DEFAULT = 128  # For TransformerEncoderLayer in Expert
ROUTER_HIDDEN_DIM_DEFAULT = 64
MAX_PATH_LEN_DEFAULT = 4
MAX_VISITS_PER_EXPERT_IN_PATH_DEFAULT = 1

# Training HPs
LEARNING_RATE = 1e-3
BATCH_SIZE = 8  # Smaller batch for potentially larger models / slower steps
ROUTER_ENTROPY_COEF = 0.01
WEIGHT_DECAY = 1e-4

# Animation
ANIMATION_DELAY_DEFAULT = 0.8


# Globals to be updated by UI - these will be set by MainWindow._get_current_hps()
# and passed to model/worker initialization.
# We avoid direct global modification from UI events for clarity.

# --- Dataset and Preprocessing ---
class Vocabulary:
    def __init__(self, max_size=None, min_freq=1):
        self.word_to_idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.word_counts = collections.Counter()
        self.max_size = max_size
        self.min_freq = min_freq
        self._finalized = False

    def add_word(self, word):
        if self._finalized:
            raise RuntimeError("Vocabulary is finalized. Cannot add new words.")
        self.word_counts[word] += 1

    def build_vocab(self):
        if self._finalized: return

        sorted_words = self.word_counts.most_common()

        for word, count in sorted_words:
            if self.max_size and len(self.word_to_idx) >= self.max_size:
                print(f"Warning: Vocab truncated at {self.max_size} words.")
                break
            if count < self.min_freq and word not in self.word_to_idx:  # keep special tokens
                continue
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        self._finalized = True
        print(f"Vocabulary built with {len(self.word_to_idx)} words.")

    def __len__(self):
        return len(self.word_to_idx)

    def numericalize(self, text_tokens: List[str]) -> List[int]:
        if not self._finalized:
            raise RuntimeError("Vocabulary not built yet. Call build_vocab().")
        return [self.word_to_idx.get(token, self.word_to_idx["<unk>"]) for token in text_tokens]

    def denumericalize(self, indices: List[int]) -> List[str]:
        return [self.idx_to_word.get(idx, "<unk>") for idx in indices]


def tokenize_text(text: str) -> List[str]:
    return text.lower().split()


def create_lm_sequences(token_ids: List[int], seq_len: int) -> List[Tuple[List[int], int]]:
    sequences = []
    if not token_ids: return sequences
    for i in range(len(token_ids) - seq_len):
        input_seq = token_ids[i: i + seq_len]
        target_word = token_ids[i + seq_len]
        sequences.append((input_seq, target_word))
    return sequences


class LanguageModelDataset(torch.utils.data.Dataset):
    def __init__(self, text_data: str, vocab: Vocabulary, seq_len: int):
        self.vocab = vocab
        self.seq_len = seq_len

        tokens = tokenize_text(text_data)
        if not self.vocab._finalized:  # Build vocab from this data if not already done
            for token in tokens:
                self.vocab.add_word(token)
            self.vocab.build_vocab()

        numericalized_tokens = self.vocab.numericalize(tokens)
        self.sequences = create_lm_sequences(numericalized_tokens, self.seq_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_word = self.sequences[idx]
        # For animation/logging, get original text snippet
        original_text_snippet = " ".join(self.vocab.denumericalize(input_seq)) + \
                                " ...predict... " + self.vocab.denumericalize([target_word])[0]

        return (torch.tensor(input_seq, dtype=torch.long),
                torch.tensor(target_word, dtype=torch.long),
                original_text_snippet)


# --- PyTorch Model Components ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # .transpose(0, 1) -> (max_len, 1, d_model) if not batch_first
        self.register_buffer('pe', pe)  # (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Expert(nn.Module):  # Now a Transformer Encoder Layer
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout=0.1, expert_id=0):
        super().__init__()
        self.expert_id = expert_id
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: input (N, S, E)
        )
        # Optional: Add a small "specialization" vector
        self.specialization_tag = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.01)

    def forward(self, x_seq):
        # x_seq shape: (batch_size, seq_len, embed_dim)
        out_seq = self.transformer_layer(x_seq)
        return out_seq + self.specialization_tag  # Broadcast over batch and seq_len


class RoutingController(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_experts + 1)  # +1 for termination

    def forward(self, x_summary, current_visit_counts_for_batch_item, max_visits_per_expert):
        # x_summary shape: (batch_size, embed_dim)
        x = self.fc1(x_summary)
        x = self.relu(x)
        logits = self.fc2(x)

        # Mask experts based on visit counts
        if x_summary.dim() == 1:  # Single sample (unbatched or during animation focus)
            for expert_idx in range(self.num_experts):
                if current_visit_counts_for_batch_item[expert_idx] >= max_visits_per_expert:
                    logits[expert_idx] = -float('inf')
        elif x_summary.dim() == 2:  # Batched processing
            for i in range(logits.size(0)):  # Iterate over batch items
                for expert_idx in range(self.num_experts):
                    if current_visit_counts_for_batch_item[i, expert_idx] >= max_visits_per_expert:
                        logits[i, expert_idx] = -float('inf')
        return logits


class GoEModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_experts, expert_nhead, expert_dim_feedforward,
                 router_hidden_dim, max_path_len, max_visits_per_expert, max_lm_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.max_path_len = max_path_len
        self.max_visits_per_expert_in_path = max_visits_per_expert

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Assuming vocab.word_to_idx["<pad>"] == 0
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_lm_seq_len + 5)  # +5 for safety

        self.experts = nn.ModuleList([
            Expert(embed_dim, expert_nhead, expert_dim_feedforward, expert_id=i)
            for i in range(num_experts)
        ])
        self.router = RoutingController(embed_dim, router_hidden_dim, num_experts)
        self.output_lm_head = nn.Linear(embed_dim, vocab_size)

        self.animation_signals = None
        self.is_animating = False

    def _emit_signal(self, signal_type: str, *args):
        if self.is_animating and self.animation_signals:
            # Use the global ANIMATION_DELAY which is updated by UI
            delay = self.animation_signals.get_animation_delay_value()
            getattr(self.animation_signals, signal_type).emit(*args)
            if delay > 0: QThread.msleep(int(delay * 1000))

    def forward(self, input_ids_seq, current_sample_text_for_anim=None):
        # input_ids_seq shape: (batch_size, seq_len)
        batch_size = input_ids_seq.size(0)

        if self.is_animating and current_sample_text_for_anim:
            self._emit_signal("signal_input_text", current_sample_text_for_anim)

        embedded_input = self.embedding(input_ids_seq) * math.sqrt(self.embed_dim)
        current_representation_seq = self.pos_encoder(embedded_input)  # (batch, seq_len, embed_dim)

        if self.is_animating:
            # For animation, show mean-pooled representation
            self._emit_signal("signal_embedded", repr_tensor(current_representation_seq[0].mean(dim=0)))

        total_router_entropy = torch.tensor(0.0, device=input_ids_seq.device)
        paths_taken_indices = [[] for _ in range(batch_size)]
        current_visit_counts_batch = torch.zeros((batch_size, self.num_experts), dtype=torch.long,
                                                 device=input_ids_seq.device)

        active_indices_original = torch.arange(batch_size, device=input_ids_seq.device)
        # Store the full sequence representation for each sample
        final_representations_seq_batch = torch.zeros_like(current_representation_seq)

        current_active_representations_seq = current_representation_seq
        current_active_indices = active_indices_original.clone()

        for step in range(self.max_path_len):
            if not current_active_indices.numel():
                break

            # Router operates on a summary (mean-pooled) of the sequence
            current_active_summary_repr = current_active_representations_seq.mean(dim=1)  # (active_batch, embed_dim)

            anim_sample_original_idx = 0  # Focus on the first sample of the original batch for animation
            is_anim_sample_active = anim_sample_original_idx in current_active_indices

            if self.is_animating and is_anim_sample_active:
                anim_sample_current_batch_idx = \
                (current_active_indices == anim_sample_original_idx).nonzero(as_tuple=True)[0].item()
                anim_repr_summary = repr_tensor(current_active_summary_repr[anim_sample_current_batch_idx])
                anim_path = [str(p) for p in paths_taken_indices[anim_sample_original_idx]]
                self._emit_signal("signal_to_router", anim_repr_summary, anim_path)

            router_logits = self.router(current_active_summary_repr,
                                        current_visit_counts_batch[current_active_indices],
                                        self.max_visits_per_expert_in_path)

            router_probs = F.softmax(router_logits, dim=1)
            step_entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-9), dim=1).mean()
            total_router_entropy += step_entropy

            if self.training:
                one_hot_decision = F.gumbel_softmax(router_logits, tau=1.0, hard=True)
            else:  # Inference
                decision_indices = torch.argmax(router_logits, dim=1)
                one_hot_decision = F.one_hot(decision_indices, num_classes=self.num_experts + 1).float()

            terminate_decision = one_hot_decision[:, self.num_experts]  # Last column is termination
            expert_selection_one_hot = one_hot_decision[:, :self.num_experts]

            if self.is_animating and is_anim_sample_active:
                anim_sample_current_batch_idx = \
                (current_active_indices == anim_sample_original_idx).nonzero(as_tuple=True)[0].item()
                flat_probs_anim = router_probs[anim_sample_current_batch_idx].tolist()
                selected_idx_anim = one_hot_decision[anim_sample_current_batch_idx].argmax().item()
                is_terminate_anim = (selected_idx_anim == self.num_experts)
                self._emit_signal("signal_router_output", flat_probs_anim, selected_idx_anim, is_terminate_anim)

            terminated_mask_in_active = (terminate_decision == 1)
            if terminated_mask_in_active.any():
                original_indices_of_terminated_now = current_active_indices[terminated_mask_in_active]
                final_representations_seq_batch[original_indices_of_terminated_now] = \
                    current_active_representations_seq[terminated_mask_in_active]
                for orig_idx in original_indices_of_terminated_now.tolist():
                    paths_taken_indices[orig_idx].append('T')

            chose_expert_mask_in_active = (terminate_decision == 0)
            if not chose_expert_mask_in_active.any():
                current_active_indices = torch.tensor([], dtype=torch.long, device=input_ids_seq.device)
                break

            next_active_indices = current_active_indices[chose_expert_mask_in_active]
            current_active_representations_seq = current_active_representations_seq[chose_expert_mask_in_active]
            active_expert_selection_one_hot = expert_selection_one_hot[chose_expert_mask_in_active]
            chosen_expert_indices_for_active = active_expert_selection_one_hot.argmax(dim=1)

            for i, orig_idx in enumerate(next_active_indices.tolist()):
                expert_idx_chosen = chosen_expert_indices_for_active[i].item()
                paths_taken_indices[orig_idx].append(expert_idx_chosen)
                current_visit_counts_batch[orig_idx, expert_idx_chosen] += 1

            next_step_representations_seq_for_active = torch.zeros_like(current_active_representations_seq)
            for i, expert_idx_val in enumerate(chosen_expert_indices_for_active.tolist()):
                is_anim_sample_processing_expert = (self.is_animating and is_anim_sample_active and
                                                    next_active_indices[i].item() == anim_sample_original_idx)

                current_input_to_expert_seq = current_active_representations_seq[i].unsqueeze(0)  # Expert expects batch

                if is_anim_sample_processing_expert:
                    self._emit_signal("signal_to_expert", expert_idx_val,
                                      repr_tensor(
                                          current_input_to_expert_seq.mean(dim=(0, 1))))  # Mean of seq for display

                expert_output_seq = self.experts[expert_idx_val](current_input_to_expert_seq)  # (1, seq_len, embed_dim)
                next_step_representations_seq_for_active[i] = expert_output_seq.squeeze(0)

                if is_anim_sample_processing_expert:
                    self._emit_signal("signal_expert_output", expert_idx_val,
                                      repr_tensor(
                                          next_step_representations_seq_for_active[i].mean(dim=0)))  # Mean of seq

            current_active_representations_seq = next_step_representations_seq_for_active
            current_active_indices = next_active_indices

        if current_active_indices.numel() > 0:  # Reached max_path_len
            final_representations_seq_batch[current_active_indices] = current_active_representations_seq
            for orig_idx in current_active_indices.tolist():
                paths_taken_indices[orig_idx].append('T_max')

        # Use the mean of the final sequence representation for LM head
        final_summary_repr_batch = final_representations_seq_batch.mean(dim=1)  # (batch_size, embed_dim)

        if self.is_animating:
            self._emit_signal("signal_to_output_layer", repr_tensor(final_summary_repr_batch[0]))

        logits = self.output_lm_head(final_summary_repr_batch)  # (batch_size, vocab_size)

        if self.is_animating:
            predicted_token_idx = logits[0].argmax().item()
            predicted_token_str = self.animation_signals.vocab.denumericalize([predicted_token_idx])[0]
            self._emit_signal("signal_final_prediction_lm", predicted_token_str, paths_taken_indices[0])

        return logits, total_router_entropy, paths_taken_indices


# --- PyQt Components ---
def repr_tensor(tensor_data, num_elems=4):
    if tensor_data is None: return "None"
    if isinstance(tensor_data, list):
        return "[" + ", ".join([f"{x:.2f}" for x in tensor_data[:num_elems]]) + (
            "..." if len(tensor_data) > num_elems else "") + "]"
    if not isinstance(tensor_data, torch.Tensor) or tensor_data.numel() == 0:
        return str(tensor_data)

    items = tensor_data.detach().cpu().flatten().tolist()
    if len(items) > num_elems:
        return "[" + ", ".join([f"{x:.2f}" for x in items[:num_elems]]) + f"... L={len(items)}]"
    else:
        return "[" + ", ".join([f"{x:.2f}" for x in items]) + "]"


class DiagramWidget(QWidget):
    def __init__(self, num_experts=NUM_EXPERTS_DEFAULT):
        super().__init__()
        self.setMinimumSize(600, 550)  # Increased height for LM
        self.node_positions = {}
        self.active_elements = {}
        self.connections_to_draw = []
        self.current_representation_text = ""
        self.router_probs_text = ""
        self.current_path_text = ""
        self.num_experts_to_draw = num_experts
        self._setup_node_positions()

    def update_num_experts(self, num_experts):
        self.num_experts_to_draw = num_experts
        self._setup_node_positions()
        self.reset_highlights()  # Full redraw basically

    def _setup_node_positions(self):
        width = self.width() if self.width() > 0 else 600
        height = self.height() if self.height() > 0 else 550

        self.node_size = QRectF(0, 0, 100, 35)  # Slightly smaller nodes

        self.node_positions["INPUT"] = QPointF(width / 2, height * 0.08)
        self.node_positions["EMBED"] = QPointF(width / 2, height * 0.22)  # PosEnc included here
        self.node_positions["ROUTER"] = QPointF(width / 2, height * 0.40)

        expert_y = height * 0.60
        if self.num_experts_to_draw == 1:  # Should not happen based on UI constraints (min 2)
            self.node_positions["EXPERT_0"] = QPointF(width * 0.5, expert_y)
        else:
            for i in range(self.num_experts_to_draw):
                # Spread experts horizontally
                x_pos = width * (
                    0.15 + (0.7 / (self.num_experts_to_draw - 1 + 1e-9)) * i if self.num_experts_to_draw > 1 else 0.5)
                self.node_positions[f"EXPERT_{i}"] = QPointF(x_pos, expert_y)

        self.node_positions["LM_HEAD"] = QPointF(width / 2, height * 0.85)  # Was CLASSIFIER

        self.base_connections = [("INPUT", "EMBED"), ("EMBED", "ROUTER")]
        for i in range(self.num_experts_to_draw):
            self.base_connections.append(("ROUTER", f"EXPERT_{i}"))
            self.base_connections.append((f"EXPERT_{i}", "ROUTER"))
        self.base_connections.append(("ROUTER", "LM_HEAD"))

        self.reset_highlights()

    def resizeEvent(self, event):
        self._setup_node_positions()
        super().resizeEvent(event)

    def reset_highlights(self):
        self.active_elements.clear()
        self.connections_to_draw.clear()
        for src, dest in self.base_connections:
            if src in self.node_positions and dest in self.node_positions:  # Ensure nodes exist
                self.connections_to_draw.append((src, dest, QColor("lightgray"), False))
        self.current_representation_text = ""
        self.router_probs_text = ""
        self.current_path_text = "Path: "
        self.update()

    def highlight_node(self, name, active=True):
        if name not in self.node_positions: return  # Node might not exist if num_experts changed
        if active:
            self.active_elements[name] = QColor("yellow")
        else:
            if name in self.active_elements:
                del self.active_elements[name]
        self.update()

    def highlight_connection(self, from_name, to_name, active=True):
        if from_name not in self.node_positions or to_name not in self.node_positions: return

        updated = False
        for i, (s, d, c, was_active) in enumerate(self.connections_to_draw):
            if (s == from_name and d == to_name):
                self.connections_to_draw[i] = (s, d, QColor("orange") if active else QColor("lightgray"), active)
                updated = True
                break
        if not updated:  # Add if missing (e.g. after expert count change)
            self.connections_to_draw.append(
                (from_name, to_name, QColor("orange") if active else QColor("lightgray"), active))

        if active:  # Deactivate other connections from 'from_name'
            for i, (s, d, c, was_active_flag) in enumerate(self.connections_to_draw):
                if s == from_name and d != to_name:
                    self.connections_to_draw[i] = (s, d, QColor("lightgray"), False)
        self.update()

    def set_representation_text(self, text):
        self.current_representation_text = f"Repr (mean): {text}"
        self.update()

    def set_router_probs_text(self, probs_list):
        if not probs_list: self.router_probs_text = ""; return

        expert_labels = [f"E{i}" for i in range(self.num_experts_to_draw)]
        labels_str = ",".join(expert_labels) + ",Term"

        formatted_probs = [f"{p:.2f}" for p in probs_list]
        self.router_probs_text = f"Router Probs ({labels_str}): [{', '.join(formatted_probs)}]"
        self.update()

    def set_current_path_text(self, path_list):
        self.current_path_text = "Path: " + " -> ".join(map(str, path_list))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw connections
        for from_name, to_name, color, is_active_conn in self.connections_to_draw:
            if from_name not in self.node_positions or to_name not in self.node_positions: continue
            p1 = self.node_positions[from_name]
            p2 = self.node_positions[to_name]
            pen = QPen(color, 2.5 if is_active_conn else 1.5, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(p1, p2)

            # Arrowhead
            line_vec = p2 - p1;
            length = pow(line_vec.x() ** 2 + line_vec.y() ** 2, 0.5)
            if length == 0: continue
            norm_line_vec = line_vec / length
            angle_rad = math.atan2(norm_line_vec.y(), norm_line_vec.x())
            arrow_size = 8.0
            arrow_p1 = p2 - QPointF(arrow_size * math.cos(angle_rad - math.pi / 7),
                                    arrow_size * math.sin(angle_rad - math.pi / 7))
            arrow_p2 = p2 - QPointF(arrow_size * math.cos(angle_rad + math.pi / 7),
                                    arrow_size * math.sin(angle_rad + math.pi / 7))
            arrow_head = QPolygonF([p2, arrow_p1, arrow_p2])
            painter.setBrush(QBrush(color));
            painter.drawPolygon(arrow_head)

        # Draw nodes
        node_font = QFont("Arial", 7)
        painter.setFont(node_font)
        for name, pos in self.node_positions.items():
            rect = QRectF(pos.x() - self.node_size.width() / 2, pos.y() - self.node_size.height() / 2,
                          self.node_size.width(), self.node_size.height())
            color = self.active_elements.get(name, QColor("lightblue"))
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.black))
            painter.drawRoundedRect(rect, 5, 5)

            display_name = name.replace("_", " ")
            if "EMBED" in name: display_name = "EMBED & POS"
            if "LM_HEAD" in name: display_name = "LM HEAD"
            painter.drawText(rect, Qt.AlignCenter, display_name)

        # Draw info text
        info_font = QFont("Arial", 8);
        painter.setFont(info_font);
        painter.setPen(Qt.black)
        text_y_start = 12
        painter.drawText(QPointF(10, text_y_start), self.current_representation_text)
        painter.drawText(QPointF(10, text_y_start + 18), self.router_probs_text)
        painter.drawText(QPointF(10, text_y_start + 36), self.current_path_text)


class AnimationSignals(QWidget):  # QWidget for QThread parent compatibility
    signal_input_text = pyqtSignal(str)
    signal_embedded = pyqtSignal(str)
    signal_to_router = pyqtSignal(str, list)
    signal_router_output = pyqtSignal(list, int,
                                      bool)  # probs, selected_idx (0..N-1 for expert, N for term), is_terminate
    signal_to_expert = pyqtSignal(int, str)  # expert_idx, repr_preview
    signal_expert_output = pyqtSignal(int, str)  # expert_idx, repr_preview
    signal_to_output_layer = pyqtSignal(str)  # repr_preview
    signal_final_prediction_lm = pyqtSignal(str, list)  # predicted_token_str, path_taken

    log_message = pyqtSignal(str)
    epoch_completed = pyqtSignal(int, float, float, dict)  # epoch_idx, loss, perplexity, pathway_stats
    inference_completed = pyqtSignal(str, str, str, list)  # input_text, predicted_token, true_token, path_taken

    def __init__(self, vocab_ref, anim_delay_getter_fn, parent=None):
        super().__init__(parent)
        self.vocab = vocab_ref  # Keep a reference to the vocab for denumericalization
        self.get_animation_delay_value = anim_delay_getter_fn


class Worker(QThread):
    def __init__(self, model, data_loader, optimizer, criterion, mode="train", sample_to_infer=None,
                 hps=None, signals_ref=None):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.mode = mode
        self.sample_to_infer = sample_to_infer  # (input_ids_seq, target_id, original_text_snippet)
        self.hps = hps  # Current hyperparameters dictionary

        self.signals = signals_ref  # Passed from MainWindow
        self.model.animation_signals = self.signals  # Connect model to signals for emitting
        self.pathway_stats = collections.Counter()

    def run(self):
        self.model.is_animating = True  # Enable animation signals in model
        if self.mode == "train":
            self._train_epoch()
        elif self.mode == "infer":
            self._infer_sample()
        self.model.is_animating = False

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        total_samples = 0
        has_animated_this_epoch = False

        for batch_idx, (input_ids_seq, target_ids, original_texts) in enumerate(self.data_loader):
            current_sample_text_for_anim = None
            if not has_animated_this_epoch:
                self.model.is_animating = True
                current_sample_text_for_anim = original_texts[0]  # Animate first sample of first batch
            else:
                self.model.is_animating = False

            self.optimizer.zero_grad()
            logits, router_entropy, paths = self.model(input_ids_seq, current_sample_text_for_anim)

            task_loss = self.criterion(logits, target_ids)
            loss = task_loss + self.hps[
                'router_entropy_coef'] * router_entropy  # Note: spec says minimize task loss, so entropy is positive here if it's a cost.
            # If entropy is a bonus, it should be - router_entropy.
            # Let's assume it's a penalty on low entropy (peaked distributions)
            # So we want to MAXIMIZE entropy. Thus, -coeff * entropy in loss.
            # Or, if coeff is positive, add coeff * (-entropy).
            # The spec says "Entropy penalty on routing probabilities" -> add -H to loss or H to reward.
            # So, + coeff * entropy if entropy is H = -p log p.
            # If router_entropy is already -sum(p log p), then add it.
            # My router_entropy is calculated as -sum(p log p), so it's positive.
            # Adding it with a positive coefficient will try to maximize it.
            # Let's re-read: "Entropy penalty on routing probabilities to encourage balanced expert usage"
            # This means we want to MINIMIZE (-Entropy), or MINIMIZE (negative_entropy_term).
            # So, if H = -sum(p log p), we want to maximize H. Loss = TaskLoss - C*H.
            # My `total_router_entropy` is sum of H for each step. So it's correct.

            loss.backward()
            self.optimizer.step()

            total_loss += task_loss.item() * input_ids_seq.size(0)  # Use task_loss for perplexity
            total_samples += input_ids_seq.size(0)

            for path in paths: self.pathway_stats[tuple(path)] += 1

            if not has_animated_this_epoch:
                pred_token_idx = logits[0].argmax().item()
                pred_token_str = self.signals.vocab.denumericalize([pred_token_idx])[0]
                true_token_str = self.signals.vocab.denumericalize([target_ids[0].item()])[0]
                self.signals.log_message.emit(
                    f"Anim sample loss: {loss.item():.4f}. True: '{true_token_str}', Pred: '{pred_token_str}'")
                has_animated_this_epoch = True

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss != float('inf') else float('inf')
        self.signals.epoch_completed.emit(0, avg_loss, perplexity, dict(self.pathway_stats))

    def _infer_sample(self):
        self.model.eval()
        self.model.is_animating = True

        input_ids, target_id, original_text_snippet = self.sample_to_infer
        input_ids_batch = input_ids.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            logits, _, paths = self.model(input_ids_batch, current_sample_text_for_anim=original_text_snippet)

        pred_token_idx = logits.argmax(dim=1).item()
        path_taken = paths[0]
        self.pathway_stats[tuple(path_taken)] += 1

        pred_token_str = self.signals.vocab.denumericalize([pred_token_idx])[0]
        true_token_str = self.signals.vocab.denumericalize([target_id.item()])[0]

        self.signals.inference_completed.emit(original_text_snippet, pred_token_str, true_token_str, path_taken)
        self.signals.log_message.emit(f"Inferred. Path: {path_taken}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph of Experts for Language Modeling Demo")
        self.setGeometry(50, 50, 1200, 850)  # Larger window

        self.hps = self._get_default_hps()  # Initialize HPs
        self.vocab = Vocabulary(max_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQ)
        self.dataset = None  # Will be created on init_model_and_data
        self.data_loader = None

        self.model = None
        self.optimizer = None
        self.criterion = None

        self.current_epoch = 0
        self.pathway_frequencies = collections.Counter()

        self._init_ui()
        self._connect_signals_basic()

        # Create AnimationSignals instance here and pass its getter for anim_delay
        self.animation_signals_obj = AnimationSignals(self.vocab, lambda: self.anim_delay_spin.value())

        self.init_model_and_data()  # Initial model setup

    def _get_default_hps(self):
        return {
            'embed_dim': EMBED_DIM_DEFAULT,
            'num_experts': NUM_EXPERTS_DEFAULT,
            'expert_nhead': EXPERT_NHEAD_DEFAULT,
            'expert_dim_feedforward': EXPERT_DIM_FEEDFORWARD_DEFAULT,
            'router_hidden_dim': ROUTER_HIDDEN_DIM_DEFAULT,
            'max_path_len': MAX_PATH_LEN_DEFAULT,
            'max_visits_per_expert': MAX_VISITS_PER_EXPERT_IN_PATH_DEFAULT,
            'max_lm_seq_len': MAX_LM_SEQ_LEN,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'router_entropy_coef': ROUTER_ENTROPY_COEF,
            'weight_decay': WEIGHT_DECAY,
            'animation_delay': ANIMATION_DELAY_DEFAULT
        }

    def _get_current_hps_from_ui(self):
        return {
            'embed_dim': self.embed_dim_spin.value(),
            'num_experts': self.num_experts_spin.value(),
            'expert_nhead': self.expert_nhead_spin.value(),
            'expert_dim_feedforward': self.expert_ff_spin.value(),
            'router_hidden_dim': self.router_hidden_spin.value(),
            'max_path_len': self.max_path_len_spin.value(),
            'max_visits_per_expert': self.max_visits_spin.value(),
            'max_lm_seq_len': MAX_LM_SEQ_LEN,  # Fixed for this demo based on data
            'learning_rate': self.lr_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'router_entropy_coef': self.entropy_coef_spin.value(),
            'weight_decay': WEIGHT_DECAY,  # Fixed for simplicity
            'animation_delay': self.anim_delay_spin.value()
        }

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Diagram on the left
        self.diagram_widget = DiagramWidget(num_experts=self.hps['num_experts'])
        layout.addWidget(self.diagram_widget, stretch=2)

        # Controls and Info on the right
        controls_info_layout = QVBoxLayout()

        # --- Hyperparameters Group ---
        hps_group = QGroupBox("Hyperparameters")
        hps_form = QFormLayout()
        # Model HPs
        self.num_experts_spin = QSpinBox();
        self.num_experts_spin.setRange(2, 5);
        self.num_experts_spin.setValue(self.hps['num_experts'])
        self.embed_dim_spin = QSpinBox();
        self.embed_dim_spin.setRange(16, 128);
        self.embed_dim_spin.setStepType(QSpinBox.AdaptiveDecimalStepType);
        self.embed_dim_spin.setValue(self.hps['embed_dim'])
        self.expert_nhead_spin = QSpinBox();
        self.expert_nhead_spin.setRange(1, 8);
        self.expert_nhead_spin.setValue(self.hps['expert_nhead'])
        self.expert_ff_spin = QSpinBox();
        self.expert_ff_spin.setRange(32, 256);
        self.expert_ff_spin.setStepType(QSpinBox.AdaptiveDecimalStepType);
        self.expert_ff_spin.setValue(self.hps['expert_dim_feedforward'])
        self.router_hidden_spin = QSpinBox();
        self.router_hidden_spin.setRange(16, 128);
        self.router_hidden_spin.setStepType(QSpinBox.AdaptiveDecimalStepType);
        self.router_hidden_spin.setValue(self.hps['router_hidden_dim'])
        self.max_path_len_spin = QSpinBox();
        self.max_path_len_spin.setRange(1, 10);
        self.max_path_len_spin.setValue(self.hps['max_path_len'])
        self.max_visits_spin = QSpinBox();
        self.max_visits_spin.setRange(1, 5);
        self.max_visits_spin.setValue(self.hps['max_visits_per_expert'])
        # Training HPs
        self.lr_spin = QDoubleSpinBox();
        self.lr_spin.setRange(1e-5, 1e-2);
        self.lr_spin.setSingleStep(1e-4);
        self.lr_spin.setDecimals(5);
        self.lr_spin.setValue(self.hps['learning_rate'])
        self.batch_size_spin = QSpinBox();
        self.batch_size_spin.setRange(1, 32);
        self.batch_size_spin.setValue(self.hps['batch_size'])
        self.entropy_coef_spin = QDoubleSpinBox();
        self.entropy_coef_spin.setRange(0.0, 0.1);
        self.entropy_coef_spin.setSingleStep(0.005);
        self.entropy_coef_spin.setDecimals(4);
        self.entropy_coef_spin.setValue(self.hps['router_entropy_coef'])
        self.anim_delay_spin = QDoubleSpinBox();
        self.anim_delay_spin.setRange(0.0, 3.0);
        self.anim_delay_spin.setSingleStep(0.1);
        self.anim_delay_spin.setValue(self.hps['animation_delay'])

        hps_form.addRow("Num Experts:", self.num_experts_spin)
        hps_form.addRow("Embedding Dim:", self.embed_dim_spin)
        hps_form.addRow("Expert N Heads:", self.expert_nhead_spin)
        hps_form.addRow("Expert FF Dim:", self.expert_ff_spin)
        hps_form.addRow("Router Hidden Dim:", self.router_hidden_spin)
        hps_form.addRow("Max Path Len:", self.max_path_len_spin)
        hps_form.addRow("Max Visits/Expert:", self.max_visits_spin)
        hps_form.addRow("Learning Rate:", self.lr_spin)
        hps_form.addRow("Batch Size:", self.batch_size_spin)
        hps_form.addRow("Router Entropy Coeff:", self.entropy_coef_spin)
        hps_form.addRow("Animation Delay (s):", self.anim_delay_spin)
        hps_group.setLayout(hps_form)
        controls_info_layout.addWidget(hps_group)

        # --- Controls Group ---
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        self.reinit_button = QPushButton("Apply Settings & Reinitialize Model")
        self.train_button = QPushButton("Train 1 Epoch")
        self.infer_button = QPushButton("Infer One Sample")
        self.reset_anim_button = QPushButton("Reset Animation Highlights")
        actions_layout.addWidget(self.reinit_button)
        actions_layout.addWidget(self.train_button)
        actions_layout.addWidget(self.infer_button)
        actions_layout.addWidget(self.reset_anim_button)
        actions_group.setLayout(actions_layout)
        controls_info_layout.addWidget(actions_group)

        # --- Info Text Edits ---
        self.log_text_edit = QTextEdit();
        self.log_text_edit.setReadOnly(True);
        self.log_text_edit.setFixedHeight(120)
        controls_info_layout.addWidget(QLabel("Log:"));
        controls_info_layout.addWidget(self.log_text_edit)
        self.stats_text_edit = QTextEdit();
        self.stats_text_edit.setReadOnly(True);
        self.stats_text_edit.setFixedHeight(150)
        controls_info_layout.addWidget(QLabel("Epoch Stats & Pathway Frequencies (for Compilation):"));
        controls_info_layout.addWidget(self.stats_text_edit)
        self.current_sample_text_edit = QTextEdit();
        self.current_sample_text_edit.setReadOnly(True);
        self.current_sample_text_edit.setFixedHeight(80)
        controls_info_layout.addWidget(QLabel("Current Sample Info:"));
        controls_info_layout.addWidget(self.current_sample_text_edit)

        controls_info_layout.addStretch(1)
        layout.addLayout(controls_info_layout, stretch=1)

    def _connect_signals_basic(self):
        self.reinit_button.clicked.connect(self.init_model_and_data)
        self.train_button.clicked.connect(self.start_training)
        self.infer_button.clicked.connect(self.start_inference)
        self.reset_anim_button.clicked.connect(self.reset_animation_display)
        # No need to connect spinboxes directly to update_model_hps if we use a reinit button

    def _connect_worker_signals(self, worker_signals_obj):
        # worker_signals_obj is self.animation_signals_obj
        worker_signals_obj.signal_input_text.connect(self.on_input_text)
        worker_signals_obj.signal_embedded.connect(self.on_embedded)
        worker_signals_obj.signal_to_router.connect(self.on_to_router)
        worker_signals_obj.signal_router_output.connect(self.on_router_output)
        worker_signals_obj.signal_to_expert.connect(self.on_to_expert)
        worker_signals_obj.signal_expert_output.connect(self.on_expert_output)
        worker_signals_obj.signal_to_output_layer.connect(self.on_to_output_layer)
        worker_signals_obj.signal_final_prediction_lm.connect(self.on_final_prediction_lm_anim)  # Changed
        worker_signals_obj.log_message.connect(self.log_message)
        worker_signals_obj.epoch_completed.connect(self.on_epoch_completed)
        worker_signals_obj.inference_completed.connect(self.on_inference_completed_lm)  # Changed

    def init_model_and_data(self):
        self.log_message("Reinitializing model and data with current settings...")
        self.hps = self._get_current_hps_from_ui()  # Update HPs from UI

        # 1. (Re)build Vocabulary and Dataset
        self.vocab = Vocabulary(max_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQ)
        self.dataset = LanguageModelDataset(WIKITEXT_SAMPLE, self.vocab, self.hps['max_lm_seq_len'])
        if not self.dataset.sequences:
            self.log_message("ERROR: No sequences generated from dataset. Check text and seq_len.")
            self.train_button.setEnabled(False);
            self.infer_button.setEnabled(False)
            return

        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.hps['batch_size'], shuffle=True)
        self.animation_signals_obj.vocab = self.vocab  # Update vocab ref in signals

        # 2. (Re)create Model
        self.model = GoEModel(
            vocab_size=len(self.vocab),
            embed_dim=self.hps['embed_dim'],
            num_experts=self.hps['num_experts'],
            expert_nhead=self.hps['expert_nhead'],
            expert_dim_feedforward=self.hps['expert_dim_feedforward'],
            router_hidden_dim=self.hps['router_hidden_dim'],
            max_path_len=self.hps['max_path_len'],
            max_visits_per_expert=self.hps['max_visits_per_expert'],
            max_lm_seq_len=self.hps['max_lm_seq_len']
        )
        # 3. (Re)create Optimizer and Criterion
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.hps['learning_rate'],
                                     weight_decay=self.hps['weight_decay'])
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.word_to_idx["<pad>"])

        # 4. Reset training state
        self.current_epoch = 0
        self.pathway_frequencies.clear()
        self.stats_text_edit.clear()
        self.log_text_edit.clear()
        self.current_sample_text_edit.clear()

        # 5. Update Diagram
        self.diagram_widget.update_num_experts(self.hps['num_experts'])

        self.log_message(f"Model reinitialized. Vocab size: {len(self.vocab)}. Num experts: {self.hps['num_experts']}.")
        self.log_message(f"Dataset has {len(self.dataset)} sequences.")
        self.train_button.setEnabled(True);
        self.infer_button.setEnabled(True)

    def reset_animation_display(self):
        self.diagram_widget.reset_highlights()
        self.current_sample_text_edit.clear()
        self.log_message("Animation display reset.")

    def start_training(self):
        if not self.model or not self.data_loader:
            self.log_message("Model or data not initialized. Please Reinitialize.")
            return
        self.reset_animation_display()
        self.log_message(f"Starting training epoch {self.current_epoch + 1}...")
        self.train_button.setEnabled(False);
        self.infer_button.setEnabled(False);
        self.reinit_button.setEnabled(False)

        self.worker = Worker(self.model, self.data_loader, self.optimizer, self.criterion,
                             mode="train", hps=self.hps, signals_ref=self.animation_signals_obj)
        self._connect_worker_signals(self.animation_signals_obj)  # Ensure connections are fresh
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def start_inference(self):
        if not self.model or not self.dataset or not self.dataset.sequences:
            self.log_message("Model or data not initialized / no sequences. Please Reinitialize.")
            return
        self.reset_animation_display()
        sample_idx = random.randint(0, len(self.dataset) - 1)
        input_ids_seq_list, target_id_tensor, original_text_snippet = self.dataset[sample_idx]

        # Convert list to tensor for worker
        input_ids_seq_tensor = torch.tensor(input_ids_seq_list, dtype=torch.long)

        self.log_message(f"Starting inference for: {original_text_snippet}")
        self.current_sample_text_edit.setText(f"Input: {original_text_snippet}")

        self.train_button.setEnabled(False);
        self.infer_button.setEnabled(False);
        self.reinit_button.setEnabled(False)

        self.worker = Worker(self.model, None, None, None, mode="infer",
                             sample_to_infer=(input_ids_seq_tensor, target_id_tensor, original_text_snippet),
                             hps=self.hps, signals_ref=self.animation_signals_obj)
        self._connect_worker_signals(self.animation_signals_obj)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self):
        self.log_message("Worker finished.")
        self.train_button.setEnabled(True);
        self.infer_button.setEnabled(True);
        self.reinit_button.setEnabled(True)
        if hasattr(self.worker, 'pathway_stats') and self.worker.pathway_stats:
            self.pathway_frequencies.update(self.worker.pathway_stats)
            self.display_pathway_stats()  # Update stats display

    def log_message(self, msg):
        self.log_text_edit.append(msg)

    def display_pathway_stats(self):
        path_stats_str = f"--- Pathway Frequencies (Total {sum(self.pathway_frequencies.values())}) ---\n"
        sorted_paths = sorted(self.pathway_frequencies.items(), key=lambda item: item[1], reverse=True)
        for path, count in sorted_paths[:10]:  # Show top 10
            path_str = " -> ".join(map(str, path))
            path_stats_str += f"{path_str}: {count}\n"

        total_paths_logged = sum(self.pathway_frequencies.values())
        if total_paths_logged > 0:
            path_stats_str += "\n--- Potential Compilation Candidates (>{:.0f}% usage) ---\n".format(
                0.1 * 100)  # 10% threshold
            for path, count in sorted_paths:
                if count / total_paths_logged > 0.10:
                    path_str = " -> ".join(map(str, path))
                    path_stats_str += f"MERGE CANDIDATE: {path_str} ({count / total_paths_logged:.2%})\n"

        # Preserve existing epoch summaries
        current_stats_text = self.stats_text_edit.toPlainText()
        epoch_summaries = ""
        if "--- Pathway Frequencies ---" in current_stats_text:
            epoch_summaries = current_stats_text.split("--- Pathway Frequencies ---")[0]

        self.stats_text_edit.setText(epoch_summaries.strip() + "\n" + path_stats_str)

    # --- Animation Slot Handlers ---
    def on_input_text(self, text):
        self.diagram_widget.reset_highlights()
        self.diagram_widget.highlight_node("INPUT")
        self.current_sample_text_edit.setText(f"Input: {text}")

    def on_embedded(self, repr_preview):  # repr_preview is mean of sequence
        self.diagram_widget.highlight_node("INPUT", False)
        self.diagram_widget.highlight_connection("INPUT", "EMBED")
        self.diagram_widget.highlight_node("EMBED")
        self.diagram_widget.set_representation_text(repr_preview)

    def on_to_router(self, repr_preview, current_path):  # repr_preview is mean of sequence
        self.diagram_widget.highlight_node("EMBED", False)
        if current_path:
            last_elem = current_path[-1]
            if isinstance(last_elem, int):  # An expert index
                self.diagram_widget.highlight_node(f"EXPERT_{last_elem}", False)
                self.diagram_widget.highlight_connection(f"EXPERT_{last_elem}", "ROUTER")
        else:  # Came from EMBED
            self.diagram_widget.highlight_connection("EMBED", "ROUTER")
        self.diagram_widget.highlight_node("ROUTER")
        self.diagram_widget.set_representation_text(repr_preview)
        self.diagram_widget.set_current_path_text(current_path)

    def on_router_output(self, probs, selected_index, is_terminate):
        self.diagram_widget.set_router_probs_text(probs)
        self.diagram_widget.highlight_node("ROUTER", False)
        if is_terminate:
            self.diagram_widget.highlight_connection("ROUTER", "LM_HEAD")
        else:  # Selected an expert
            self.diagram_widget.highlight_connection("ROUTER", f"EXPERT_{selected_index}")

    def on_to_expert(self, expert_idx, repr_preview):  # repr_preview is mean of sequence
        self.diagram_widget.highlight_node(f"EXPERT_{expert_idx}")
        self.diagram_widget.set_representation_text(repr_preview)

    def on_expert_output(self, expert_idx, repr_preview):  # repr_preview is mean of sequence
        self.diagram_widget.set_representation_text(repr_preview)  # Expert stays highlighted

    def on_to_output_layer(self, repr_preview):  # repr_preview is mean of sequence
        self.diagram_widget.highlight_node("ROUTER", False)  # Assume came from router
        # Could also come from max_path_len termination after an expert, diagram logic might need refinement for that edge case visual
        self.diagram_widget.highlight_node("LM_HEAD")
        self.diagram_widget.set_representation_text(repr_preview)

    def on_final_prediction_lm_anim(self, predicted_token_str, path_taken):
        self.diagram_widget.highlight_node("LM_HEAD", True)  # Keep LM_HEAD highlighted
        path_str = " -> ".join(map(str, path_taken))
        self.diagram_widget.set_current_path_text(path_taken)
        current_text = self.current_sample_text_edit.toPlainText()
        self.current_sample_text_edit.setText(
            f"{current_text}\nPredicted Token: '{predicted_token_str}'\nPath: {path_str}")

    def on_epoch_completed(self, epoch_num_ignored, loss, perplexity, pathway_freq_update):
        self.current_epoch += 1
        epoch_msg = f"Epoch {self.current_epoch}: Loss={loss:.4f}, Perplexity={perplexity:.2f}"
        self.log_message(epoch_msg)

        # Prepend epoch summary to stats text
        current_stats = self.stats_text_edit.toPlainText()
        pathway_section = ""
        if "--- Pathway Frequencies ---" in current_stats:
            pathway_section = "--- Pathway Frequencies ---" + current_stats.split("--- Pathway Frequencies ---", 1)[1]
        self.stats_text_edit.setText(epoch_msg + "\n" + pathway_section)

        self.pathway_frequencies.update(pathway_freq_update)  # Update internal counter
        self.display_pathway_stats()  # Refresh the pathway part of the stats

    def on_inference_completed_lm(self, input_text_snippet, pred_token, true_token, path_taken):
        self.log_message(
            f"Inference for: {input_text_snippet}. Pred: '{pred_token}', True: '{true_token}'. Path: {path_taken}")
        path_str = " -> ".join(map(str, path_taken))
        self.current_sample_text_edit.setText(
            f"Input: {input_text_snippet}\nTrue Token: '{true_token}'\nPredicted Token: '{pred_token}'\nPath: {path_str}")
        self.pathway_frequencies.update({tuple(path_taken): 1})
        self.display_pathway_stats()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        import torch
    except ImportError:
        print("CRITICAL: PyTorch not found. Please install PyTorch.")
        sys.exit(1)

    if torch.cuda.is_available():
        print("CUDA available. Demo runs on CPU for simplicity.")
    else:
        print("CUDA not available. Demo runs on CPU.")
    torch.set_default_device('cpu')

    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())