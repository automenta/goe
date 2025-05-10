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
from torch.optim.lr_scheduler import StepLR

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QSizePolicy, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QTabWidget, QProgressBar
)
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QTimer

try:
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("WARNING: pyqtgraph not found. Live plotting will be disabled. Install with 'pip install pyqtgraph'")

# --- Constants and Configuration ---
WIKITEXT_SAMPLE = """
= Valkyria Chronicles III = 
Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria Chronicles series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the original Valkyria Chronicles and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who are assigned to sensitive operations and missions deemed too dangerous for the regular army . 
The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's theme is " despair " , with the plot exploring the Gallian military 's persecution of those deemed undesirable and the Nameless 's struggle for redemption . 
Upon release , the game received positive reviews and was praised by both Japanese and western critics . After release , it received downloadable content , as well as manga and drama CD adaptations . Despite the positive reception , Valkyria Chronicles III was not localized for Western territories . The decision was attributed to the poor sales of Valkyria Chronicles II , a troubled history with the PlayStation Portable , and the popularity of mobile gaming in the West . Various English fan translation projects were initiated , but were left unfinished due to the complexity of the game 's script and coding . A full English fan translation was eventually released in 2014 . 

= Gameplay =
Like its predecessors , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and participate in missions against enemy forces . The game is presented in a series of chapters , with story cutscenes and battles delivered in a book @-@ like format . Players navigate through menus to prepare their units , watch story segments , and progress through the game . Missions generally have a single objective , such as capturing an enemy base camp or defeating a specific enemy unit . Some missions have defeat conditions , such as the death of a key unit or the destruction of a key allied vehicle . 
The gameplay is divided between two modes : Command Mode and Action Mode . In Command Mode , players view an overhead map of the battlefield , showing allied and enemy units . Each turn , players are given a set number of Command Points ( CP ) , which are used to activate individual units or issue special commands . Selecting a unit initiates Action Mode , where players directly control the unit in a third @-@ person perspective . During Action Mode , units can move a limited distance and perform one action , such as attacking an enemy or healing an ally . After the action is performed , the unit remains on the field unless retreated by the player . Taking control of a unit costs 1 CP . If a unit is taken down by enemy fire during Action Mode , it is considered critically injured and must be evacuated by an allied unit within a set number of turns , or it will be permanently lost . 
Units are divided into several classes , each with unique abilities and weapons . For example , Scouts are highly mobile units with good reconnaissance capabilities , while Shocktroopers are slower but more heavily armed . Engineers can repair tanks and resupply ammunition , while Lancers are anti @-@ tank specialists . Each unit has a unique set of Potentials , which are special abilities that can be triggered under certain conditions , such as being near an ally or having low health . The game also features a " Command " system , allowing players to issue orders to multiple units simultaneously . These orders can provide various effects , such as increasing attack power or boosting defense . 
The story of Valkyria Chronicles III focuses on Squad 422 , also known as the " Nameless " . This penal military unit is composed of deserters , criminals , and other individuals deemed undesirable by the Gallian military . Led by Kurt Irving , a former Gallian Army officer falsely accused of treason , the Nameless are assigned to the most dangerous and morally ambiguous missions of the Second Europan War . The unit operates outside the regular military structure and is often treated with disdain and suspicion by other Gallian soldiers . Despite their circumstances , the members of the Nameless strive for redemption and fight to protect their country , even as they face discrimination and internal conflicts . The narrative explores themes of sacrifice , prejudice , and the true meaning of heroism in the face of overwhelming despair .
"""
MAX_VOCAB_SIZE = 2500  # Slightly increased due to more text
MIN_FREQ = 1
MAX_LM_SEQ_LEN = 10
TRAIN_SPLIT_RATIO = 0.8

# Model HPs (Defaults) - Adjusted
EMBED_DIM_DEFAULT = 48
NUM_EXPERTS_DEFAULT = 3
EXPERT_NHEAD_DEFAULT = 3  # Divisor of EMBED_DIM
EXPERT_DIM_FEEDFORWARD_DEFAULT = 96
ROUTER_HIDDEN_DIM_DEFAULT = 48
MAX_PATH_LEN_DEFAULT = 4  # Increased
MAX_VISITS_PER_EXPERT_IN_PATH_DEFAULT = 1

# Training HPs
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
ROUTER_ENTROPY_COEF = 0.005  # Adjusted
WEIGHT_DECAY = 1e-4
GRAD_CLIP_MAX_NORM = 1.0  # Added
SCHEDULER_STEP_SIZE_MULTIPLIER = 10  # Scheduler steps every X eval cycles
SCHEDULER_GAMMA = 0.90  # LR decay factor

TRAIN_STEPS_PER_EVAL = 50
SMOOTHING_WINDOW_LOSS = 20

# Animation
ANIMATION_DELAY_DEFAULT = 0.5

# Plotting
PLOT_UPDATE_INTERVAL_MS = 1000
MAX_PLOT_POINTS = 100


# --- Dataset and Preprocessing (largely same as before) ---
class Vocabulary:
    def __init__(self, max_size=None, min_freq=1):
        self.word_to_idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.word_counts = collections.Counter()
        self.max_size = max_size
        self.min_freq = min_freq
        self._finalized = False

    def add_word(self, word):
        if self._finalized and len(self.word_to_idx) >= (self.max_size or float('inf')): return
        self.word_counts[word] += 1

    def build_vocab(self):
        if self._finalized and len(self.word_to_idx) > 4: return

        sorted_words = self.word_counts.most_common()
        self.word_to_idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}  # Reset for fresh build
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        for word, count in sorted_words:
            if self.max_size and len(self.word_to_idx) >= self.max_size: break
            if count < self.min_freq and word not in self.word_to_idx: continue
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        self._finalized = True
        print(f"Vocabulary built with {len(self.word_to_idx)} words.")

    def __len__(self):
        return len(self.word_to_idx)

    def numericalize(self, text_tokens: List[str]) -> List[int]:
        if not self._finalized: self.build_vocab()
        return [self.word_to_idx.get(token, self.word_to_idx["<unk>"]) for token in text_tokens]

    def denumericalize(self, indices: List[int]) -> List[str]:
        return [self.idx_to_word.get(idx, "<unk>") for idx in indices]


def tokenize_text(text: str) -> List[str]: return text.lower().split()


def create_lm_sequences(token_ids: List[int], seq_len: int) -> List[Tuple[List[int], int]]:
    sequences = []
    if not token_ids or len(token_ids) <= seq_len: return sequences
    for i in range(len(token_ids) - seq_len):
        sequences.append((token_ids[i: i + seq_len], token_ids[i + seq_len]))
    return sequences


class LanguageModelDataset(torch.utils.data.Dataset):
    def __init__(self, text_data_tokens: List[str], vocab: Vocabulary, seq_len: int, dataset_name="train"):
        self.vocab = vocab
        self.seq_len = seq_len

        if dataset_name == "train" and not self.vocab._finalized:
            for token in text_data_tokens: self.vocab.add_word(token)
            self.vocab.build_vocab()

        numericalized_tokens = self.vocab.numericalize(text_data_tokens)
        self.sequences = create_lm_sequences(numericalized_tokens, self.seq_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq_ids, target_word_id = self.sequences[idx]
        original_text_snippet = " ".join(self.vocab.denumericalize(input_seq_ids)) + \
                                " ...predict... " + self.vocab.denumericalize([target_word_id])[0]
        return (torch.tensor(input_seq_ids, dtype=torch.long),
                torch.tensor(target_word_id, dtype=torch.long),
                original_text_snippet)


# --- PyTorch Model Components (largely same as before) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Expert(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout=0.1, expert_id=0):
        super().__init__()
        self.expert_id = expert_id
        # Ensure nhead is a divisor of embed_dim for MultiheadAttention
        if embed_dim % nhead != 0:
            # Find the largest nhead <= requested that is a divisor
            valid_nhead = nhead
            while embed_dim % valid_nhead != 0 and valid_nhead > 1:
                valid_nhead -= 1
            if embed_dim % valid_nhead != 0:  # Fallback to 1 if no divisor found
                print(
                    f"Warning: Expert {expert_id}: embed_dim {embed_dim} not divisible by nhead {nhead}. Using nhead=1.")
                valid_nhead = 1
            else:
                print(
                    f"Warning: Expert {expert_id}: embed_dim {embed_dim} not divisible by nhead {nhead}. Adjusted nhead to {valid_nhead}.")
            nhead = valid_nhead

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.specialization_tag = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.01)

    def forward(self, x_seq):
        return self.transformer_layer(x_seq) + self.specialization_tag


class RoutingController(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_experts + 1)

    def forward(self, x_summary, current_visit_counts_for_batch_item, max_visits_per_expert):
        x = F.relu(self.fc1(x_summary))
        logits = self.fc2(x)
        is_batched = x_summary.dim() == 2
        for i in range(logits.size(0) if is_batched else 1):
            visits = current_visit_counts_for_batch_item[i] if is_batched else current_visit_counts_for_batch_item
            current_logits = logits[i] if is_batched else logits
            for expert_idx in range(self.num_experts):
                if visits[expert_idx] >= max_visits_per_expert:
                    current_logits[expert_idx] = -float('inf')
        return logits


class GoEModel(nn.Module):  # Logic largely same, ensure HPs are used
    def __init__(self, vocab_size, embed_dim, num_experts, expert_nhead, expert_dim_feedforward,
                 router_hidden_dim, max_path_len, max_visits_per_expert, max_lm_seq_len):
        super().__init__()
        self.vocab_size = vocab_size;
        self.embed_dim = embed_dim;
        self.num_experts = num_experts
        self.max_path_len = max_path_len;
        self.max_visits_per_expert_in_path = max_visits_per_expert

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_lm_seq_len + 5)
        self.experts = nn.ModuleList(
            [Expert(embed_dim, expert_nhead, expert_dim_feedforward, expert_id=i) for i in range(num_experts)])
        self.router = RoutingController(embed_dim, router_hidden_dim, num_experts)
        self.output_lm_head = nn.Linear(embed_dim, vocab_size)

        self.animation_signals_obj = None
        self.is_animating_sample = False
        self.current_hps = {}

    def _emit_signal(self, signal_type: str, *args):
        if self.is_animating_sample and self.animation_signals_obj:
            delay = self.current_hps.get('animation_delay', 0.1)
            getattr(self.animation_signals_obj, signal_type).emit(*args)
            if delay > 0: QThread.msleep(int(delay * 1000))

    def forward(self, input_ids_seq, current_sample_text_for_anim=None):
        batch_size = input_ids_seq.size(0)
        if self.is_animating_sample and current_sample_text_for_anim:
            self._emit_signal("signal_input_text", current_sample_text_for_anim)

        embedded_input = self.embedding(input_ids_seq) * math.sqrt(self.embed_dim)
        current_representation_seq = self.pos_encoder(embedded_input)

        if self.is_animating_sample:
            self._emit_signal("signal_embedded", repr_tensor(current_representation_seq[0].mean(dim=0)))

        total_router_entropy = torch.tensor(0.0, device=input_ids_seq.device)
        paths_taken_indices = [[] for _ in range(batch_size)]
        current_visit_counts_batch = torch.zeros((batch_size, self.num_experts), dtype=torch.long,
                                                 device=input_ids_seq.device)
        final_representations_seq_batch = torch.zeros_like(current_representation_seq)
        active_indices_original = torch.arange(batch_size, device=input_ids_seq.device)
        current_active_representations_seq = current_representation_seq
        current_active_indices = active_indices_original.clone()

        for step in range(self.max_path_len):
            if not current_active_indices.numel(): break
            current_active_summary_repr = current_active_representations_seq.mean(dim=1)

            anim_sample_original_idx = 0
            is_anim_sample_active_for_signals = self.is_animating_sample and (
                        anim_sample_original_idx in current_active_indices)

            if is_anim_sample_active_for_signals:
                anim_idx_in_curr_batch = (current_active_indices == anim_sample_original_idx).nonzero(as_tuple=True)[
                    0].item()
                self._emit_signal("signal_to_router", repr_tensor(current_active_summary_repr[anim_idx_in_curr_batch]),
                                  [str(p) for p in paths_taken_indices[anim_sample_original_idx]])

            router_logits = self.router(current_active_summary_repr, current_visit_counts_batch[current_active_indices],
                                        self.max_visits_per_expert_in_path)
            router_probs = F.softmax(router_logits, dim=1)
            # Add small epsilon before log for stability if probs can be zero
            safe_router_probs = torch.clamp(router_probs, min=1e-9)
            step_entropy = -torch.sum(safe_router_probs * torch.log(safe_router_probs), dim=1).mean()

            total_router_entropy += step_entropy

            one_hot_decision = F.gumbel_softmax(router_logits, tau=1.0, hard=True) if self.training else F.one_hot(
                torch.argmax(router_logits, dim=1), num_classes=self.num_experts + 1).float()
            terminate_decision = one_hot_decision[:, self.num_experts]
            expert_selection_one_hot = one_hot_decision[:, :self.num_experts]

            if is_anim_sample_active_for_signals:
                anim_idx_in_curr_batch = (current_active_indices == anim_sample_original_idx).nonzero(as_tuple=True)[
                    0].item()
                self._emit_signal("signal_router_output", router_probs[anim_idx_in_curr_batch].tolist(),
                                  one_hot_decision[anim_idx_in_curr_batch].argmax().item(),
                                  one_hot_decision[anim_idx_in_curr_batch, self.num_experts].item() == 1)

            terminated_mask_in_active = (terminate_decision == 1)
            if terminated_mask_in_active.any():
                original_indices_terminated = current_active_indices[terminated_mask_in_active]
                final_representations_seq_batch[original_indices_terminated] = current_active_representations_seq[
                    terminated_mask_in_active]
                for orig_idx in original_indices_terminated.tolist(): paths_taken_indices[orig_idx].append('T')

            chose_expert_mask_in_active = (terminate_decision == 0)
            if not chose_expert_mask_in_active.any(): current_active_indices = torch.tensor([], dtype=torch.long,
                                                                                            device=input_ids_seq.device); break

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
                is_anim_sample_proc_expert_now = is_anim_sample_active_for_signals and (
                            next_active_indices[i].item() == anim_sample_original_idx)
                expert_input_seq_single = current_active_representations_seq[i].unsqueeze(0)
                if is_anim_sample_proc_expert_now: self._emit_signal("signal_to_expert", expert_idx_val, repr_tensor(
                    expert_input_seq_single.mean(dim=(0, 1))))
                expert_output_seq_single = self.experts[expert_idx_val](expert_input_seq_single)
                next_step_representations_seq_for_active[i] = expert_output_seq_single.squeeze(0)
                if is_anim_sample_proc_expert_now: self._emit_signal("signal_expert_output", expert_idx_val,
                                                                     repr_tensor(
                                                                         next_step_representations_seq_for_active[
                                                                             i].mean(dim=0)))

            current_active_representations_seq = next_step_representations_seq_for_active
            current_active_indices = next_active_indices

        if current_active_indices.numel() > 0:
            final_representations_seq_batch[current_active_indices] = current_active_representations_seq
            for orig_idx in current_active_indices.tolist(): paths_taken_indices[orig_idx].append('T_max')

        final_summary_repr_batch = final_representations_seq_batch.mean(dim=1)
        if self.is_animating_sample: self._emit_signal("signal_to_output_layer",
                                                       repr_tensor(final_summary_repr_batch[0]))

        logits = self.output_lm_head(final_summary_repr_batch)
        if self.is_animating_sample:
            predicted_token_idx = logits[0].argmax().item()
            predicted_token_str = self.animation_signals_obj.vocab.denumericalize([predicted_token_idx])[
                0] if self.animation_signals_obj and self.animation_signals_obj.vocab else str(predicted_token_idx)
            self._emit_signal("signal_final_prediction_lm", predicted_token_str, paths_taken_indices[0])

        return logits, total_router_entropy, paths_taken_indices


# --- PyQt Components (DiagramWidget, AnimationSignals, ContinuousLearningSignals are same) ---
def repr_tensor(tensor_data, num_elems=3):
    if tensor_data is None: return "None"
    if isinstance(tensor_data, list): return "[" + ", ".join([f"{x:.2f}" for x in tensor_data[:num_elems]]) + (
        "..." if len(tensor_data) > num_elems else "") + "]"
    if not isinstance(tensor_data, torch.Tensor) or tensor_data.numel() == 0: return str(tensor_data)
    items = tensor_data.detach().cpu().flatten().tolist()
    return "[" + ", ".join([f"{x:.2f}" for x in items[:num_elems]]) + (
        f"...L{len(items)}]" if len(items) > num_elems else "]")


class DiagramWidget(QWidget):
    def __init__(self, num_experts=NUM_EXPERTS_DEFAULT):
        super().__init__()
        self.setMinimumSize(500, 450);
        self.node_positions = {};
        self.active_elements = {}
        self.connections_to_draw = [];
        self.current_representation_text = ""
        self.router_probs_text = "";
        self.current_path_text = ""
        self.num_experts_to_draw = num_experts;
        self._setup_node_positions()

    def update_num_experts(self, num_experts):
        self.num_experts_to_draw = num_experts; self._setup_node_positions(); self.reset_highlights()

    def _setup_node_positions(self):
        W, H = self.width() or 500, self.height() or 450;
        self.node_size = QRectF(0, 0, 90, 30)
        self.node_positions = {"INPUT": QPointF(W / 2, H * 0.08), "EMBED": QPointF(W / 2, H * 0.22),
                               "ROUTER": QPointF(W / 2, H * 0.40)}
        expert_y = H * 0.60
        for i in range(self.num_experts_to_draw): self.node_positions[f"EXPERT_{i}"] = QPointF(
            W * (0.15 + (0.7 / (max(1, self.num_experts_to_draw - 1))) * i if self.num_experts_to_draw > 1 else 0.5),
            expert_y)
        self.node_positions["LM_HEAD"] = QPointF(W / 2, H * 0.85)
        self.base_connections = [("INPUT", "EMBED"), ("EMBED", "ROUTER")] + [("ROUTER", f"EXPERT_{i}") for i in
                                                                             range(self.num_experts_to_draw)] + [
                                    (f"EXPERT_{i}", "ROUTER") for i in range(self.num_experts_to_draw)] + [
                                    ("ROUTER", "LM_HEAD")]
        self.reset_highlights()

    def resizeEvent(self, event):
        self._setup_node_positions(); super().resizeEvent(event)

    def reset_highlights(self):
        self.active_elements.clear();
        self.connections_to_draw = []
        for s, d in self.base_connections:
            if s in self.node_positions and d in self.node_positions: self.connections_to_draw.append(
                (s, d, QColor("lightgray"), False))
        self.current_representation_text = "";
        self.router_probs_text = "";
        self.current_path_text = "Path: ";
        self.update()

    def highlight_node(self, name, active=True):
        if name not in self.node_positions: return
        if active:
            self.active_elements[name] = QColor("yellow")
        elif name in self.active_elements:
            del self.active_elements[name]
        self.update()

    def highlight_connection(self, from_name, to_name, active=True):
        if from_name not in self.node_positions or to_name not in self.node_positions: return
        updated = False
        for i, (s, d, c, _) in enumerate(self.connections_to_draw):
            if s == from_name and d == to_name: self.connections_to_draw[i] = (s, d,
                                                                               QColor("orange") if active else QColor(
                                                                                   "lightgray"),
                                                                               active); updated = True; break
        if not updated: self.connections_to_draw.append(
            (from_name, to_name, QColor("orange") if active else QColor("lightgray"), active))

        if active:
            for i, (s, d, c, _) in enumerate(self.connections_to_draw):
                if s == from_name and d != to_name: self.connections_to_draw[i] = (s, d, QColor("lightgray"), False)
        self.update()

    def set_representation_text(self, t):
        self.current_representation_text = f"Repr(mean): {t}"; self.update()

    def set_router_probs_text(self, p):
        if not p: self.router_probs_text = ""; return
        labels = ",".join([f"E{i}" for i in range(self.num_experts_to_draw)]) + ",Term"
        self.router_probs_text = f"Router({labels}): [{','.join([f'{x:.2f}' for x in p])}]";
        self.update()

    def set_current_path_text(self, p):
        self.current_path_text = "Path: " + "->".join(map(str, p)); self.update()

    def paintEvent(self, event):
        painter = QPainter(self);
        painter.setRenderHint(QPainter.Antialiasing)
        for s_name, d_name, col, is_active in self.connections_to_draw:
            if s_name not in self.node_positions or d_name not in self.node_positions: continue
            p1, p2 = self.node_positions[s_name], self.node_positions[d_name];
            painter.setPen(QPen(col, 2.5 if is_active else 1.5));
            painter.drawLine(p1, p2)
            angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x());
            arr_sz = 7
            arr_p1 = QPointF(p2.x() - arr_sz * math.cos(angle - math.pi / 7),
                             p2.y() - arr_sz * math.sin(angle - math.pi / 7))
            arr_p2 = QPointF(p2.x() - arr_sz * math.cos(angle + math.pi / 7),
                             p2.y() - arr_sz * math.sin(angle + math.pi / 7))
            painter.setBrush(col);
            painter.drawPolygon(QPolygonF([p2, arr_p1, arr_p2]))
        font = QFont("Arial", 7);
        painter.setFont(font)
        for name, pos in self.node_positions.items():
            r = QRectF(pos.x() - self.node_size.width() / 2, pos.y() - self.node_size.height() / 2,
                       self.node_size.width(), self.node_size.height())
            painter.setBrush(self.active_elements.get(name, QColor("lightblue")));
            painter.setPen(Qt.black);
            painter.drawRoundedRect(r, 3, 3)
            dn = name.replace("_", " ");
            painter.drawText(r, Qt.AlignCenter, dn if "EXPERT" not in dn else dn.split(" ")[0][0] + dn.split(" ")[1])
        info_font = QFont("Arial", 8);
        painter.setFont(info_font);
        painter.setPen(Qt.black)
        painter.drawText(QPointF(5, 12), self.current_representation_text);
        painter.drawText(QPointF(5, 28), self.router_probs_text);
        painter.drawText(QPointF(5, 44), self.current_path_text)


class AnimationSignals(QWidget):
    signal_input_text = pyqtSignal(str);
    signal_embedded = pyqtSignal(str)
    signal_to_router = pyqtSignal(str, list);
    signal_router_output = pyqtSignal(list, int, bool)
    signal_to_expert = pyqtSignal(int, str);
    signal_expert_output = pyqtSignal(int, str)
    signal_to_output_layer = pyqtSignal(str);
    signal_final_prediction_lm = pyqtSignal(str, list)

    def __init__(self, vocab_ref, parent=None): super().__init__(parent); self.vocab = vocab_ref


class ContinuousLearningSignals(QWidget):
    log_message = pyqtSignal(str);
    stats_update = pyqtSignal(dict)
    model_initialization_done = pyqtSignal();
    learning_stopped = pyqtSignal()

    def __init__(self, parent=None): super().__init__(parent)


# --- Continuous Learning Worker (with Grad Clip & Scheduler) ---
class ContinuousLearningWorker(QThread):
    def __init__(self, model_provider_fn, scheduler_provider_fn, data_provider_fn, hps, signals_obj,
                 parent=None):  # Added scheduler_provider_fn
        super().__init__(parent)
        self.model_provider_fn = model_provider_fn
        self.scheduler_provider_fn = scheduler_provider_fn  # For LR scheduler
        self.data_provider_fn = data_provider_fn
        self.hps = hps
        self.signals = signals_obj
        self._is_running = True;
        self._pause_for_animation = False;
        self._animate_this_sample_info = None
        self.train_loss_history = collections.deque(maxlen=MAX_PLOT_POINTS);
        self.val_loss_history = collections.deque(maxlen=MAX_PLOT_POINTS)
        self.val_perplexity_history = collections.deque(maxlen=MAX_PLOT_POINTS);
        self.avg_path_len_history = collections.deque(maxlen=MAX_PLOT_POINTS)
        self.router_entropy_history = collections.deque(maxlen=MAX_PLOT_POINTS);
        self.global_step = 0
        self.pathway_frequencies = collections.Counter();
        self.expert_usage_counts = collections.Counter()

    def run(self):
        self.signals.log_message.emit("Continuous learning worker started.")
        model, optimizer, criterion = self.model_provider_fn()
        scheduler = self.scheduler_provider_fn()  # Get scheduler
        train_loader, val_loader = self.data_provider_fn()

        if not model or not train_loader or not scheduler:
            self.signals.log_message.emit("ERROR: Worker: Model, Dataloader or Scheduler not available.")
            self._is_running = False;
            self.signals.learning_stopped.emit();
            return

        model.current_hps = self.hps
        train_iter = iter(train_loader)
        recent_losses = collections.deque(maxlen=SMOOTHING_WINDOW_LOSS)

        while self._is_running:
            if self._pause_for_animation and self._animate_this_sample_info:
                self.signals.log_message.emit("Pausing for animation...")
                model.is_animating_sample = True;
                ids, _, txt = self._animate_this_sample_info
                with torch.no_grad(): model(ids.unsqueeze(0), current_sample_text_for_anim=txt)
                model.is_animating_sample = False;
                self._animate_this_sample_info = None;
                self._pause_for_animation = False
                self.signals.log_message.emit("Animation finished, resuming...")

            try:
                input_ids_seq, target_ids, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader); input_ids_seq, target_ids, _ = next(train_iter)

            model.train();
            optimizer.zero_grad()
            logits, router_entropy, paths = model(input_ids_seq)
            task_loss = criterion(logits, target_ids)
            loss = task_loss - self.hps['router_entropy_coef'] * router_entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.hps['grad_clip_max_norm'])  # Grad clipping
            optimizer.step()
            if scheduler: scheduler.step()  # Step the LR scheduler

            recent_losses.append(task_loss.item())
            if len(recent_losses) == SMOOTHING_WINDOW_LOSS: self.train_loss_history.append(
                sum(recent_losses) / SMOOTHING_WINDOW_LOSS)
            for p in paths: self.pathway_frequencies[tuple(p)] += 1
            for p in paths:
                for expert_idx in p:
                    if isinstance(expert_idx, int): self.expert_usage_counts[expert_idx] += 1
            self.global_step += 1

            if self.global_step % self.hps['train_steps_per_eval'] == 0:
                model.eval();
                val_loss_accum = 0.0;
                val_samples = 0;
                path_lengths_eval = [];
                router_entropies_eval = []
                with torch.no_grad():
                    for val_in_ids, val_tgt_ids, _ in val_loader:
                        val_logits, val_r_ent, val_paths = model(val_in_ids)
                        val_loss_accum += criterion(val_logits, val_tgt_ids).item() * val_in_ids.size(0);
                        val_samples += val_in_ids.size(0)
                        for p_val in val_paths: path_lengths_eval.append(
                            len([el for el in p_val if el != 'T' and el != 'T_max']))
                        router_entropies_eval.append(val_r_ent.item())
                avg_val_loss = val_loss_accum / val_samples if val_samples > 0 else float('inf');
                perplexity = math.exp(avg_val_loss) if avg_val_loss != float('inf') else float('inf')
                self.val_loss_history.append(avg_val_loss);
                self.val_perplexity_history.append(perplexity)
                if path_lengths_eval: self.avg_path_len_history.append(sum(path_lengths_eval) / len(path_lengths_eval))
                if router_entropies_eval: self.router_entropy_history.append(
                    sum(router_entropies_eval) / len(router_entropies_eval))
                stats_payload = {'global_step': self.global_step, 'train_loss_hist': list(self.train_loss_history),
                                 'val_loss_hist': list(self.val_loss_history),
                                 'val_perp_hist': list(self.val_perplexity_history),
                                 'path_len_hist': list(self.avg_path_len_history),
                                 'router_entropy_hist': list(self.router_entropy_history),
                                 'pathway_freq': dict(self.pathway_frequencies),
                                 'expert_usage': dict(self.expert_usage_counts),
                                 'current_val_loss': avg_val_loss, 'current_perplexity': perplexity,
                                 'current_lr': optimizer.param_groups[0]['lr']}  # Log current LR
                self.signals.stats_update.emit(stats_payload)
            QThread.msleep(5)
        self.signals.log_message.emit("Continuous learning worker stopped.");
        self.signals.learning_stopped.emit()

    def stop_learning(self):
        self._is_running = False

    def request_animation(self, sample_info):
        self._animate_this_sample_info = sample_info; self._pause_for_animation = True


class MainWindow(QMainWindow):  # UI Init and connections largely same
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GoE Continuous Learning & Self-Instrumentation Engine v2")
        self.setGeometry(30, 30, 1500, 950)

        self.hps = self._get_default_hps()
        self.vocab = Vocabulary(max_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQ)
        self.train_dataset, self.val_dataset = None, None
        self.train_loader, self.val_loader = None, None
        self.model, self.optimizer, self.criterion, self.scheduler = None, None, None, None  # Added scheduler

        self.animation_signals_obj = AnimationSignals(self.vocab)
        self.continuous_learning_signals_obj = ContinuousLearningSignals()
        self.learning_worker = None

        self._init_ui()  # UI setup is complex, keeping it as is mostly
        self._connect_signals_basic()
        self.init_system_data_and_model()

        if PYQTGRAPH_AVAILABLE:
            self.plot_timer = QTimer();
            self.plot_timer.timeout.connect(self.update_plots_from_cache)
            self.plot_data_cache = None
        else:
            self.log_message("Plotting disabled as pyqtgraph is not available.")

    def _get_default_hps(self):
        return {'embed_dim': EMBED_DIM_DEFAULT, 'num_experts': NUM_EXPERTS_DEFAULT,
                'expert_nhead': EXPERT_NHEAD_DEFAULT,
                'expert_dim_feedforward': EXPERT_DIM_FEEDFORWARD_DEFAULT,
                'router_hidden_dim': ROUTER_HIDDEN_DIM_DEFAULT,
                'max_path_len': MAX_PATH_LEN_DEFAULT, 'max_visits_per_expert': MAX_VISITS_PER_EXPERT_IN_PATH_DEFAULT,
                'max_lm_seq_len': MAX_LM_SEQ_LEN, 'learning_rate': LEARNING_RATE, 'batch_size': BATCH_SIZE,
                'router_entropy_coef': ROUTER_ENTROPY_COEF, 'weight_decay': WEIGHT_DECAY,
                'grad_clip_max_norm': GRAD_CLIP_MAX_NORM,  # Added
                'scheduler_step_size_multiplier': SCHEDULER_STEP_SIZE_MULTIPLIER,  # Added
                'scheduler_gamma': SCHEDULER_GAMMA,  # Added
                'animation_delay': ANIMATION_DELAY_DEFAULT, 'train_steps_per_eval': TRAIN_STEPS_PER_EVAL}

    def _get_current_hps_from_ui(self):
        return {'embed_dim': self.embed_dim_spin.value(), 'num_experts': self.num_experts_spin.value(),
                'expert_nhead': self.expert_nhead_spin.value(), 'expert_dim_feedforward': self.expert_ff_spin.value(),
                'router_hidden_dim': self.router_hidden_spin.value(), 'max_path_len': self.max_path_len_spin.value(),
                'max_visits_per_expert': self.max_visits_spin.value(), 'max_lm_seq_len': MAX_LM_SEQ_LEN,
                'learning_rate': self.lr_spin.value(), 'batch_size': self.batch_size_spin.value(),
                'router_entropy_coef': self.entropy_coef_spin.value(), 'weight_decay': WEIGHT_DECAY,
                'grad_clip_max_norm': self.grad_clip_spin.value(),  # Added UI element needed
                'scheduler_step_size_multiplier': self.scheduler_step_mult_spin.value(),  # Added UI element needed
                'scheduler_gamma': self.scheduler_gamma_spin.value(),  # Added UI element needed
                'animation_delay': self.anim_delay_spin.value(),
                'train_steps_per_eval': self.train_eval_steps_spin.value()}

    def _init_ui(self):  # Add new HP controls
        main_widget = QWidget();
        self.setCentralWidget(main_widget);
        top_layout = QHBoxLayout(main_widget)
        left_panel = QVBoxLayout();
        left_panel.setSpacing(10)
        hps_group = QGroupBox("Hyperparameters");
        hps_form = QFormLayout()
        # Existing HPs
        self.num_experts_spin = QSpinBox();
        self.num_experts_spin.setRange(2, 5);
        self.num_experts_spin.setValue(self.hps['num_experts'])
        self.embed_dim_spin = QSpinBox();
        self.embed_dim_spin.setRange(16, 128);
        self.embed_dim_spin.setValue(self.hps['embed_dim'])  # Increased range
        self.expert_nhead_spin = QSpinBox();
        self.expert_nhead_spin.setRange(1, 8);
        self.expert_nhead_spin.setValue(self.hps['expert_nhead'])  # Increased range
        self.expert_ff_spin = QSpinBox();
        self.expert_ff_spin.setRange(32, 256);
        self.expert_ff_spin.setValue(self.hps['expert_dim_feedforward'])  # Increased range
        self.router_hidden_spin = QSpinBox();
        self.router_hidden_spin.setRange(16, 128);
        self.router_hidden_spin.setValue(self.hps['router_hidden_dim'])  # Increased range
        self.max_path_len_spin = QSpinBox();
        self.max_path_len_spin.setRange(1, 8);
        self.max_path_len_spin.setValue(self.hps['max_path_len'])
        self.max_visits_spin = QSpinBox();
        self.max_visits_spin.setRange(1, 3);
        self.max_visits_spin.setValue(self.hps['max_visits_per_expert'])
        self.lr_spin = QDoubleSpinBox();
        self.lr_spin.setRange(1e-5, 1e-2);
        self.lr_spin.setSingleStep(1e-4);
        self.lr_spin.setDecimals(5);
        self.lr_spin.setValue(self.hps['learning_rate'])
        self.batch_size_spin = QSpinBox();
        self.batch_size_spin.setRange(4, 32);
        self.batch_size_spin.setValue(self.hps['batch_size'])
        self.entropy_coef_spin = QDoubleSpinBox();
        self.entropy_coef_spin.setRange(0.0, 0.1);
        self.entropy_coef_spin.setSingleStep(0.001);
        self.entropy_coef_spin.setDecimals(4);
        self.entropy_coef_spin.setValue(self.hps['router_entropy_coef'])
        self.anim_delay_spin = QDoubleSpinBox();
        self.anim_delay_spin.setRange(0.0, 2.0);
        self.anim_delay_spin.setSingleStep(0.1);
        self.anim_delay_spin.setValue(self.hps['animation_delay'])
        self.train_eval_steps_spin = QSpinBox();
        self.train_eval_steps_spin.setRange(10, 500);
        self.train_eval_steps_spin.setSingleStep(10);
        self.train_eval_steps_spin.setValue(self.hps['train_steps_per_eval'])
        # New HPs for convergence
        self.grad_clip_spin = QDoubleSpinBox();
        self.grad_clip_spin.setRange(0.1, 10.0);
        self.grad_clip_spin.setSingleStep(0.1);
        self.grad_clip_spin.setValue(self.hps['grad_clip_max_norm'])
        self.scheduler_step_mult_spin = QSpinBox();
        self.scheduler_step_mult_spin.setRange(1, 50);
        self.scheduler_step_mult_spin.setValue(self.hps['scheduler_step_size_multiplier'])
        self.scheduler_gamma_spin = QDoubleSpinBox();
        self.scheduler_gamma_spin.setRange(0.5, 0.99);
        self.scheduler_gamma_spin.setSingleStep(0.01);
        self.scheduler_gamma_spin.setDecimals(2);
        self.scheduler_gamma_spin.setValue(self.hps['scheduler_gamma'])

        hps_widgets = [("Experts", self.num_experts_spin), ("EmbDim", self.embed_dim_spin),
                       ("E.NHead", self.expert_nhead_spin),
                       ("E.FFDim", self.expert_ff_spin), ("RtrHid", self.router_hidden_spin),
                       ("MaxPath", self.max_path_len_spin),
                       ("MaxVisits", self.max_visits_spin), ("LR", self.lr_spin), ("BatchSz", self.batch_size_spin),
                       ("EntrCoef", self.entropy_coef_spin), ("GradClip", self.grad_clip_spin),
                       ("Sched.StepMult", self.scheduler_step_mult_spin), ("Sched.Gamma", self.scheduler_gamma_spin),
                       ("AnimDelay", self.anim_delay_spin), ("EvalSteps", self.train_eval_steps_spin)]
        for name, widget in hps_widgets: hps_form.addRow(name + ":", widget)
        hps_group.setLayout(hps_form);
        left_panel.addWidget(hps_group)

        actions_group = QGroupBox("System Control");
        actions_layout = QVBoxLayout()
        self.reinit_button = QPushButton("Apply Settings & Reinitialize System")
        self.start_learn_button = QPushButton("Start Continuous Learning")
        self.stop_learn_button = QPushButton("Stop Learning");
        self.stop_learn_button.setEnabled(False)
        self.animate_val_button = QPushButton("Animate One Validation Sample");
        self.animate_val_button.setEnabled(False)
        actions_layout.addWidget(self.reinit_button);
        actions_layout.addWidget(self.start_learn_button)
        actions_layout.addWidget(self.stop_learn_button);
        actions_layout.addWidget(self.animate_val_button)
        actions_group.setLayout(actions_layout);
        left_panel.addWidget(actions_group)

        self.diagram_widget = DiagramWidget(num_experts=self.hps['num_experts'])
        left_panel.addWidget(self.diagram_widget);
        left_panel.addStretch(1)
        top_layout.addLayout(left_panel, stretch=1)

        right_panel = QVBoxLayout();
        self.tabs = QTabWidget()
        plots_tab = QWidget();
        plots_layout = QVBoxLayout(plots_tab)
        if PYQTGRAPH_AVAILABLE:
            pg.setConfigOptions(antialias=True, background='w', foreground='k')
            self.loss_plot_widget = pg.PlotWidget(title="Loss Curves");
            self.loss_plot_widget.addLegend(offset=(-10, 10))
            self.train_loss_curve = self.loss_plot_widget.plot(pen=pg.mkPen('b', width=2), name="Train Loss (Smooth)");
            self.val_loss_curve = self.loss_plot_widget.plot(pen=pg.mkPen('r', width=2), name="Val Loss")
            self.loss_plot_widget.setLabel('left', "Loss");
            self.loss_plot_widget.setLabel('bottom', "Eval Steps");
            plots_layout.addWidget(self.loss_plot_widget)
            self.perplexity_plot_widget = pg.PlotWidget(title="Validation Perplexity")
            self.perplexity_curve = self.perplexity_plot_widget.plot(pen=pg.mkPen('g', width=2), name="Perplexity")
            self.perplexity_plot_widget.setLabel('left', "Perplexity");
            self.perplexity_plot_widget.setLabel('bottom', "Eval Steps");
            plots_layout.addWidget(self.perplexity_plot_widget)
            self.aux_metrics_plot_widget = pg.PlotWidget(title="Auxiliary Metrics");
            self.aux_metrics_plot_widget.addLegend(offset=(-10, 10))
            self.path_len_curve = self.aux_metrics_plot_widget.plot(pen=pg.mkPen('c', width=2), name="Avg.PathLen");
            self.router_entropy_curve = self.aux_metrics_plot_widget.plot(pen=pg.mkPen('m', width=2),
                                                                          name="Avg.RouterEntr")
            self.aux_metrics_plot_widget.setLabel('left', "Value");
            self.aux_metrics_plot_widget.setLabel('bottom', "Eval Steps");
            plots_layout.addWidget(self.aux_metrics_plot_widget)
        else:
            plots_layout.addWidget(QLabel("pyqtgraph not installed. Plotting disabled."))
        self.tabs.addTab(plots_tab, "Live Metrics")

        stats_tab = QWidget();
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.addWidget(QLabel("Expert Usage Frequency:"))
        if PYQTGRAPH_AVAILABLE:
            self.expert_usage_plot_widget = pg.PlotWidget();
            self.expert_usage_bars = None
            stats_layout.addWidget(self.expert_usage_plot_widget, stretch=1)
        else:
            stats_layout.addWidget(QLabel("pyqtgraph not installed for bar chart."))
        stats_layout.addWidget(QLabel("Pathway Frequencies & Compilation Candidates:"))
        self.pathway_stats_text_edit = QTextEdit();
        self.pathway_stats_text_edit.setReadOnly(True)
        stats_layout.addWidget(self.pathway_stats_text_edit, stretch=1);
        self.tabs.addTab(stats_tab, "Expert & Pathway Analysis")

        log_sample_tab = QWidget();
        log_sample_layout = QVBoxLayout(log_sample_tab)
        log_sample_layout.addWidget(QLabel("System Log:"));
        self.log_text_edit = QTextEdit();
        self.log_text_edit.setReadOnly(True);
        self.log_text_edit.setFixedHeight(200)
        log_sample_layout.addWidget(self.log_text_edit);
        log_sample_layout.addWidget(QLabel("Current Animated Sample Info:"))
        self.current_sample_text_edit = QTextEdit();
        self.current_sample_text_edit.setReadOnly(True);
        self.current_sample_text_edit.setFixedHeight(100)
        log_sample_layout.addWidget(self.current_sample_text_edit);
        log_sample_layout.addStretch(1);
        self.tabs.addTab(log_sample_tab, "Log & Sample")
        right_panel.addWidget(self.tabs);
        top_layout.addLayout(right_panel, stretch=2)

    def _connect_signals_basic(self):  # Same
        self.reinit_button.clicked.connect(self.init_system_data_and_model)
        self.start_learn_button.clicked.connect(self.start_continuous_learning)
        self.stop_learn_button.clicked.connect(self.stop_continuous_learning)
        self.animate_val_button.clicked.connect(self.trigger_one_sample_animation)
        asig = self.animation_signals_obj;
        asig.signal_input_text.connect(self.on_input_text_anim);
        asig.signal_embedded.connect(self.on_embedded_anim)
        asig.signal_to_router.connect(self.on_to_router_anim);
        asig.signal_router_output.connect(self.on_router_output_anim)
        asig.signal_to_expert.connect(self.on_to_expert_anim);
        asig.signal_expert_output.connect(self.on_expert_output_anim)
        asig.signal_to_output_layer.connect(self.on_to_output_layer_anim);
        asig.signal_final_prediction_lm.connect(self.on_final_prediction_lm_anim)
        csig = self.continuous_learning_signals_obj;
        csig.log_message.connect(self.log_message);
        csig.stats_update.connect(self.handle_stats_update)
        csig.learning_stopped.connect(self.on_learning_stopped_ui_update)

    def init_system_data_and_model(self):  # Modified to include scheduler
        self.log_message("Initializing system...")
        if self.learning_worker and self.learning_worker.isRunning():
            self.log_message("Stopping active learning before reinitialization...")
            self.stop_continuous_learning();
            self.learning_worker.wait(2000)
        self.hps = self._get_current_hps_from_ui()

        all_tokens = tokenize_text(WIKITEXT_SAMPLE);
        split_idx = int(len(all_tokens) * TRAIN_SPLIT_RATIO)
        train_tokens, val_tokens = all_tokens[:split_idx], all_tokens[split_idx:]
        self.vocab = Vocabulary(max_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQ)
        for token in train_tokens: self.vocab.add_word(token)
        self.vocab.build_vocab();
        self.animation_signals_obj.vocab = self.vocab
        self.train_dataset = LanguageModelDataset(train_tokens, self.vocab, self.hps['max_lm_seq_len'], "train")
        self.val_dataset = LanguageModelDataset(val_tokens, self.vocab, self.hps['max_lm_seq_len'], "val")
        if not self.train_dataset.sequences or not self.val_dataset.sequences:
            self.log_message("ERROR: Not enough data. Adjust text or seq_len.");
            self.start_learn_button.setEnabled(False);
            return
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hps['batch_size'],
                                                        shuffle=True, drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hps['batch_size'],
                                                      shuffle=False)
        self.log_message(
            f"Data: Train {len(self.train_dataset)}s, Val {len(self.val_dataset)}s. Vocab: {len(self.vocab)}w.")

        self.model = GoEModel(len(self.vocab), self.hps['embed_dim'], self.hps['num_experts'], self.hps['expert_nhead'],
                              self.hps['expert_dim_feedforward'], self.hps['router_hidden_dim'],
                              self.hps['max_path_len'],
                              self.hps['max_visits_per_expert'], self.hps['max_lm_seq_len'])
        self.model.animation_signals_obj = self.animation_signals_obj
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.hps['learning_rate'],
                                     weight_decay=self.hps['weight_decay'])
        # Scheduler setup
        scheduler_actual_step_size = self.hps['train_steps_per_eval'] * self.hps['scheduler_step_size_multiplier']
        self.scheduler = StepLR(self.optimizer, step_size=max(1, scheduler_actual_step_size),
                                gamma=self.hps['scheduler_gamma'])
        self.log_message(
            f"LR Scheduler: StepLR, step_size={scheduler_actual_step_size} train batches, gamma={self.hps['scheduler_gamma']:.2f}")

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.word_to_idx["<pad>"])
        self.log_message(f"Model reinitialized with {self.hps['num_experts']} experts.")
        self.diagram_widget.update_num_experts(self.hps['num_experts'])
        self.pathway_stats_text_edit.clear();
        self.current_sample_text_edit.clear();
        self.reset_plots()
        self.start_learn_button.setEnabled(True);
        self.stop_learn_button.setEnabled(False);
        self.animate_val_button.setEnabled(False)
        self.continuous_learning_signals_obj.model_initialization_done.emit()
        self.log_message("System initialization complete.")

    def reset_plots(self):  # Same
        if not PYQTGRAPH_AVAILABLE: return
        self.train_loss_curve.clear();
        self.val_loss_curve.clear();
        self.perplexity_curve.clear()
        self.path_len_curve.clear();
        self.router_entropy_curve.clear()
        if self.expert_usage_bars: self.expert_usage_plot_widget.removeItem(
            self.expert_usage_bars); self.expert_usage_bars = None
        self.plot_data_cache = None

    def start_continuous_learning(self):  # Modified to pass scheduler_provider_fn
        if not self.model or not self.train_loader or not self.val_loader or not self.scheduler:
            self.log_message("System not fully initialized. Click 'Apply Settings & Reinitialize'.");
            return
        self.log_message("Starting continuous learning...")
        self.learning_worker = ContinuousLearningWorker(
            model_provider_fn=lambda: (self.model, self.optimizer, self.criterion),
            scheduler_provider_fn=lambda: self.scheduler,  # Pass scheduler
            data_provider_fn=lambda: (self.train_loader, self.val_loader),
            hps=self.hps, signals_obj=self.continuous_learning_signals_obj)
        self.learning_worker.start()
        if PYQTGRAPH_AVAILABLE and not self.plot_timer.isActive(): self.plot_timer.start(PLOT_UPDATE_INTERVAL_MS)
        self.start_learn_button.setEnabled(False);
        self.stop_learn_button.setEnabled(True)
        self.animate_val_button.setEnabled(True);
        self.reinit_button.setEnabled(False)

    def stop_continuous_learning(self):  # Same
        if self.learning_worker and self.learning_worker.isRunning():
            self.log_message("Requesting continuous learning to stop...");
            self.learning_worker.stop_learning()
        if PYQTGRAPH_AVAILABLE and self.plot_timer.isActive(): self.plot_timer.stop()

    def on_learning_stopped_ui_update(self):  # Same
        self.start_learn_button.setEnabled(True);
        self.stop_learn_button.setEnabled(False)
        self.animate_val_button.setEnabled(False);
        self.reinit_button.setEnabled(True)
        self.log_message("Learning process has stopped.")

    def trigger_one_sample_animation(self):  # Same
        if not self.learning_worker or not self.learning_worker.isRunning(): self.log_message(
            "Learning not active."); return
        if not self.val_dataset or not self.val_dataset.sequences: self.log_message(
            "Val dataset not available."); return
        self.diagram_widget.reset_highlights();
        self.current_sample_text_edit.clear()
        sample_idx = random.randint(0, len(self.val_dataset) - 1);
        input_ids, target_id, txt_snippet = self.val_dataset[sample_idx]
        self.log_message(f"Requesting animation for val sample: {txt_snippet[:50]}...")
        self.learning_worker.request_animation((input_ids, target_id, txt_snippet))

    def log_message(self, msg):
        self.log_text_edit.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def handle_stats_update(self, stats_payload):  # Modified to log LR
        self.plot_data_cache = stats_payload
        self.pathway_stats_text_edit.clear()
        path_freq = stats_payload.get('pathway_freq', {});
        total_paths = sum(path_freq.values())
        self.pathway_stats_text_edit.append(f"--- Pathway Frequencies (Total Logged: {total_paths}) ---")
        sorted_paths = sorted(path_freq.items(), key=lambda item: item[1], reverse=True)
        for i, (path, count) in enumerate(sorted_paths):
            if i < 15: self.pathway_stats_text_edit.append(
                f"{'->'.join(map(str, path))}: {count} ({count / max(1, total_paths):.2%})")
        if len(sorted_paths) > 15: self.pathway_stats_text_edit.append("...")
        self.pathway_stats_text_edit.append("\n--- Top Compilation Candidates (Usage > 5%) ---")
        found_cand = False
        for path, count in sorted_paths:
            if total_paths > 0 and count / total_paths > 0.05:
                self.pathway_stats_text_edit.append(f"MERGE: {'->'.join(map(str, path))} ({count / total_paths:.2%})");
                found_cand = True
        if not found_cand: self.pathway_stats_text_edit.append("No pathways currently meet >5% usage threshold.")
        gs = stats_payload.get('global_step', 0);
        cl = stats_payload.get('current_val_loss', float('nan'))
        cp = stats_payload.get('current_perplexity', float('nan'));
        curr_lr = stats_payload.get('current_lr', float('nan'))
        self.log_message(f"Step {gs}: Val Loss={cl:.4f}, Val PPL={cp:.2f}, LR={curr_lr:.2e}")

    def update_plots_from_cache(self):  # Same
        if not PYQTGRAPH_AVAILABLE or not self.plot_data_cache: return; stats = self.plot_data_cache
        self.train_loss_curve.setData(stats.get('train_loss_hist', []));
        self.val_loss_curve.setData(stats.get('val_loss_hist', []))
        self.perplexity_curve.setData(stats.get('val_perp_hist', []))
        self.path_len_curve.setData(stats.get('path_len_hist', []));
        self.router_entropy_curve.setData(stats.get('router_entropy_hist', []))
        expert_usage = stats.get('expert_usage', {});
        num_def_exp = self.hps.get('num_experts', NUM_EXPERTS_DEFAULT)
        x_ticks = [];
        heights = []
        for i in range(num_def_exp): x_ticks.append((i + 0.5, f"E{i}")); heights.append(expert_usage.get(i, 0))
        if self.expert_usage_bars: self.expert_usage_plot_widget.removeItem(self.expert_usage_bars)
        self.expert_usage_bars = pg.BarGraphItem(x=list(range(num_def_exp)), height=heights, width=0.6, brush='teal')
        self.expert_usage_plot_widget.addItem(self.expert_usage_bars)
        ax = self.expert_usage_plot_widget.getAxis('bottom');
        ax.setTicks([x_ticks])
        self.expert_usage_plot_widget.setLabel('left', "Usage Count");
        self.expert_usage_plot_widget.setTitle("Expert Usage Frequency")
        self.plot_data_cache = None

    # Animation Slot Handlers (Same)
    def on_input_text_anim(self, t):
        self.diagram_widget.reset_highlights();self.diagram_widget.highlight_node(
            "INPUT");self.current_sample_text_edit.setText(f"Input: {t[:100]}...")

    def on_embedded_anim(self, r):
        self.diagram_widget.highlight_node("INPUT", False);self.diagram_widget.highlight_connection("INPUT",
                                                                                                    "EMBED");self.diagram_widget.highlight_node(
            "EMBED");self.diagram_widget.set_representation_text(r)

    def on_to_router_anim(self, r, p):
        self.diagram_widget.highlight_node("EMBED", False); (last := p[-1],
                                                             self.diagram_widget.highlight_node(f"EXPERT_{last}",
                                                                                                False),
                                                             self.diagram_widget.highlight_connection(f"EXPERT_{last}",
                                                                                                      "ROUTER")) if p else self.diagram_widget.highlight_connection(
            "EMBED", "ROUTER"); self.diagram_widget.highlight_node(
            "ROUTER");self.diagram_widget.set_representation_text(r);self.diagram_widget.set_current_path_text(p)

    def on_router_output_anim(self, probs, idx, is_term):
        self.diagram_widget.set_router_probs_text(probs);self.diagram_widget.highlight_node("ROUTER",
                                                                                            False);self.diagram_widget.highlight_connection(
            "ROUTER", "LM_HEAD" if is_term else f"EXPERT_{idx}")

    def on_to_expert_anim(self, idx, r):
        self.diagram_widget.highlight_node(f"EXPERT_{idx}");self.diagram_widget.set_representation_text(r)

    def on_expert_output_anim(self, idx, r):
        self.diagram_widget.set_representation_text(r)

    def on_to_output_layer_anim(self, r):
        self.diagram_widget.highlight_node("ROUTER", False);self.diagram_widget.highlight_node(
            "LM_HEAD");self.diagram_widget.set_representation_text(r)

    def on_final_prediction_lm_anim(self, pred_tok, path):
        self.diagram_widget.highlight_node("LM_HEAD", True);self.diagram_widget.set_current_path_text(
            path);curr = self.current_sample_text_edit.toPlainText();self.current_sample_text_edit.setText(
            f"{curr}\nPred:'{pred_tok}',Path:{'->'.join(map(str, path))}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if not PYQTGRAPH_AVAILABLE: print(
        "CRITICAL ERROR: pyqtgraph is required. pip install pyqtgraph")  # Exit if no plots
    try:
        import torch
    except ImportError:
        print("CRITICAL: PyTorch not found."); sys.exit(1)
    if torch.cuda.is_available(): print("CUDA available. Demo runs on CPU for simplicity.")
    torch.set_default_device('cpu')
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())