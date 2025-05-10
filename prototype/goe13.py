import math
import sys
import time
import random
import collections
import warnings
from threading import Lock
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, \
    QTextEdit, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QTabWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QTimer
import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLScatterPlotItem

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("WARNING: pyqtgraph not found. Install with 'pip install pyqtgraph'")

MAX_VOCAB_SIZE = 10000
MIN_FREQ = 1
MAX_LM_SEQ_LEN = 32
TRAIN_SPLIT_RATIO = 0.8
EMBED_DIM_DEFAULT = 256
NUM_EXPERTS_DEFAULT = 8
EXPERT_NHEAD_DEFAULT = 8
EXPERT_DIM_FEEDFORWARD_DEFAULT = 1024
EXPERT_LAYERS_DEFAULT = 4
ROUTER_HIDDEN_DIM_DEFAULT = 256
MAX_PATH_LEN_DEFAULT = 8
MAX_VISITS_PER_EXPERT_DEFAULT = 3
GUMBEL_TAU_INIT = 2.0
DIVERSITY_LOSS_COEF = 0.02
Q_LOSS_COEF = 0.01
CONTRASTIVE_LOSS_COEF = 0.05
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
EFFECTIVE_BATCH_SIZE = 64
ROUTER_ENTROPY_COEF = 0.03
GRADIENT_CLIP_NORM = 0.3
WEIGHT_DECAY = 1e-6
TRAIN_STEPS_PER_EVAL = 10
SMOOTHING_WINDOW_LOSS = 5
WARMUP_STEPS = 200
TOTAL_STEPS = 2000
RESTART_PERIOD = 500
ANIMATION_DELAY_DEFAULT = 0.2
PLOT_UPDATE_INTERVAL_MS = 300
MAX_PLOT_POINTS = 500
DIAGRAM_NODE_WIDTH = 120
DIAGRAM_NODE_HEIGHT = 40
DIAGRAM_ARROW_SIZE = 10.0
WIKITEXT_SAMPLE = """
= Valkyria Chronicles III = 
Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria Chronicles series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the original Valkyria Chronicles and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who are assigned to sensitive operations and missions deemed too dangerous for the regular army . 
The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's theme is " despair " , with the plot exploring the Gallian military 's persecution of those deemed undesirable and the Nameless 's struggle for redemption . 
Upon release , the game received positive reviews and was praised by both Japanese and western critics . After release , it received downloadable content , as well as manga and drama CD adaptations . Despite the positive reception , Valkyria Chronicles III was not localized for Western territories . The decision was attributed to the poor sales of Valkyria Chronicles II , a troubled history with the PlayStation Portable , and the popularity of mobile gaming in the West . Various English fan translation projects were initiated , but were left unfinished due to the complexity of the game 's script and coding . A full English fan translation was eventually released in 2014 . 
= Gameplay =
Like its predecessors , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and participate in missions against enemy forces . The game is presented in a series of chapters , with story cutscenes and battles delivered in a book @-@ like format . Players navigate through menus to prepare their units , watch story segments , and progress through the game . Missions generally have a single objective , such as capturing an enemy base camp or defeating a specific enemy unit . Some missions have defeat conditions , such as the death of a key unit or the destruction of a key allied vehicle . 
The gameplay is divided between two modes : Command Mode and Action Mode . In Command Mode , players view an overhead map of the battlefield , showing allied and enemy units . Each turn , players are given a set number of Command Points ( CP ) , which are used to activate individual units or issue special commands . Selecting a unit initiates Action Mode , where players directly control the unit in a third @-@ person perspective . During Action Mode , units can move a limited distance and perform one action , such as attacking an enemy or healing an ally . After the action is performed , the unit remains on the field unless retreated by the player . Taking control of a unit costs 1 CP . If a unit is taken down by enemy fire during Action Mode , it is considered critically injured and must be evacuated by an allied unit within a set number of turns , or it will be permanently lost . 
Units are divided into several classes , each with unique abilities and weapons . For example , Scouts are highly mobile units with good reconnaissance capabilities , while Shocktroopers are slower but more heavily armed . Engineers can repair tanks and resupply ammunition , while Lancers are anti @-@ tank specialists . Each unit has a unique set of Potentials , which are special abilities that can be triggered under certain conditions , such.
"""


def augment_text(text: str) -> str:
    tokens = text.lower().split()
    synonyms = {"game": "title", "tactical": "strategic", "unit": "team", "mission": "operation", "battle": "conflict",
                "army": "force", "play": "engage", "story": "narrative"}
    return " ".join(synonyms.get(t, t) if random.random() < 0.15 else t + random.choice(
        ["", " ", "s"]) if random.random() < 0.05 else t for t in tokens).replace("play", "engage").replace("game",
                                                                                                            "title")


class Vocabulary:
    def __init__(self, max_size: Optional[int] = None, min_freq: int = 1):
        self.word_to_idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.word_counts = collections.Counter()
        self.max_size = max_size
        self.min_freq = min_freq
        self._finalized = False

    def add_word(self, word: str):
        self.word_counts[word] += 1

    def build_vocab(self):
        if self._finalized and len(self.word_to_idx) > 4: return
        self.word_to_idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        for word, count in self.word_counts.most_common():
            if self.max_size and len(self.word_to_idx) >= self.max_size: break
            if count < self.min_freq or word in self.word_to_idx: continue
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        self._finalized = True
        print(f"Vocabulary built with {len(self.word_to_idx)} words.")

    def __len__(self) -> int:
        return len(self.word_to_idx)

    def numericalize(self, text_tokens: List[str]) -> List[int]:
        if not self._finalized: warnings.warn("Vocabulary not finalized. Building now."); self.build_vocab()
        return [self.word_to_idx.get(token, self.word_to_idx["<unk>"]) for token in text_tokens]

    def denumericalize(self, indices: List[int]) -> List[str]:
        return [self.idx_to_word.get(idx, "<unk>") for idx in indices]


def tokenize_text(text: str) -> List[str]:
    return text.lower().split()


def create_lm_sequences(token_ids: List[int], seq_len: int) -> List[Tuple[List[int], int]]:
    return [(token_ids[i:i + seq_len], token_ids[i + seq_len]) for i in
            range(len(token_ids) - seq_len)] if token_ids and len(token_ids) > seq_len else []


class LanguageModelDataset(torch.utils.data.Dataset):
    def __init__(self, text_data_tokens: List[str], vocab: Vocabulary, seq_len: int, dataset_name: str = "train"):
        if not text_data_tokens: raise ValueError(f"{dataset_name} dataset received empty token list.")
        self.vocab = vocab
        self.seq_len = seq_len
        if dataset_name == "train" and not self.vocab._finalized:
            for token in text_data_tokens: self.vocab.add_word(token)
            self.vocab.build_vocab()
        self.sequences = create_lm_sequences(self.vocab.numericalize(text_data_tokens), self.seq_len)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        input_seq_ids, target_word_id = self.sequences[idx]
        try:
            input_text = " ".join(self.vocab.denumericalize(input_seq_ids))
            target_text = self.vocab.denumericalize([target_word_id])[0]
            original_text_snippet = f"{input_text} ...predict... {target_text}"
        except (KeyError, IndexError) as e:
            print(f"Error in __getitem__: Vocabulary issue - {e}")
            original_text_snippet = "Error creating text snippet"
        return torch.tensor(input_seq_ids, dtype=torch.long), torch.tensor(target_word_id,
                                                                           dtype=torch.long), original_text_snippet


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class GatedAttention(nn.Module):
    def __init__(self, embed_dim: int, nhead: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        gate_input = torch.cat([x, attn_output], dim=-1)
        return self.norm(x + torch.sigmoid(self.gate(gate_input)) * attn_output)


class Expert(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dim_feedforward: int, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({
            'attn': GatedAttention(embed_dim, nhead),
            'ffn': nn.Sequential(nn.Linear(embed_dim, dim_feedforward), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, embed_dim)),
            'norm1': nn.LayerNorm(embed_dim),
            'norm2': nn.LayerNorm(embed_dim)
        }) for _ in range(num_layers)])
        self.specialization_tag = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            residual = x_seq
            x_seq = layer['norm1'](x_seq)
            x_seq = layer['attn'](x_seq)
            x_seq = residual + self.dropout(x_seq)
            residual = x_seq
            x_seq = layer['norm2'](x_seq)
            x_seq = layer['ffn'](x_seq)
            x_seq = residual + self.dropout(x_seq)
        return x_seq + self.specialization_tag


class RoutingController(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts + 1)
        self.norm = nn.LayerNorm(num_experts + 1)
        self.q_values = nn.Parameter(torch.zeros(num_experts + 1))

    def forward(self, x_summary: torch.Tensor, visit_counts: torch.Tensor, max_visits: int) -> torch.Tensor:
        x = F.relu(self.fc1(x_summary))
        logits = self.norm(self.fc2(x).clamp(min=-10.0, max=10.0))
        mask = visit_counts >= max_visits
        extended_mask = torch.cat([mask, torch.zeros_like(mask[..., :1], device=mask.device)], dim=-1)
        return logits.masked_fill(extended_mask, -float('inf')) + self.q_values


class RunningStat:
    def __init__(self):
        self.mean = 0.0
        self.var = 0.0
        self.count = 0

    def update(self, x: torch.Tensor):
        if x.numel() == 0: return
        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item() if x.numel() > 1 else 0.0
        batch_count = x.numel()
        if self.count == 0:
            self.mean, self.var, self.count = batch_mean, batch_var, batch_count
        else:
            new_count = self.count + batch_count
            m2_old, m2_batch = self.var * self.count, batch_var * batch_count
            new_mean = (self.mean * self.count + batch_mean * batch_count) / new_count
            delta_mean_sq = (self.mean - batch_mean) ** 2
            m2_new = m2_old + m2_batch + delta_mean_sq * self.count * batch_count / new_count
            self.mean, self.var, self.count = new_mean, m2_new / new_count, new_count

    @property
    def std(self) -> float:
        return math.sqrt(self.var) if self.var >= 0 else 0.0

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        current_std = self.std
        return (x - self.mean) / (current_std + 1e-8)


class GoEModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_experts: int, expert_nhead: int,
                 expert_dim_feedforward: int, expert_layers: int, router_hidden_dim: int, max_path_len: int,
                 max_visits_per_expert: int, max_lm_seq_len: int, gumbel_tau_init: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.max_path_len = max_path_len
        self.max_visits_per_expert = max_visits_per_expert
        self.gumbel_tau_init = gumbel_tau_init
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.experts = nn.ModuleList(
            [Expert(embed_dim, expert_nhead, expert_dim_feedforward, expert_layers) for _ in range(num_experts)])
        self.router = RoutingController(embed_dim, router_hidden_dim, num_experts)
        self.output_lm_head = nn.Linear(embed_dim, vocab_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.path_optimizer = nn.Linear(embed_dim, self.max_path_len + 1)
        self.meta_controller = nn.Linear(embed_dim, 2)
        self.reward_stat = RunningStat()
        self.animation_signals_obj = None
        self.is_animating_sample = False
        self.current_hps = {}

    def _emit_signal(self, signal_type: str, *args):
        if self.is_animating_sample and self.animation_signals_obj:
            delay = self.current_hps.get('animation_delay', 0.1)
            getattr(self.animation_signals_obj, signal_type).emit(*args)
            if delay > 0: QThread.msleep(int(delay * 1000))

    def compute_diversity_loss(self, router_probs: torch.Tensor, usage_counts: Dict[int, int]) -> torch.Tensor:
        avg_probs = router_probs.mean(dim=0)
        total_usage = sum(usage_counts.values()) + 1e-10
        diversity_bonus = sum(1.0 / (usage_counts.get(i, 0) / total_usage + 1e-3) for i in range(self.num_experts))
        return -torch.sum(avg_probs * torch.log(avg_probs + 1e-10)) + 0.01 * diversity_bonus

    def compute_q_loss(self, logits: torch.Tensor, rewards: torch.Tensor, discount: float) -> torch.Tensor:
        q_values = self.router.q_values
        selected_actions = logits.argmax(dim=-1)
        selected_q = q_values[selected_actions]
        target_q = rewards + discount * q_values.max().detach().expand_as(rewards)
        return F.mse_loss(selected_q, target_q.clamp(-1.0, 1.0))

    def compute_contrastive_loss(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        if len(expert_outputs) < 2: return torch.tensor(0.0, device=expert_outputs[
            0].device if expert_outputs else torch.device('cpu'))
        sim_matrix = torch.stack(
            [F.cosine_similarity(o.mean(dim=1), e.mean(dim=1), dim=-1) for o in expert_outputs for e in
             expert_outputs]).view(len(expert_outputs), len(expert_outputs))
        return F.binary_cross_entropy_with_logits(sim_matrix, torch.eye(len(expert_outputs), device=sim_matrix.device))

    def forward(self, input_ids_seq: torch.Tensor, current_sample_text: Optional[str] = None,
                target_ids: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, List[List[Any]], torch.Tensor]:
        batch_size = input_ids_seq.size(0)
        device = input_ids_seq.device
        if self.is_animating_sample and current_sample_text: self._emit_signal("signal_input_text", current_sample_text)

        embedded_input = self.embedding(input_ids_seq) * math.sqrt(self.embed_dim)
        current_representations_seq = self.norm(self.pos_encoder(embedded_input))
        if self.is_animating_sample: self._emit_signal("signal_embedded",
                                                       repr_tensor(current_representations_seq[0].mean(dim=0)))

        total_router_entropy = torch.tensor(0.0, device=device)
        total_diversity_loss = torch.tensor(0.0, device=device)
        total_q_loss = torch.tensor(0.0, device=device)
        total_contrastive_loss = torch.tensor(0.0, device=device)

        paths_taken = [[] for _ in range(batch_size)]
        visit_counts_batch = torch.zeros((batch_size, self.num_experts), dtype=torch.long, device=device)
        final_representations_batch = current_representations_seq.clone()

        active_original_indices = torch.arange(batch_size, device=device)
        active_representations_seq = current_representations_seq

        expert_outputs_collected = []
        expert_usage_counts_step = collections.Counter()
        epsilon = 1e-10

        global_step = self.current_hps.get('global_step', 0)
        tau = max(self.gumbel_tau_init * (1 - global_step / max(1, TOTAL_STEPS)), 0.1)
        discount = 0.9 + 0.09 * (global_step / max(1, TOTAL_STEPS))

        step_logits_and_indices_list = []

        for _ in range(self.max_path_len):
            if not active_original_indices.numel(): break

            summary_repr = self.norm(active_representations_seq.mean(dim=1))

            # Animation logic for the 0-th sample of the original batch
            anim_original_idx_0 = 0
            is_anim_active_for_idx_0 = self.is_animating_sample and (anim_original_idx_0 in active_original_indices)
            if is_anim_active_for_idx_0:
                anim_current_batch_idx = (active_original_indices == anim_original_idx_0).nonzero(as_tuple=True)[
                    0].item()
                self._emit_signal("signal_to_router", repr_tensor(summary_repr[anim_current_batch_idx]),
                                  [str(p) for p in paths_taken[anim_original_idx_0]])

            router_logits = self.router(summary_repr, visit_counts_batch[active_original_indices],
                                        self.max_visits_per_expert)
            router_probs = F.softmax(router_logits, dim=1).clamp(min=epsilon, max=1.0 - epsilon)

            step_entropy = -torch.sum(router_probs * torch.log(router_probs + epsilon), dim=1).mean()
            total_router_entropy += step_entropy
            total_diversity_loss += self.compute_diversity_loss(router_probs, expert_usage_counts_step)

            gumbel_hard = self.training
            one_hot_decision = F.gumbel_softmax(router_logits, tau=tau, hard=gumbel_hard) if gumbel_hard else F.one_hot(
                torch.argmax(router_logits, dim=1), num_classes=self.num_experts + 1).float()

            step_logits_and_indices_list.append((router_logits.clone(), active_original_indices.clone()))

            terminate_decision = one_hot_decision[:, self.num_experts]
            expert_selection = one_hot_decision[:, :self.num_experts]

            if is_anim_active_for_idx_0:
                anim_current_batch_idx = (active_original_indices == anim_original_idx_0).nonzero(as_tuple=True)[
                    0].item()
                self._emit_signal("signal_router_output", router_probs[anim_current_batch_idx].tolist(),
                                  one_hot_decision[anim_current_batch_idx].argmax().item(),
                                  one_hot_decision[anim_current_batch_idx, self.num_experts].item() == 1)

            terminated_mask_active = (terminate_decision == 1)
            if terminated_mask_active.any():
                terminated_orig_indices = active_original_indices[terminated_mask_active]
                final_representations_batch[terminated_orig_indices] = active_representations_seq[
                    terminated_mask_active]
                for orig_idx in terminated_orig_indices.tolist(): paths_taken[orig_idx].append('T')

            chose_expert_mask_active = (terminate_decision == 0)
            if not chose_expert_mask_active.any():
                active_original_indices = torch.tensor([], dtype=torch.long, device=device)
                break

            active_original_indices = active_original_indices[chose_expert_mask_active]
            active_representations_seq = active_representations_seq[chose_expert_mask_active]
            current_expert_selection = expert_selection[chose_expert_mask_active]
            chosen_expert_indices_active = current_expert_selection.argmax(dim=1)

            for i, orig_idx in enumerate(active_original_indices.tolist()):
                expert_idx = chosen_expert_indices_active[i].item()
                paths_taken[orig_idx].append(expert_idx)
                visit_counts_batch[orig_idx, expert_idx] += 1
                expert_usage_counts_step[expert_idx] += 1

            next_active_representations_seq = torch.zeros_like(active_representations_seq)
            current_step_expert_outputs = []
            for i, expert_idx in enumerate(chosen_expert_indices_active.tolist()):
                is_anim_for_this_expert_io = self.is_animating_sample and (
                            active_original_indices[i].item() == anim_original_idx_0)
                expert_input = active_representations_seq[i].unsqueeze(0)
                if is_anim_for_this_expert_io: self._emit_signal("signal_to_expert", expert_idx,
                                                                 repr_tensor(expert_input.mean(dim=(0, 1))))
                expert_output_tensor = self.experts[expert_idx](expert_input)
                next_active_representations_seq[i] = expert_output_tensor.squeeze(0)
                current_step_expert_outputs.append(expert_output_tensor)
                if is_anim_for_this_expert_io: self._emit_signal("signal_expert_output", expert_idx, repr_tensor(
                    next_active_representations_seq[i].mean(dim=0)))

            expert_outputs_collected.extend(current_step_expert_outputs)
            active_representations_seq = next_active_representations_seq

        if active_original_indices.numel():
            final_representations_batch[active_original_indices] = active_representations_seq
            for orig_idx in active_original_indices.tolist(): paths_taken[orig_idx].append('T_max')

        final_summary_repr = self.norm(final_representations_batch.mean(dim=1))
        logits = self.output_lm_head(final_summary_repr)

        path_loss = torch.tensor(0.0, device=device)
        meta_loss = torch.tensor(0.0, device=device)

        if self.training:
            path_logits_calc = self.path_optimizer(final_summary_repr)
            # Path length is number of expert steps, so len(path_taken) - 1 (due to 'T' or 'T_max')
            path_targets = torch.tensor([len(p) - 1 for p in paths_taken], device=device, dtype=torch.long).clamp(0,
                                                                                                                  self.max_path_len)
            path_loss = F.cross_entropy(path_logits_calc, path_targets)

            meta_logits_calc = self.meta_controller(final_summary_repr)
            meta_target = torch.full_like(meta_logits_calc[:, 0], self.num_experts, dtype=torch.float)
            meta_loss = F.mse_loss(meta_logits_calc[:, 0], meta_target)
            total_contrastive_loss = self.compute_contrastive_loss(expert_outputs_collected)

            rewards_full_batch = -F.cross_entropy(logits, target_ids,
                                                  reduction='none').detach() if target_ids is not None else torch.ones(
                batch_size, device=device)
            self.reward_stat.update(rewards_full_batch)
            rewards_normalized_full_batch = self.reward_stat.normalize(rewards_full_batch).clamp(-1.0, 1.0)

            for step_router_logits, step_active_indices in step_logits_and_indices_list:
                if step_active_indices.numel() > 0:
                    rewards_for_step = rewards_normalized_full_batch[step_active_indices]
                    total_q_loss += self.compute_q_loss(step_router_logits, rewards_for_step, discount)

        if self.is_animating_sample:
            self._emit_signal("signal_to_output_layer", repr_tensor(final_summary_repr[anim_original_idx_0]))
            predicted_token_idx = logits[anim_original_idx_0].argmax().item()
            vocab = self.animation_signals_obj.vocab if self.animation_signals_obj else None
            predicted_token_str = vocab.denumericalize([predicted_token_idx])[0] if vocab else "UNK_ANIM"
            self._emit_signal("signal_final_prediction_lm", predicted_token_str, paths_taken[anim_original_idx_0])

        aux_loss = total_diversity_loss + Q_LOSS_COEF * total_q_loss + CONTRASTIVE_LOSS_COEF * total_contrastive_loss + path_loss + 0.01 * meta_loss
        return logits, total_router_entropy, paths_taken, aux_loss


def repr_tensor(tensor_data: Any, num_elems: int = 3) -> str:
    if tensor_data is None: return "None"
    if isinstance(tensor_data,
                  list): return f"[{(', '.join(f'{x:.2f}' for x in tensor_data[:num_elems])) + ('...' if len(tensor_data) > num_elems else '')}]"
    if not isinstance(tensor_data, torch.Tensor) or tensor_data.numel() == 0: return str(tensor_data)
    items = tensor_data.detach().cpu().flatten().tolist()
    return f"[{(', '.join(f'{x:.2f}' for x in items[:num_elems])) + (f'...L{len(items)}]' if len(items) > num_elems else ']')}"


class DiagramWidget(QWidget):
    def __init__(self, num_experts: int = NUM_EXPERTS_DEFAULT):
        super().__init__()
        self.setMinimumSize(600, 500)
        self.node_positions = {}
        self.active_elements = {}
        self.connections_to_draw = []
        self.current_representation_text = self.router_probs_text = self.current_path_text = self.expert_usage_text = self.stability_text = ""
        self.num_experts = num_experts
        self.node_size = QRectF(0, 0, DIAGRAM_NODE_WIDTH, DIAGRAM_NODE_HEIGHT)
        self._setup_node_positions()

    def update_num_experts(self, num_experts: int):
        if num_experts < 2: raise ValueError("Number of experts must be at least 2.")
        self.num_experts = num_experts
        self._setup_node_positions()
        self.reset_highlights()

    def _setup_node_positions(self):
        W, H = self.width() or 600, self.height() or 500
        self.node_positions = {"INPUT": QPointF(W / 2, H * 0.08), "EMBED": QPointF(W / 2, H * 0.22),
                               "ROUTER": QPointF(W / 2, H * 0.40)}
        for i in range(self.num_experts):
            x_pos = W * (0.1 + (0.8 / max(1, self.num_experts - 1)) * i if self.num_experts > 1 else 0.5)
            self.node_positions[f"EXPERT_{i}"] = QPointF(x_pos, H * 0.60)
        self.node_positions["LM_HEAD"] = QPointF(W / 2, H * 0.85)
        self.base_connections = [("INPUT", "EMBED"), ("EMBED", "ROUTER")] + \
                                [(f"ROUTER", f"EXPERT_{i}") for i in range(self.num_experts)] + \
                                [(f"EXPERT_{i}", "ROUTER") for i in range(self.num_experts)] + \
                                [("ROUTER", "LM_HEAD")]
        self.reset_highlights()

    def resizeEvent(self, event):
        self._setup_node_positions()
        super().resizeEvent(event)

    def reset_highlights(self):
        self.active_elements.clear()
        self.connections_to_draw = [(s, d, QColor("lightgray"), False) for s, d in self.base_connections]
        self.current_representation_text = self.router_probs_text = self.current_path_text = self.expert_usage_text = self.stability_text = ""
        self.update()

    def highlight_node(self, name: str, active: bool = True):
        if name not in self.node_positions: return
        if active:
            self.active_elements[name] = QColor("yellow")
        elif name in self.active_elements:
            del self.active_elements[name]
        self.update()

    def highlight_connection(self, from_name: str, to_name: str, active: bool = True):
        if from_name not in self.node_positions or to_name not in self.node_positions: return
        updated = False
        for i, (s, d, _, _) in enumerate(self.connections_to_draw):
            if s == from_name and d == to_name:
                self.connections_to_draw[i] = (s, d, QColor("orange") if active else QColor("lightgray"), active)
                updated = True;
                break
        if not updated: self.connections_to_draw.append(
            (from_name, to_name, QColor("orange") if active else QColor("lightgray"), active))

        if active:  # De-highlight other connections from the same source node
            for i, (s, d, _, is_act) in enumerate(self.connections_to_draw):
                if s == from_name and d != to_name and is_act:
                    self.connections_to_draw[i] = (s, d, QColor("lightgray"), False)
        self.update()

    def set_representation_text(self, text: str):
        self.current_representation_text = f"Repr(mean): {text}"; self.update()

    def set_router_probs_text(self, probs: List[float]):
        if not probs: self.router_probs_text = ""; return
        labels = ",".join([f"E{i}" for i in range(self.num_experts)]) + ",Term"
        self.router_probs_text = f"Router({labels}): [{','.join(f'{x:.2f}' for x in probs)}]";
        self.update()

    def set_current_path_text(self, path: List[Any]):
        self.current_path_text = "Path: " + "->".join(map(str, path)); self.update()

    def set_expert_usage_text(self, usage_counts: Dict[int, int]):
        total = sum(usage_counts.values()) + 1e-10
        self.expert_usage_text = f"Expert Usage: {', '.join(f'E{i}:{usage_counts.get(i, 0) / total:.2%}' for i in range(self.num_experts))}"
        self.stability_text = "WARNING: Routing collapse!" if total > 0 and any(
            c / total > 0.8 for c in usage_counts.values()) else "Routing Stable"
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self);
        painter.setRenderHint(QPainter.Antialiasing)
        for s_name, d_name, col, is_active in self.connections_to_draw:
            if s_name not in self.node_positions or d_name not in self.node_positions: continue
            p1, p2 = self.node_positions[s_name], self.node_positions[d_name]
            painter.setPen(QPen(col, 3.0 if is_active else 1.5));
            painter.drawLine(p1, p2)
            angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
            arr_p1 = QPointF(p2.x() - DIAGRAM_ARROW_SIZE * math.cos(angle - math.pi / 7),
                             p2.y() - DIAGRAM_ARROW_SIZE * math.sin(angle - math.pi / 7))
            arr_p2 = QPointF(p2.x() - DIAGRAM_ARROW_SIZE * math.cos(angle + math.pi / 7),
                             p2.y() - DIAGRAM_ARROW_SIZE * math.sin(angle + math.pi / 7))
            painter.setBrush(col);
            painter.drawPolygon(QPolygonF([p2, arr_p1, arr_p2]))

        font = QFont("Arial", 8);
        painter.setFont(font)
        for name, pos in self.node_positions.items():
            r = QRectF(pos.x() - self.node_size.width() / 2, pos.y() - self.node_size.height() / 2,
                       self.node_size.width(), self.node_size.height())
            painter.setBrush(self.active_elements.get(name, QColor("lightblue")));
            painter.setPen(Qt.black)
            painter.drawRoundedRect(r, 5, 5)
            display_name = name.replace("_", " ").replace("EXPERT", "E").replace("EMBED", "EMBED+POS")
            painter.drawText(r, Qt.AlignCenter, display_name)

        info_font = QFont("Arial", 9);
        painter.setFont(info_font);
        painter.setPen(Qt.black)
        texts_to_draw = [self.current_representation_text, self.router_probs_text, self.current_path_text,
                         self.expert_usage_text]
        for i, text in enumerate(texts_to_draw): painter.drawText(QPointF(5, 15 + i * 20), text)
        painter.setPen(QColor("red") if "WARNING" in self.stability_text else Qt.black)
        painter.drawText(QPointF(5, 15 + len(texts_to_draw) * 20), self.stability_text)


class AnimationSignals(QWidget):
    signal_input_text = pyqtSignal(str);
    signal_embedded = pyqtSignal(str)
    signal_to_router = pyqtSignal(str, list);
    signal_router_output = pyqtSignal(list, int, bool)
    signal_to_expert = pyqtSignal(int, str);
    signal_expert_output = pyqtSignal(int, str)
    signal_to_output_layer = pyqtSignal(str);
    signal_final_prediction_lm = pyqtSignal(str, list)

    def __init__(self, vocab_ref: Vocabulary, parent: Optional[QWidget] = None): super().__init__(
        parent); self.vocab = vocab_ref


class ContinuousLearningSignals(QWidget):
    log_message = pyqtSignal(str);
    stats_update = pyqtSignal(dict)
    model_initialization_done = pyqtSignal();
    learning_stopped = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None): super().__init__(parent)


class ContinuousLearningWorker(QThread):
    def __init__(self, model_provider_fn: callable, data_provider_fn: callable, hps: Dict[str, Any],
                 signals_obj: ContinuousLearningSignals, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model_provider_fn, self.data_provider_fn, self.hps, self.signals = model_provider_fn, data_provider_fn, hps.copy(), signals_obj
        self._is_running, self._pause_for_animation, self._animate_sample_info = True, False, None
        self.train_loss_history = collections.deque(maxlen=MAX_PLOT_POINTS);
        self.val_loss_history = collections.deque(maxlen=MAX_PLOT_POINTS)
        self.val_perplexity_history = collections.deque(maxlen=MAX_PLOT_POINTS);
        self.avg_path_len_history = collections.deque(maxlen=MAX_PLOT_POINTS)
        self.router_entropy_history = collections.deque(maxlen=MAX_PLOT_POINTS);
        self.reward_history = collections.deque(maxlen=MAX_PLOT_POINTS)
        self.global_step, self.pathway_frequencies, self.expert_usage_counts = 0, collections.Counter(), collections.Counter()
        self.grad_accum_steps = EFFECTIVE_BATCH_SIZE // self.hps['batch_size']

    def run(self):
        self.signals.log_message.emit("Continuous learning started.")
        model, optimizer, criterion = self.model_provider_fn()
        train_loader, val_loader = self.data_provider_fn()
        if not all([model, optimizer, criterion, train_loader, val_loader]):
            self.signals.log_message.emit("ERROR: Missing model components or data.");
            self._is_running = False;
            self.signals.learning_stopped.emit();
            return

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=RESTART_PERIOD, T_mult=1, eta_min=1e-6)
        train_iter = iter(train_loader)
        recent_losses = collections.deque(maxlen=SMOOTHING_WINDOW_LOSS)
        accumulated_grads, lr_adjust_factor, grad_norm = 0, 1.0, 0.0

        for _ in range(TOTAL_STEPS):
            if not self._is_running: break

            model.current_hps = self.hps.copy();
            model.current_hps['global_step'] = self.global_step

            if self._pause_for_animation and self._animate_sample_info:
                self.signals.log_message.emit("Pausing for animation...");
                model.is_animating_sample = True
                ids, _, txt = self._animate_sample_info
                with torch.no_grad(): model(ids.unsqueeze(0), current_sample_text=txt)
                model.is_animating_sample = False;
                self._animate_sample_info = None;
                self._pause_for_animation = False
                self.signals.log_message.emit("Animation finished, resuming...");
                QThread.msleep(50)

            try:
                input_ids_seq, target_ids, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader); input_ids_seq, target_ids, _ = next(train_iter)

            model.train()
            current_entropy_coef = self.hps['router_entropy_coef'] * (1 - self.global_step / max(1, TOTAL_STEPS))
            logits, router_entropy, paths, aux_loss = model(input_ids_seq, target_ids=target_ids)

            task_loss = criterion(logits,
                                  target_ids) if logits.numel() > 0 and target_ids.numel() > 0 else torch.tensor(0.0,
                                                                                                                 device=logits.device)
            loss = (task_loss - current_entropy_coef * router_entropy + aux_loss) / self.grad_accum_steps
            loss.backward();
            accumulated_grads += 1

            if accumulated_grads >= self.grad_accum_steps:
                clip_norm_val = self.hps.get('gradient_clip_norm', GRADIENT_CLIP_NORM)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm_val).item()
                if grad_norm > 5.0 * clip_norm_val: lr_adjust_factor = max(0.5,
                                                                           lr_adjust_factor * 0.9); self.signals.log_message.emit(
                    f"High grad norm: {grad_norm:.2f}, LR factor: {lr_adjust_factor:.2f}")
                for param_group in optimizer.param_groups: param_group['lr'] = self.hps[
                                                                                   'learning_rate'] * lr_adjust_factor
                optimizer.step();
                optimizer.zero_grad()
                if self.global_step >= WARMUP_STEPS: scheduler.step()
                accumulated_grads = 0

            recent_losses.append(task_loss.item())
            if len(recent_losses) == SMOOTHING_WINDOW_LOSS: self.train_loss_history.append(
                sum(recent_losses) / SMOOTHING_WINDOW_LOSS)

            for p in paths: self.pathway_frequencies[tuple(p)] += 1; self.expert_usage_counts.update(
                el for el in p if isinstance(el, int))
            if logits.numel() > 0 and target_ids.numel() > 0: self.reward_history.append(
                -F.cross_entropy(logits, target_ids, reduction='none').detach().mean().item())

            self.global_step += 1
            if self.global_step % self.hps['train_steps_per_eval'] == 0:
                model.eval();
                val_loss_accum, val_samples = 0.0, 0.0;
                path_lens, router_ents = [], []
                with torch.no_grad():
                    for val_in_ids, val_tgt_ids, _ in val_loader:
                        val_logits, val_router_ent, val_paths, _ = model(val_in_ids, target_ids=val_tgt_ids)
                        if val_logits.numel() > 0 and val_tgt_ids.numel() > 0:
                            val_loss_accum += criterion(val_logits, val_tgt_ids).item() * val_in_ids.size(0)
                            val_samples += val_in_ids.size(0)
                        path_lens.extend(len([el for el in p if el not in ('T', 'T_max')]) for p in val_paths)
                        router_ents.append(val_router_ent.item())
                avg_val_loss = val_loss_accum / val_samples if val_samples > 0 else float('inf')
                perplexity = math.exp(avg_val_loss) if 0 <= avg_val_loss <= 700 else float(
                    'inf')  # Avoid math.exp overflow
                self.val_loss_history.append(avg_val_loss);
                self.val_perplexity_history.append(perplexity)
                self.avg_path_len_history.append(sum(path_lens) / len(path_lens) if path_lens else 0.0)
                self.router_entropy_history.append(sum(router_ents) / len(router_ents) if router_ents else 0.0)

                self.signals.stats_update.emit({
                    'global_step': self.global_step, 'train_loss_hist': list(self.train_loss_history),
                    'val_loss_hist': list(self.val_loss_history), 'val_perp_hist': list(self.val_perplexity_history),
                    'path_len_hist': list(self.avg_path_len_history),
                    'router_entropy_hist': list(self.router_entropy_history),
                    'pathway_freq': dict(self.pathway_frequencies), 'expert_usage': dict(self.expert_usage_counts),
                    'current_val_loss': avg_val_loss, 'current_perplexity': perplexity, 'grad_norm': grad_norm,
                    'reward_mean': sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0.0
                })
            QThread.msleep(1)
        self.signals.log_message.emit("Continuous learning stopped.");
        self.signals.learning_stopped.emit()

    def stop_learning(self):
        self._is_running = False

    def request_animation(self, sample_info: Tuple[torch.Tensor, torch.Tensor, str]):
        self._animate_sample_info, self._pause_for_animation = sample_info, True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph-of-Experts Language Model");
        self.setGeometry(30, 30, 1600, 1000)
        self.hps = self._get_default_hps()
        self.vocab = Vocabulary(max_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQ)
        self.train_dataset, self.val_dataset, self.train_loader, self.val_loader = None, None, None, None
        self.model, self.optimizer, self.criterion, self.learning_worker = None, None, None, None
        self.plot_data_cache_lock = Lock();
        self.plot_data_cache = None
        self.animation_signals_obj = AnimationSignals(self.vocab);
        self.continuous_learning_signals_obj = ContinuousLearningSignals()
        self._init_ui();
        self._connect_signals();
        self.init_system_data_and_model()
        if PYQTGRAPH_AVAILABLE: self.plot_timer = QTimer(); self.plot_timer.timeout.connect(
            self.update_plots_from_cache)

    def _create_spinbox(self, min_val, max_val, default_val, step=None, decimals=None):
        sb = QSpinBox() if isinstance(default_val, int) else QDoubleSpinBox()
        sb.setRange(min_val, max_val);
        sb.setValue(default_val)
        if step: sb.setSingleStep(step)
        if decimals: sb.setDecimals(decimals)
        return sb

    def _get_default_hps(self) -> Dict[str, Any]:
        return {'embed_dim': EMBED_DIM_DEFAULT, 'num_experts': NUM_EXPERTS_DEFAULT,
                'expert_nhead': EXPERT_NHEAD_DEFAULT, 'expert_dim_feedforward': EXPERT_DIM_FEEDFORWARD_DEFAULT,
                'expert_layers': EXPERT_LAYERS_DEFAULT, 'router_hidden_dim': ROUTER_HIDDEN_DIM_DEFAULT,
                'max_path_len': MAX_PATH_LEN_DEFAULT, 'max_visits_per_expert': MAX_VISITS_PER_EXPERT_DEFAULT,
                'max_lm_seq_len': MAX_LM_SEQ_LEN, 'gumbel_tau': GUMBEL_TAU_INIT, 'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE, 'router_entropy_coef': ROUTER_ENTROPY_COEF, 'weight_decay': WEIGHT_DECAY,
                'animation_delay': ANIMATION_DELAY_DEFAULT, 'train_steps_per_eval': TRAIN_STEPS_PER_EVAL,
                'gradient_clip_norm': GRADIENT_CLIP_NORM, 'global_step': 0}

    def _get_current_hps_from_ui(self) -> Dict[str, Any]:
        return {name: spin.value() for name, spin in self.hps_spins.items()} | {'max_lm_seq_len': MAX_LM_SEQ_LEN,
                                                                                'weight_decay': WEIGHT_DECAY,
                                                                                'gradient_clip_norm': GRADIENT_CLIP_NORM,
                                                                                'global_step': 0}  # Add non-spin HPs

    def validate_hps(self, hps: Dict[str, Any]) -> List[str]:
        errors = []
        if hps['embed_dim'] % hps['expert_nhead'] != 0: errors.append("embed_dim must be divisible by expert_nhead")
        if hps['max_path_len'] < 1: errors.append("max_path_len must be at least 1")
        if self.train_dataset and hps['batch_size'] > len(self.train_dataset): errors.append(
            f"batch_size ({hps['batch_size']}) > train dataset size ({len(self.train_dataset)})")
        if EFFECTIVE_BATCH_SIZE % hps['batch_size'] != 0: errors.append(
            f"batch_size ({hps['batch_size']}) must divide effective batch size ({EFFECTIVE_BATCH_SIZE})")
        return errors

    def _init_ui(self):
        main_widget = QWidget();
        self.setCentralWidget(main_widget);
        top_layout = QHBoxLayout(main_widget)
        left_panel = QVBoxLayout();
        left_panel.setSpacing(10)
        hps_group = QGroupBox("Hyperparameters");
        hps_form = QFormLayout()
        self.hps_spins = {
            'num_experts': self._create_spinbox(2, 12, NUM_EXPERTS_DEFAULT),
            'embed_dim': self._create_spinbox(128, 512, EMBED_DIM_DEFAULT),
            'expert_nhead': self._create_spinbox(1, 16, EXPERT_NHEAD_DEFAULT),
            'expert_dim_feedforward': self._create_spinbox(512, 2048, EXPERT_DIM_FEEDFORWARD_DEFAULT),
            'expert_layers': self._create_spinbox(2, 6, EXPERT_LAYERS_DEFAULT),
            'router_hidden_dim': self._create_spinbox(128, 512, ROUTER_HIDDEN_DIM_DEFAULT),
            'max_path_len': self._create_spinbox(1, 16, MAX_PATH_LEN_DEFAULT),
            'max_visits_per_expert': self._create_spinbox(1, 5, MAX_VISITS_PER_EXPERT_DEFAULT),
            'gumbel_tau': self._create_spinbox(0.5, 3.0, GUMBEL_TAU_INIT, 0.1),
            'learning_rate': self._create_spinbox(1e-5, 5e-4, LEARNING_RATE, 1e-5, 5),
            'batch_size': self._create_spinbox(8, 64, BATCH_SIZE),
            'router_entropy_coef': self._create_spinbox(0.0, 0.05, ROUTER_ENTROPY_COEF, 0.001, 4),
            'animation_delay': self._create_spinbox(0.0, 1.0, ANIMATION_DELAY_DEFAULT, 0.05),
            'train_steps_per_eval': self._create_spinbox(5, 50, TRAIN_STEPS_PER_EVAL)
        }
        spin_labels = {"num_experts": "Experts", "embed_dim": "EmbDim", "expert_nhead": "E.NHead",
                       "expert_dim_feedforward": "E.FFDim", "expert_layers": "E.Layers", "router_hidden_dim": "RtrHid",
                       "max_path_len": "MaxPath", "max_visits_per_expert": "MaxVisits", "gumbel_tau": "GumbelTau",
                       "learning_rate": "LR", "batch_size": "BatchSz", "router_entropy_coef": "EntrCoef",
                       "animation_delay": "AnimDelay", "train_steps_per_eval": "EvalSteps"}
        for name, spin in self.hps_spins.items(): hps_form.addRow(f"{spin_labels.get(name, name)}:", spin)
        hps_group.setLayout(hps_form);
        left_panel.addWidget(hps_group)

        actions_group = QGroupBox("System Control");
        actions_layout = QVBoxLayout()
        self.reinit_button = QPushButton("Apply Settings & Reinitialize");
        self.start_learn_button = QPushButton("Start Learning")
        self.stop_learn_button = QPushButton("Stop Learning", enabled=False);
        self.animate_val_button = QPushButton("Animate Validation Sample", enabled=False)
        for btn in [self.reinit_button, self.start_learn_button, self.stop_learn_button,
                    self.animate_val_button]: actions_layout.addWidget(btn)
        actions_group.setLayout(actions_layout);
        left_panel.addWidget(actions_group)
        self.diagram_widget = DiagramWidget(num_experts=self.hps['num_experts']);
        left_panel.addWidget(self.diagram_widget);
        left_panel.addStretch(1)
        top_layout.addLayout(left_panel, stretch=1)

        right_panel = QVBoxLayout();
        self.tabs = QTabWidget()
        plots_tab = QWidget();
        plots_layout = QVBoxLayout(plots_tab)
        if PYQTGRAPH_AVAILABLE:
            pg_config_opts = {'antialias': True, 'background': self.palette().color(self.backgroundRole()).getRgb()[:3],
                              'foreground': self.palette().color(self.foregroundRole()).getRgb()[:3]}
            pg.setConfigOptions(**pg_config_opts)
            plot_configs = [
                ("loss_plot_widget", "Loss Curves",
                 [("train_loss_curve", 'b', "Train Loss"), ("val_loss_curve", 'r', "Val Loss")], "Loss", "Eval Steps"),
                ("perplexity_plot_widget", "Validation Perplexity", [("perplexity_curve", 'g', "Perplexity")],
                 "Perplexity", "Eval Steps"),
                ("aux_metrics_plot_widget", "Auxiliary Metrics",
                 [("path_len_curve", 'c', "Avg.PathLen"), ("router_entropy_curve", 'm', "RouterEntropy")], "Value",
                 "Eval Steps")
            ]
            for widget_name, title, curves_conf, y_label, x_label in plot_configs:
                plot_widget = pg.PlotWidget(title=title);
                setattr(self, widget_name, plot_widget)
                if len(curves_conf) > 1: plot_widget.addLegend(offset=(-10, 10))
                for curve_attr, color, name in curves_conf: setattr(self, curve_attr,
                                                                    plot_widget.plot(pen=pg.mkPen(color, width=2),
                                                                                     name=name))
                plot_widget.setLabel('left', y_label);
                plot_widget.setLabel('bottom', x_label);
                plots_layout.addWidget(plot_widget)
            self.path_viz_widget = GLViewWidget();
            self.path_viz_widget.setCameraPosition(distance=50);
            plots_layout.addWidget(self.path_viz_widget)
        else:
            plots_layout.addWidget(QLabel("pyqtgraph not installed. Visualization disabled."))
        self.tabs.addTab(plots_tab, "Live Metrics")

        stats_tab = QWidget();
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.addWidget(QLabel("Expert Usage Frequency:"))
        if PYQTGRAPH_AVAILABLE:
            self.expert_usage_plot_widget = pg.PlotWidget(); self.expert_usage_bars = None; stats_layout.addWidget(
                self.expert_usage_plot_widget, stretch=1)
        else:
            stats_layout.addWidget(QLabel("pyqtgraph not available for bar chart."))
        stats_layout.addWidget(QLabel("Pathway Frequencies:"));
        self.pathway_stats_text_edit = QTextEdit(readOnly=True);
        stats_layout.addWidget(self.pathway_stats_text_edit, stretch=1)
        self.tabs.addTab(stats_tab, "Analysis")

        log_sample_tab = QWidget();
        log_sample_layout = QVBoxLayout(log_sample_tab)
        log_sample_layout.addWidget(QLabel("System Log:"));
        self.log_text_edit = QTextEdit(readOnly=True);
        self.log_text_edit.setFixedHeight(250);
        log_sample_layout.addWidget(self.log_text_edit)
        log_sample_layout.addWidget(QLabel("Current Sample Info:"));
        self.current_sample_text_edit = QTextEdit(readOnly=True);
        self.current_sample_text_edit.setFixedHeight(150);
        log_sample_layout.addWidget(self.current_sample_text_edit)
        log_sample_layout.addStretch(1);
        self.tabs.addTab(log_sample_tab, "Log & Sample")
        right_panel.addWidget(self.tabs);
        top_layout.addLayout(right_panel, stretch=2)
        self.hps_spins['embed_dim'].valueChanged.connect(self.update_nhead_options);
        self.update_nhead_options()

    def update_nhead_options(self):
        embed_dim = self.hps_spins['embed_dim'].value()
        nhead_spin = self.hps_spins['expert_nhead']
        valid_nheads = [i for i in range(1, nhead_spin.maximum() + 1) if embed_dim % i == 0]
        current_nhead = nhead_spin.value()
        nhead_spin.setRange(min(valid_nheads) if valid_nheads else 1,
                            max(valid_nheads) if valid_nheads else 1)  # Ensure range is valid
        if valid_nheads and current_nhead not in valid_nheads:
            nhead_spin.setValue(valid_nheads[0])
        elif not valid_nheads:
            nhead_spin.setValue(1)

    def _connect_signals(self):
        self.reinit_button.clicked.connect(self.init_system_data_and_model)
        self.start_learn_button.clicked.connect(self.start_continuous_learning)
        self.stop_learn_button.clicked.connect(self.stop_continuous_learning)
        self.animate_val_button.clicked.connect(self.trigger_one_sample_animation)
        asig = self.animation_signals_obj
        asig.signal_input_text.connect(self.on_input_text_anim);
        asig.signal_embedded.connect(self.on_embedded_anim)
        asig.signal_to_router.connect(self.on_to_router_anim);
        asig.signal_router_output.connect(self.on_router_output_anim)
        asig.signal_to_expert.connect(self.on_to_expert_anim);
        asig.signal_expert_output.connect(self.on_expert_output_anim)
        asig.signal_to_output_layer.connect(self.on_to_output_layer_anim);
        asig.signal_final_prediction_lm.connect(self.on_final_prediction_lm_anim)
        csig = self.continuous_learning_signals_obj
        csig.log_message.connect(self.log_message);
        csig.stats_update.connect(self.handle_stats_update);
        csig.learning_stopped.connect(self.on_learning_stopped_ui_update)

    def init_system_data_and_model(self):
        self.log_message("Initializing system...")
        if self.learning_worker and self.learning_worker.isRunning():
            self.log_message("Stopping active learning...");
            self.stop_continuous_learning()
            if not self.learning_worker.wait(3000): self.log_message(
                "Warning: Worker did not stop gracefully."); self.learning_worker.terminate(); self.learning_worker.wait()
            self.learning_worker = None

        self.hps = self._get_current_hps_from_ui()
        if errors := self.validate_hps(self.hps): self.log_message(f"ERROR: Invalid HPs: {'; '.join(errors)}"); return

        all_tokens = tokenize_text(WIKITEXT_SAMPLE)
        for _ in range(3): all_tokens.extend(tokenize_text(augment_text(WIKITEXT_SAMPLE)))  # Restored augmentation

        split_idx = int(len(all_tokens) * TRAIN_SPLIT_RATIO);
        train_tokens, val_tokens = all_tokens[:split_idx], all_tokens[split_idx:]
        self.vocab = Vocabulary(max_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQ)
        self.train_dataset = LanguageModelDataset(train_tokens, self.vocab, self.hps['max_lm_seq_len'], "train")
        self.animation_signals_obj.vocab = self.vocab
        self.val_dataset = LanguageModelDataset(val_tokens, self.vocab, self.hps['max_lm_seq_len'], "val")

        if not self.train_dataset.sequences or not self.val_dataset.sequences: self.log_message(
            "ERROR: Not enough data."); self.start_learn_button.setEnabled(False); return

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hps['batch_size'],
                                                        shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hps['batch_size'],
                                                      shuffle=False, num_workers=2, pin_memory=True)
        self.log_message(f"Data: Train {len(self.train_dataset)}, Val {len(self.val_dataset)} sequences.")

        self.model = GoEModel(
            vocab_size=len(self.vocab), embed_dim=self.hps['embed_dim'], num_experts=self.hps['num_experts'],
            expert_nhead=self.hps['expert_nhead'], expert_dim_feedforward=self.hps['expert_dim_feedforward'],
            expert_layers=self.hps['expert_layers'], router_hidden_dim=self.hps['router_hidden_dim'],
            max_path_len=self.hps['max_path_len'], max_visits_per_expert=self.hps['max_visits_per_expert'],
            max_lm_seq_len=self.hps['max_lm_seq_len'], gumbel_tau_init=self.hps['gumbel_tau']
        )
        self.model.animation_signals_obj = self.animation_signals_obj
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.hps['learning_rate'],
                                     weight_decay=self.hps['weight_decay'])
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.word_to_idx["<pad>"])

        self.log_message(f"Model initialized with {self.hps['num_experts']} experts.");
        self.diagram_widget.update_num_experts(self.hps['num_experts'])
        self.pathway_stats_text_edit.clear();
        self.current_sample_text_edit.clear();
        self.reset_plots()
        self.start_learn_button.setEnabled(True);
        self.stop_learn_button.setEnabled(False);
        self.animate_val_button.setEnabled(False)
        self.continuous_learning_signals_obj.model_initialization_done.emit();
        self.log_message("Initialization complete.")

    def reset_plots(self):
        if not PYQTGRAPH_AVAILABLE: return
        for curve_attr in ["train_loss_curve", "val_loss_curve", "perplexity_curve", "path_len_curve",
                           "router_entropy_curve"]:
            if hasattr(self, curve_attr) and getattr(self, curve_attr): getattr(self, curve_attr).clear()
        if hasattr(self, 'expert_usage_bars') and self.expert_usage_bars: self.expert_usage_plot_widget.removeItem(
            self.expert_usage_bars); self.expert_usage_bars = None
        if hasattr(self, 'path_viz_widget') and self.path_viz_widget: self.path_viz_widget.clear()
        with self.plot_data_cache_lock:
            self.plot_data_cache = None

    def start_continuous_learning(self):
        if not all([self.model, self.train_loader, self.val_loader]): self.log_message(
            "System not initialized."); return
        self.log_message("Starting learning...")
        self.learning_worker = ContinuousLearningWorker(lambda: (self.model, self.optimizer, self.criterion),
                                                        lambda: (self.train_loader, self.val_loader), self.hps,
                                                        self.continuous_learning_signals_obj)
        self.learning_worker.start()
        if PYQTGRAPH_AVAILABLE and hasattr(self,
                                           'plot_timer') and not self.plot_timer.isActive(): self.plot_timer.start(
            PLOT_UPDATE_INTERVAL_MS)
        self.start_learn_button.setEnabled(False);
        self.stop_learn_button.setEnabled(True);
        self.animate_val_button.setEnabled(True);
        self.reinit_button.setEnabled(False)

    def stop_continuous_learning(self):
        if self.learning_worker and self.learning_worker.isRunning(): self.log_message(
            "Stopping learning..."); self.learning_worker.stop_learning()
        if PYQTGRAPH_AVAILABLE and hasattr(self, 'plot_timer') and self.plot_timer.isActive(): self.plot_timer.stop()

    def on_learning_stopped_ui_update(self):
        self.start_learn_button.setEnabled(True);
        self.stop_learn_button.setEnabled(False);
        self.animate_val_button.setEnabled(False);
        self.reinit_button.setEnabled(True)
        self.log_message("Learning stopped.")

    def trigger_one_sample_animation(self):
        if not (self.learning_worker and self.learning_worker.isRunning()): self.log_message(
            "Learning not active."); return
        if not (self.val_dataset and self.val_dataset.sequences): self.log_message(
            "Validation data unavailable."); return
        self.diagram_widget.reset_highlights();
        self.current_sample_text_edit.clear()
        input_ids, target_id, text_snippet = self.val_dataset[random.randint(0, len(self.val_dataset) - 1)]
        self.log_message(f"Animating: {text_snippet[:50]}...");
        self.model.current_hps = self.hps.copy()
        self.learning_worker.request_animation((input_ids, target_id, text_snippet))

    def log_message(self, msg: str):
        self.log_text_edit.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def handle_stats_update(self, stats: Dict[str, Any]):
        with self.plot_data_cache_lock:
            self.plot_data_cache = stats.copy()
        self.pathway_stats_text_edit.clear();
        path_freq = stats.get('pathway_freq', {});
        total_paths = sum(path_freq.values())
        self.pathway_stats_text_edit.append(f"--- Pathway Frequencies (Total: {total_paths}) ---")
        sorted_paths = sorted(path_freq.items(), key=lambda x: x[1], reverse=True)
        for i, (path, count) in enumerate(sorted_paths[:20]): self.pathway_stats_text_edit.append(
            f"{'->'.join(map(str, path))}: {count} ({count / max(1, total_paths):.2%})")
        if len(sorted_paths) > 20: self.pathway_stats_text_edit.append("...")
        self.pathway_stats_text_edit.append("\n--- Compilation Candidates (>5%) ---")
        if not any(
            self.pathway_stats_text_edit.append(f"MERGE: {'->'.join(map(str, p))}({c / max(1, total_paths):.2%})") for
            p, c in sorted_paths if c / max(1, total_paths) > 0.05): self.pathway_stats_text_edit.append("None.")
        self.log_message(
            f"Step {stats.get('global_step', 0)}: VL={stats.get('current_val_loss', float('nan')):.4f}, PPL={stats.get('current_perplexity', float('nan')):.2f}, GN={stats.get('grad_norm', float('nan')):.2f}, R={stats.get('reward_mean', float('nan')):.2f}")
        self.diagram_widget.set_expert_usage_text(stats.get('expert_usage', {}))

        if PYQTGRAPH_AVAILABLE and hasattr(self, 'path_viz_widget') and self.path_viz_widget:
            self.path_viz_widget.clear();
            num_exp = self.hps['num_experts']
            # Node coords: 0=router, 1..N=experts, N+1=output
            nodes_coords = [[0, 0, 0]] + [
                [5 * math.cos(2 * math.pi * i / num_exp), 5 * math.sin(2 * math.pi * i / num_exp), 0] for i in
                range(num_exp)] + [[0, 0, 5]]
            node_pos_np = np.array(nodes_coords, dtype=np.float32)
            self.path_viz_widget.addItem(GLScatterPlotItem(pos=node_pos_np, size=10, color=(1, 1, 0, 1)))

            for path_tuple, count in sorted_paths[:5]:  # path_tuple is like (0, 1, 'T')
                path_node_indices_viz = [0]  # Start at router (node 0)
                for p_element in path_tuple:
                    if isinstance(p_element, int):
                        path_node_indices_viz.append(p_element + 1)  # Experts are nodes 1 to num_exp
                    elif p_element in ('T', 'T_max'):
                        path_node_indices_viz.append(num_exp + 1); break  # Output is node num_exp + 1

                if len(path_node_indices_viz) > 1:
                    edges = np.array([[path_node_indices_viz[i], path_node_indices_viz[i + 1]] for i in
                                      range(len(path_node_indices_viz) - 1)], dtype=np.int32)
                    if edges.size > 0:  # Ensure edges exist
                        self.path_viz_widget.addItem(GLLinePlotItem(pos=node_pos_np[edges].reshape(-1, 3),
                                                                    color=(0, 1, 0, count / max(1, total_paths)),
                                                                    width=2))

    def update_plots_from_cache(self):
        if not PYQTGRAPH_AVAILABLE or not self.plot_data_cache: return
        with self.plot_data_cache_lock:
            stats = self.plot_data_cache.copy()
        x_axis_eval = list(range(len(stats.get('val_loss_hist', []))))  # For eval-step based plots

        if hasattr(self, 'train_loss_curve'): self.train_loss_curve.setData(
            x=list(range(len(stats.get('train_loss_hist', [])))), y=stats.get('train_loss_hist', []))
        if hasattr(self, 'val_loss_curve'): self.val_loss_curve.setData(x=x_axis_eval, y=stats.get('val_loss_hist', []))
        if hasattr(self, 'perplexity_curve'): self.perplexity_curve.setData(x=x_axis_eval,
                                                                            y=stats.get('val_perp_hist', []))
        if hasattr(self, 'path_len_curve'): self.path_len_curve.setData(x=x_axis_eval, y=stats.get('path_len_hist', []))
        if hasattr(self, 'router_entropy_curve'): self.router_entropy_curve.setData(x=x_axis_eval,
                                                                                    y=stats.get('router_entropy_hist',
                                                                                                []))

        if hasattr(self, 'expert_usage_plot_widget'):
            expert_usage = stats.get('expert_usage', {});
            num_exp = self.hps.get('num_experts', NUM_EXPERTS_DEFAULT)
            x_ticks = [(i, f"E{i}") for i in range(num_exp)];
            heights = [expert_usage.get(i, 0) for i in range(num_exp)]
            if self.expert_usage_bars: self.expert_usage_plot_widget.removeItem(self.expert_usage_bars)
            self.expert_usage_bars = pg.BarGraphItem(x=list(range(num_exp)), height=heights, width=0.6, brush='teal')
            self.expert_usage_plot_widget.addItem(self.expert_usage_bars)
            self.expert_usage_plot_widget.getAxis('bottom').setTicks([x_ticks]);
            self.expert_usage_plot_widget.setLabel('left', "Usage");
            self.expert_usage_plot_widget.setTitle("Expert Usage")

    def on_input_text_anim(self, text: str):
        self.diagram_widget.reset_highlights(); self.diagram_widget.highlight_node(
            "INPUT"); self.current_sample_text_edit.setText(f"Input: {text[:100]}...")

    def on_embedded_anim(self, r: str):
        self.diagram_widget.highlight_node("INPUT", False); self.diagram_widget.highlight_connection("INPUT",
                                                                                                     "EMBED"); self.diagram_widget.highlight_node(
            "EMBED"); self.diagram_widget.set_representation_text(r)

    def on_to_router_anim(self, r: str, p: List[Any]):  # p is current path taken so far
        # De-highlight previous node (Embed or an Expert)
        if not p or p[-1] == "EMBED":  # Should not happen, path starts after embed
            self.diagram_widget.highlight_node("EMBED", False)
            self.diagram_widget.highlight_connection("EMBED", "ROUTER")
        elif isinstance(p[-1], int):  # Came from an expert
            self.diagram_widget.highlight_node(f"EXPERT_{p[-1]}", False)
            self.diagram_widget.highlight_connection(f"EXPERT_{p[-1]}", "ROUTER")
        else:  # Default from embed if path is empty (first step)
            self.diagram_widget.highlight_node("EMBED", False)
            self.diagram_widget.highlight_connection("EMBED", "ROUTER")

        self.diagram_widget.highlight_node("ROUTER");
        self.diagram_widget.set_representation_text(r);
        self.diagram_widget.set_current_path_text(p)

    def on_router_output_anim(self, probs: List[float], idx: int, is_term: bool):
        self.diagram_widget.set_router_probs_text(probs);
        self.diagram_widget.highlight_node("ROUTER", False)
        self.diagram_widget.highlight_connection("ROUTER", "LM_HEAD" if is_term else f"EXPERT_{idx}")
        if is_term: self.diagram_widget.highlight_node("LM_HEAD")  # Highlight LM_HEAD if terminating
        # else: expert node will be highlighted by on_to_expert_anim

    def on_to_expert_anim(self, idx: int, r: str):
        self.diagram_widget.highlight_node(f"EXPERT_{idx}"); self.diagram_widget.set_representation_text(r)

    def on_expert_output_anim(self, idx: int, r: str):
        self.diagram_widget.set_representation_text(r)  # Expert node EXPEidxRT_ is already highlighted

    def on_to_output_layer_anim(self, r: str):
        # The LM_HEAD should have been highlighted by on_router_output if termination occurred there.
        # This signal confirms the final representation is being fed to LM_HEAD.
        self.diagram_widget.highlight_node("LM_HEAD", True)
        self.diagram_widget.set_representation_text(r)

    def on_final_prediction_lm_anim(self, pred_tok: str, path: List[Any]):
        self.diagram_widget.highlight_node("LM_HEAD", True)  # Keep LM_HEAD highlighted
        self.diagram_widget.set_current_path_text(path)
        self.current_sample_text_edit.setText(
            f"{self.current_sample_text_edit.toPlainText()}\nPred: '{pred_tok}', Path: {'->'.join(map(str, path))}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if not PYQTGRAPH_AVAILABLE: print("ERROR: pyqtgraph is required for visualization."); sys.exit(1)
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not found."); sys.exit(1)
    if torch.cuda.is_available(): print("CUDA available. Forcing CPU for this run (as per original setup).")
    torch.set_default_device('cpu')  # Ensure CPU for consistency with original problem context
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())