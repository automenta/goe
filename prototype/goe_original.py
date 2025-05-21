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
    QTextEdit, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QTimer

import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLScatterPlotItem

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print(
        "WARNING: pyqtgraph not found. Install with 'pip install pyqtgraph pyopengl'. OpenGL support is recommended for 3D view.")

# --- Global Device Configuration ---
if 'torch' in sys.modules:  # Ensure torch is imported before this
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
else:
    print("Error: PyTorch not imported yet. Device configuration will fail.")
    DEVICE = torch.device("cpu")  # Fallback

# --- Constants ---
MAX_VOCAB_SIZE = 10000
MIN_FREQ = 1
MAX_LM_SEQ_LEN = 32  # Max sequence length for creating LM examples
TRAIN_SPLIT_RATIO = 0.8
EMBED_DIM_DEFAULT = 256
NUM_EXPERTS_DEFAULT = 4
EXPERT_NHEAD_DEFAULT = 4
EXPERT_DIM_FEEDFORWARD_DEFAULT = 1024
EXPERT_LAYERS_DEFAULT = 4  # Number of transformer blocks per expert
ROUTER_HIDDEN_DIM_DEFAULT = 256
MAX_PATH_LEN_DEFAULT = 3  # Max number of experts in a path
MAX_VISITS_PER_EXPERT_DEFAULT = 2  # Max times one expert can be visited in a single path
GUMBEL_TAU_INIT = 2.0  # Initial Gumbel-Softmax temperature
DIVERSITY_LOSS_COEF = 0.02
Q_LOSS_COEF = 0.01
CONTRASTIVE_LOSS_COEF = 0.05
LEARNING_RATE = 3e-4
BATCH_SIZE = 32  # Actual batch size for one optimizer step
EFFECTIVE_BATCH_SIZE = 64  # Accumulate gradients for this many samples
ROUTER_ENTROPY_COEF = 0.03
GRADIENT_CLIP_NORM = 0.3
WEIGHT_DECAY = 1e-6
TRAIN_STEPS_PER_EVAL = 10  # How often to run validation
SMOOTHING_WINDOW_LOSS = 5  # For plotting smoothed training loss
WARMUP_STEPS = 200  # LR scheduler warmup
TOTAL_STEPS = 2000  # Total training steps for one run
RESTART_PERIOD = 500  # For CosineAnnealingWarmRestarts scheduler
ANIMATION_DELAY_DEFAULT = 0.2  # Seconds
PLOT_UPDATE_INTERVAL_MS = 300  # Refresh rate for plots
MAX_PLOT_POINTS = 500  # Max data points for live plots
DIAGRAM_NODE_WIDTH = 120
DIAGRAM_NODE_HEIGHT = 40
DIAGRAM_ARROW_SIZE = 10.0
LR_FACTOR = 0.999  # 1 to disable

# New constants for enhancements
PATH_REWARD_PENALTY_COEF_DEFAULT = 0.005
ROUTER_LR_MULT_DEFAULT = 0.5  # Router might need smaller LR
EXPERT_LR_MULT_DEFAULT = 1.0  # Experts at base LR

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


# --- Utility Functions ---
def augment_text(text: str) -> str:
    tokens = text.lower().split()
    synonyms = {"game": "title", "tactical": "strategic", "unit": "team", "mission": "operation", "battle": "conflict",
                "army": "force", "play": "engage", "story": "narrative"}
    augmented_tokens = []
    for t in tokens:
        r_syn = random.random()
        r_suf = random.random()
        if r_syn < 0.15 and t in synonyms:
            augmented_tokens.append(synonyms[t])
        elif r_suf < 0.05:  # Minor morphological changes
            augmented_tokens.append(t + random.choice(["", " ", "s"]))  # keep some original form
        else:
            augmented_tokens.append(t)
    return " ".join(augmented_tokens).replace("play", "engage").replace("game", "title")  # Global replacements last


def tokenize_text(text: str) -> List[str]: return text.lower().split()


def create_lm_sequences(token_ids: List[int], seq_len: int) -> List[Tuple[List[int], int]]:
    if not token_ids or len(token_ids) <= seq_len: return []
    return [(token_ids[i:i + seq_len], token_ids[i + seq_len]) for i in range(len(token_ids) - seq_len)]


def repr_tensor(tensor_data: Any, num_elems: int = 3) -> str:
    if tensor_data is None: return "None"
    if isinstance(tensor_data, list):
        return f"[{', '.join(f'{x:.2f}' for x in tensor_data[:num_elems])}{'...' if len(tensor_data) > num_elems else ''}]"
    if not isinstance(tensor_data, torch.Tensor) or tensor_data.numel() == 0:
        return str(tensor_data)
    items = tensor_data.detach().cpu().flatten().tolist()
    return f"[{', '.join(f'{x:.2f}' for x in items[:num_elems])}{f'...L{len(items)}' if len(items) > num_elems else ']'}]"


# --- Data Handling ---
class Vocabulary:
    def __init__(self, max_size: Optional[int] = None, min_freq: int = 1):
        self.word_to_idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.word_counts = collections.Counter()
        self.max_size = max_size
        self.min_freq = min_freq
        self._finalized = False

    def add_word(self, word: str):
        if not self._finalized:
            self.word_counts[word] += 1
        else:
            warnings.warn("Cannot add word to finalized vocabulary.")

    def build_vocab(self):
        if self._finalized and len(self.word_to_idx) > 4: return  # Already built with actual words
        current_special_tokens = self.word_to_idx.copy()
        self.word_to_idx = current_special_tokens
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        eligible_words = [w for w, c in self.word_counts.most_common() if c >= self.min_freq]
        for word in eligible_words:
            if self.max_size and len(self.word_to_idx) >= self.max_size: break
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        self._finalized = True
        print(f"Vocabulary built with {len(self.word_to_idx)} words.")

    def __len__(self) -> int:
        return len(self.word_to_idx)

    def numericalize(self, tokens: List[str]) -> List[int]:
        if not self._finalized: self.build_vocab()
        return [self.word_to_idx.get(t, self.word_to_idx["<unk>"]) for t in tokens]

    def denumericalize(self, indices: List[int]) -> List[str]:
        return [self.idx_to_word.get(i, "<unk>") for i in indices]


class LanguageModelDataset(torch.utils.data.Dataset):
    def __init__(self, text_tokens: List[str], vocab: Vocabulary, seq_len: int, name: str = "train"):
        if not text_tokens: raise ValueError(f"{name} dataset received empty token list.")
        self.vocab, self.seq_len = vocab, seq_len
        if name == "train" and not self.vocab._finalized:
            for token in text_tokens: self.vocab.add_word(token)
            self.vocab.build_vocab()
        self.token_ids = self.vocab.numericalize(text_tokens)
        self.sequences = create_lm_sequences(self.token_ids, self.seq_len)
        if not self.sequences:
            warnings.warn(
                f"No sequences created for {name} dataset (len_tokens={len(self.token_ids)}, seq_len={seq_len})")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        in_ids, tgt_id = self.sequences[idx]
        try:
            in_txt = " ".join(self.vocab.denumericalize(in_ids))
            tgt_txt = self.vocab.denumericalize([tgt_id])[0]
            snippet = f"{in_txt} ...predict... {tgt_txt}"
        except Exception as e:
            snippet = f"Error creating snippet: {e}"
        return torch.tensor(in_ids, dtype=torch.long), torch.tensor(tgt_id, dtype=torch.long), snippet


# --- Model Components ---
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

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.dropout(x + self.pe[:, :x.size(1)])


class GatedAttention(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True, dropout=dropout)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
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
        self.tag = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.01)

        # Enhancement 1: Adaptive Gate
        self.gate_linear = nn.Linear(embed_dim, 1)  # Input: mean of processed output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x  # Save original input for skip connection in adaptive gate

        # Core expert processing
        for mod in self.layers:
            x = x + mod['dropout'](mod['attn'](mod['norm1'](x)))
            x = x + mod['dropout'](mod['ffn'](mod['norm2'](x)))

        # Enhancement 1: Apply adaptive gate
        # Gate value based on the mean of the processed sequence features
        # Squeeze and unsqueeze to make gate_val broadcastable: (batch_size, 1, 1)
        gate_val = torch.sigmoid(self.gate_linear(x.mean(dim=1, keepdim=True)))

        # Modulate output: gate * processed_output + (1-gate) * original_input
        x_gated = gate_val * x + (1 - gate_val) * x_input

        return x_gated + self.tag  # Add learnable tag after gating


class RoutingController(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts + 1)  # +1 for termination action
        self.norm = nn.LayerNorm(num_experts + 1)
        self.q_values = nn.Parameter(torch.zeros(num_experts + 1))

    def forward(self, x_summary: torch.Tensor, visit_counts: torch.Tensor, max_visits: int) -> torch.Tensor:
        x = F.gelu(self.fc1(x_summary))
        logits = self.fc2(x)

        # Masking based on visit counts
        mask = visit_counts >= max_visits
        # Termination action (last logit) is not masked by visit counts
        extended_mask = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=-1)

        q_values_on_device = self.q_values.unsqueeze(0).to(logits.device)
        final_logits = self.norm(logits + q_values_on_device).clamp(min=-10.0, max=10.0)
        return final_logits.masked_fill(extended_mask, -float('inf'))


class RunningStat:
    def __init__(self):
        self.mean, self.m2, self.count = 0.0, 0.0, 0

    def update(self, x_tensor: torch.Tensor):
        if x_tensor.numel() == 0: return
        for x in x_tensor.flatten().tolist():
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
        return (x - self.mean) / (self.std + 1e-8)


# --- Main Model ---
class GoEModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_experts: int, expert_nhead: int,
                 expert_dim_feedforward: int, expert_layers: int, router_hidden_dim: int, max_path_len: int,
                 max_visits_per_expert: int, max_lm_seq_len_param: int,
                 gumbel_tau_init: float, path_reward_penalty_coef: float):  # Added path_reward_penalty_coef
        super().__init__()
        self.vocab_size, self.embed_dim, self.num_experts = vocab_size, embed_dim, num_experts
        self.max_path_len, self.max_visits_per_expert = max_path_len, max_visits_per_expert
        self.gumbel_tau_init = gumbel_tau_init
        self.path_reward_penalty_coef = path_reward_penalty_coef  # Store new HP

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.1, max_len=max(512, max_lm_seq_len_param * 2))
        self.experts = nn.ModuleList(
            [Expert(embed_dim, expert_nhead, expert_dim_feedforward, expert_layers, dropout=0.1) for _ in
             range(num_experts)])

        # Enhancement 2: Router input dimension changed for mean + max pooling
        router_input_dim = embed_dim * 2
        self.router = RoutingController(router_input_dim, router_hidden_dim, num_experts)

        self.output_lm_head = nn.Linear(embed_dim, vocab_size)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.path_optimizer = nn.Linear(embed_dim, self.max_path_len + 1)
        self.meta_controller = nn.Linear(embed_dim, 2)
        self.reward_stat = RunningStat()
        self.animation_signals_obj, self.is_animating_sample, self.current_hps = None, False, {}

    @property
    def model_device(self) -> torch.device:
        return self.embedding.weight.device

    def _emit_signal(self, signal_type: str, *args):
        if self.is_animating_sample and self.animation_signals_obj:
            delay = self.current_hps.get('animation_delay', 0.1)
            getattr(self.animation_signals_obj, signal_type).emit(*args)
            if delay > 0: QThread.msleep(int(delay * 1000))

    def _compute_diversity_loss(self, router_probs: torch.Tensor, expert_usage: Dict[int, int]) -> torch.Tensor:
        if router_probs.numel() == 0: return torch.tensor(0.0, device=self.model_device)
        expert_probs = router_probs[:, :self.num_experts]
        avg_probs = expert_probs.mean(dim=0)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        total_usage = sum(expert_usage.values()) + 1e-10
        usage_prop = torch.tensor([expert_usage.get(i, 0) / total_usage for i in range(self.num_experts)],
                                  device=router_probs.device)
        load_balance_var = torch.var(usage_prop)
        return -entropy + 0.1 * load_balance_var

    def _compute_q_loss(self, logits: torch.Tensor, rewards: torch.Tensor, discount: float) -> torch.Tensor:
        q_router = self.router.q_values.to(logits.device)
        actions = logits.argmax(dim=-1)
        q_selected = q_router[actions]
        q_target = rewards + discount * q_router.max().detach().expand_as(rewards)
        return F.mse_loss(q_selected, q_target.clamp(-1.0, 1.0))

    def _compute_contrastive_loss(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        model_dev = self.model_device
        if len(outputs) < 2: return torch.tensor(0.0, device=model_dev)
        means = torch.stack([o.mean(dim=1).squeeze(0) for o in outputs])
        if means.size(0) < 2: return torch.tensor(0.0, device=model_dev)

        normed = F.normalize(means, p=2, dim=1)
        sim = torch.matmul(normed, normed.t())
        target = torch.eye(means.size(0), device=sim.device)
        return (sim - target).pow(2).mean()

    def forward(self, input_ids: torch.Tensor, current_sample_text: Optional[str] = None,
                target_ids: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, List[List[Any]], torch.Tensor]:

        bsz, seq_len = input_ids.shape
        current_device = input_ids.device

        if self.is_animating_sample and current_sample_text: self._emit_signal("signal_input_text", current_sample_text)

        x = self.input_norm(self.pos_encoder(self.embedding(input_ids) * math.sqrt(self.embed_dim)))
        if self.is_animating_sample: self._emit_signal("signal_embedded", repr_tensor(x[0].mean(dim=0)))

        total_ent, total_div, total_q, total_con = [torch.tensor(0.0, device=current_device) for _ in range(4)]
        paths: List[List[Any]] = [[] for _ in range(bsz)]
        visits = torch.zeros((bsz, self.num_experts), dtype=torch.long, device=current_device)
        final_repr = x.clone()
        active_orig_idx = torch.arange(bsz, device=current_device)
        active_x = x
        expert_outputs: List[torch.Tensor] = []
        expert_usage_batch: Dict[int, int] = collections.Counter()

        step = self.current_hps.get('global_step', 0)
        total_steps = max(1, self.current_hps.get('total_steps', TOTAL_STEPS))  # Use HP for total_steps if available
        tau = max(self.gumbel_tau_init * (1 - step / total_steps), 0.1)
        discount = 0.9 + 0.09 * (step / total_steps)

        step_logs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for _ in range(self.max_path_len):
            if not active_orig_idx.numel(): break

            # Enhancement 2: Improved summary for router
            mean_summary = active_x.mean(dim=1)
            max_summary, _ = active_x.max(dim=1)  # Max over sequence length
            summary = torch.cat((mean_summary, max_summary), dim=-1)

            anim_idx_0 = 0
            is_anim = self.is_animating_sample and (anim_idx_0 in active_orig_idx.tolist())
            if is_anim:
                anim_curr_idx = (active_orig_idx == anim_idx_0).nonzero(as_tuple=True)[0].item()
                self._emit_signal("signal_to_router", repr_tensor(summary[anim_curr_idx]),
                                  [str(p) for p in paths[anim_idx_0]])

            rtr_logits = self.router(summary, visits[active_orig_idx], self.max_visits_per_expert)
            rtr_probs = F.softmax(rtr_logits, dim=1)
            total_ent += -torch.sum(rtr_probs * torch.log(rtr_probs + 1e-10), dim=1).mean()
            total_div += self._compute_diversity_loss(rtr_probs, expert_usage_batch)

            hard = self.training
            decision = F.gumbel_softmax(rtr_logits, tau=tau, hard=hard) if hard else \
                F.one_hot(rtr_logits.argmax(dim=1), self.num_experts + 1).float()

            step_logs.append((rtr_logits.clone(), active_orig_idx.clone()))

            term_decision = decision[:, self.num_experts]
            expert_sel = decision[:, :self.num_experts]

            if is_anim:
                action = decision[anim_curr_idx].argmax().item()
                term = action == self.num_experts
                self._emit_signal("signal_router_output", rtr_probs[anim_curr_idx].tolist(), action, term)

            term_mask = (term_decision == 1)
            if term_mask.any():
                term_orig_indices = active_orig_idx[term_mask]
                final_repr[term_orig_indices] = active_x[term_mask]
                for i in term_orig_indices.tolist(): paths[i].append('T')

            expert_mask = (term_decision == 0)
            if not expert_mask.any(): active_orig_idx = torch.tensor([], dtype=torch.long, device=current_device); break

            active_orig_idx = active_orig_idx[expert_mask]
            active_x = active_x[expert_mask]
            chosen_expert_idx = expert_sel[expert_mask].argmax(dim=1)
            next_active_x = torch.zeros_like(active_x)

            for i, orig_idx_val in enumerate(active_orig_idx.tolist()):
                exp_idx = chosen_expert_idx[i].item()
                paths[orig_idx_val].append(exp_idx)
                visits[orig_idx_val, exp_idx] += 1
                expert_usage_batch[exp_idx] += 1

                exp_in = active_x[i].unsqueeze(0)
                is_anim_exp = self.is_animating_sample and (orig_idx_val == anim_idx_0)
                if is_anim_exp: self._emit_signal("signal_to_expert", exp_idx, repr_tensor(exp_in.mean(dim=(0, 1))))

                exp_out = self.experts[exp_idx](exp_in)
                next_active_x[i] = exp_out.squeeze(0)
                expert_outputs.append(exp_out)
                if is_anim_exp: self._emit_signal("signal_expert_output", exp_idx,
                                                  repr_tensor(next_active_x[i].mean(dim=0)))
            active_x = next_active_x

        if active_orig_idx.numel():
            final_repr[active_orig_idx] = active_x
            for i in active_orig_idx.tolist(): paths[i].append('T_max')

        final_summary = self.final_norm(final_repr.mean(dim=1))
        logits = self.output_lm_head(final_summary)
        path_loss, meta_loss = torch.tensor(0.0, device=current_device), torch.tensor(0.0, device=current_device)

        if self.training and target_ids is not None:
            path_pred = self.path_optimizer(final_summary.detach())
            path_len_tgt = torch.tensor([len([s for s in p if isinstance(s, int)]) for p in paths],
                                        device=current_device, dtype=torch.long).clamp(0, self.max_path_len)
            path_loss = F.cross_entropy(path_pred, path_len_tgt)

            meta_pred = self.meta_controller(final_summary.detach())
            unique_exp = torch.tensor([len(set(s for s in p if isinstance(s, int))) for p in paths],
                                      device=current_device, dtype=torch.float)
            meta_tgt = torch.stack([unique_exp, path_len_tgt.float()], dim=1)
            if meta_pred.shape == meta_tgt.shape: meta_loss = F.mse_loss(meta_pred, meta_tgt)

            total_con = self._compute_contrastive_loss(expert_outputs)

            lm_rewards = -F.cross_entropy(logits.detach(), target_ids, reduction='none')

            # Enhancement 3: Path length penalty in reward
            path_lengths_tensor = torch.tensor([len([s for s in p if isinstance(s, int)]) for p in paths],
                                               device=current_device, dtype=torch.float)
            final_rewards = lm_rewards - self.path_reward_penalty_coef * path_lengths_tensor

            self.reward_stat.update(final_rewards.cpu())
            rewards_norm = self.reward_stat.normalize(final_rewards).clamp(-1.0, 1.0)

            for rtr_lgts, act_idx in step_logs:
                if act_idx.numel() > 0:
                    total_q += self._compute_q_loss(rtr_lgts, rewards_norm[act_idx].to(rtr_lgts.device), discount)

        if self.is_animating_sample:
            self._emit_signal("signal_to_output_layer", repr_tensor(final_summary[anim_idx_0]))
            pred_idx = logits[anim_idx_0].argmax().item()
            vocab = self.animation_signals_obj.vocab
            pred_tok = vocab.denumericalize([pred_idx])[0] if vocab else "UNK"
            self._emit_signal("signal_final_prediction_lm", pred_tok, paths[anim_idx_0])

        aux_losses_sum = (DIVERSITY_LOSS_COEF * total_div + Q_LOSS_COEF * total_q +
                          CONTRASTIVE_LOSS_COEF * total_con + 0.1 * path_loss + 0.01 * meta_loss)
        return logits, total_ent, paths, aux_losses_sum


# --- GUI Components ---
class DiagramWidget(QWidget):
    def __init__(self, num_experts: int = NUM_EXPERTS_DEFAULT):
        super().__init__()
        self.setMinimumSize(350, 250)
        self.node_positions, self.active_elements, self.connections_to_draw = {}, {}, []
        self.current_representation_text = self.router_probs_text = self.current_path_text = ""
        self.expert_usage_text = self.stability_text = ""
        self.num_experts = num_experts
        self.node_size = QRectF(0, 0, DIAGRAM_NODE_WIDTH, DIAGRAM_NODE_HEIGHT)
        self._setup_node_positions()

    def update_num_experts(self, num_experts: int):
        if num_experts < 1: raise ValueError("Num experts must be >= 1.")
        self.num_experts = num_experts
        self._setup_node_positions()
        self.reset_highlights()

    def _setup_node_positions(self):
        W, H = self.width() or 400, self.height() or 300
        self.node_positions = {"INPUT": QPointF(W / 2, H * 0.08), "EMBED": QPointF(W / 2, H * 0.22),
                               "ROUTER": QPointF(W / 2, H * 0.40)}
        for i in range(self.num_experts):
            angle = (2 * math.pi / max(1, self.num_experts) * i) - (math.pi / 2)
            radius_x = W * 0.35
            radius_y = H * 0.15
            x = W / 2 + radius_x * math.cos(angle)
            y = H * 0.60 + radius_y * math.sin(angle)
            self.node_positions[f"EXPERT_{i}"] = QPointF(x, y)
        self.node_positions["LM_HEAD"] = QPointF(W / 2, H * 0.85)
        self.base_connections = [("INPUT", "EMBED"), ("EMBED", "ROUTER")] + \
                                [(f"ROUTER", f"EXPERT_{i}") for i in range(self.num_experts)] + \
                                [(f"EXPERT_{i}", "ROUTER") for i in range(self.num_experts)] + \
                                [("ROUTER", "LM_HEAD")]
        self.reset_highlights()

    def resizeEvent(self, event):
        self._setup_node_positions();
        super().resizeEvent(event)

    def reset_highlights(self):
        self.active_elements.clear()
        self.connections_to_draw = [(s, d, QColor("lightgray"), False) for s, d in self.base_connections]
        self.current_representation_text = self.router_probs_text = self.current_path_text = ""
        self.expert_usage_text = self.stability_text = ""
        self.update()

    def highlight_node(self, name: str, active: bool = True):
        if name in self.node_positions:
            self.active_elements[name] = QColor("yellow") if active else QColor("lightblue")
            self.update()

    def highlight_connection(self, from_n: str, to_n: str, active: bool = True):
        if from_n not in self.node_positions or to_n not in self.node_positions: return
        updated = False
        for i, (s, d, _, _) in enumerate(self.connections_to_draw):
            if s == from_n and d == to_n:
                self.connections_to_draw[i] = (s, d, QColor("orange") if active else QColor("lightgray"), active)
                updated = True;
                break
        if not updated:
            self.connections_to_draw.append((from_n, to_n, QColor("orange") if active else QColor("lightgray"), active))

        if active:
            for i, (s, d, col, is_act) in enumerate(self.connections_to_draw):
                if s == from_n and d != to_n and is_act:
                    self.connections_to_draw[i] = (s, d, QColor("lightgray"), False)
        self.update()

    def set_representation_text(self, t: str):
        self.current_representation_text = f"Repr(mean): {t}"; self.update()

    def set_router_probs_text(self, p: List[float]):
        if not p: self.router_probs_text = ""; return
        lbls = ",".join([f"E{i}" for i in range(self.num_experts)]) + ",T"
        self.router_probs_text = f"RtrPrbs({lbls}): [{','.join(f'{x:.2f}' for x in p)}]";
        self.update()

    def set_current_path_text(self, p: List[Any]):
        self.current_path_text = "Path: " + "->".join(map(str, p)); self.update()

    def set_expert_usage_text(self, usage: Dict[int, int]):
        tot = sum(usage.values()) + 1e-10
        usage_str = ", ".join(f'E{i}:{usage.get(i, 0) / tot:.1%}' for i in range(self.num_experts))
        self.expert_usage_text = f"Expert Usage: {usage_str}"
        self.stability_text = "WARN: Routing Collapse!" if tot > 0 and any(
            c / tot > 0.8 for c in usage.values()) else "Routing OK"
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for s, d, col, act in self.connections_to_draw:
            if s in self.node_positions and d in self.node_positions:
                p1, p2 = self.node_positions[s], self.node_positions[d]
                painter.setPen(QPen(col, 3.0 if act else 1.5))
                painter.drawLine(p1, p2)
                angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
                arr_sz = DIAGRAM_ARROW_SIZE
                p_arr1 = QPointF(p2.x() - arr_sz * math.cos(angle - math.pi / 7),
                                 p2.y() - arr_sz * math.sin(angle - math.pi / 7))
                p_arr2 = QPointF(p2.x() - arr_sz * math.cos(angle + math.pi / 7),
                                 p2.y() - arr_sz * math.sin(angle + math.pi / 7))
                painter.setBrush(col);
                painter.drawPolygon(QPolygonF([p2, p_arr1, p_arr2]))

        font = QFont("Arial", 8);
        painter.setFont(font)
        for name, pos in self.node_positions.items():
            r = QRectF(pos.x() - self.node_size.width() / 2, pos.y() - self.node_size.height() / 2,
                       self.node_size.width(), self.node_size.height())
            painter.setBrush(self.active_elements.get(name, QColor("lightblue")));
            painter.setPen(Qt.black)
            painter.drawRoundedRect(r, 5, 5)
            disp_name = name.replace("_", " ").replace("EXPERT", "E").replace("EMBED", "EMBED+POS")
            painter.drawText(r, Qt.AlignCenter, disp_name)

        info_font = QFont("Arial", 9);
        painter.setFont(info_font)
        info_texts = [self.current_representation_text, self.router_probs_text, self.current_path_text,
                      self.expert_usage_text, self.stability_text]
        for i, txt in enumerate(info_texts):
            painter.setPen(QColor("red") if "WARN" in txt and i == len(info_texts) - 1 else Qt.black)
            painter.drawText(QPointF(5, 15 + i * 20), txt)


class AnimationSignals(QWidget):
    signal_input_text = pyqtSignal(str);
    signal_embedded = pyqtSignal(str)
    signal_to_router = pyqtSignal(str, list);
    signal_router_output = pyqtSignal(list, int, bool)
    signal_to_expert = pyqtSignal(int, str);
    signal_expert_output = pyqtSignal(int, str)
    signal_to_output_layer = pyqtSignal(str);
    signal_final_prediction_lm = pyqtSignal(str, list)

    def __init__(self, parent: Optional[QWidget] = None): super().__init__(parent); self.vocab: Optional[
        Vocabulary] = None


class ContinuousLearningSignals(QWidget):
    log_message = pyqtSignal(str);
    stats_update = pyqtSignal(dict)
    model_initialization_done = pyqtSignal();
    learning_stopped = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None): super().__init__(parent)


# --- Training Worker ---
class ContinuousLearningWorker(QThread):
    def __init__(self, model_provider_fn: callable, data_provider_fn: callable, hps: Dict[str, Any],
                 signals_obj: ContinuousLearningSignals, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model_provider, self.data_provider = model_provider_fn, data_provider_fn
        self.hps = hps.copy()  # Worker gets its own copy of HPs
        self.signals = signals_obj
        self._is_running, self._pause_anim, self._anim_info = True, False, None

        self.train_loss_hist = collections.deque(maxlen=MAX_PLOT_POINTS)
        self.val_loss_hist = collections.deque(maxlen=MAX_PLOT_POINTS)
        self.val_ppl_hist = collections.deque(maxlen=MAX_PLOT_POINTS)
        self.path_len_hist = collections.deque(maxlen=MAX_PLOT_POINTS)
        self.router_ent_hist = collections.deque(maxlen=MAX_PLOT_POINTS)
        self.reward_hist = collections.deque(maxlen=MAX_PLOT_POINTS)
        self.step, self.path_freq, self.expert_usage = 0, collections.Counter(), collections.Counter()
        self.grad_accum_steps = max(1, EFFECTIVE_BATCH_SIZE // self.hps['batch_size'])
        self.initial_lrs = []  # For dynamic LR factor

    def run(self):
        self.signals.log_message.emit(f"Worker started on device: {DEVICE}.")
        model, optimizer, criterion = self.model_provider()

        # Store initial LRs for dynamic adjustment (Enhancement 4)
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]

        train_loader, val_loader = self.data_provider()

        if not all([model, optimizer, criterion, train_loader, val_loader]):
            self.signals.log_message.emit("ERROR: Missing components for training worker.")
            self._is_running = False;
            self.signals.learning_stopped.emit();
            return

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=RESTART_PERIOD, T_mult=1, eta_min=1e-7)
        train_iter = iter(train_loader)
        recent_losses = collections.deque(maxlen=SMOOTHING_WINDOW_LOSS)
        accum_grads_count = 0
        current_lr_factor = 1.0
        grad_norm_val = 0.0

        # Pass total_steps to model HPs if needed for annealing schedules inside model
        model.current_hps = self.hps.copy()
        model.current_hps['total_steps'] = self.hps.get('total_steps', TOTAL_STEPS)

        while self.step < self.hps.get('total_steps', TOTAL_STEPS) and self._is_running:
            model.current_hps['global_step'] = self.step

            if self._pause_anim and self._anim_info:
                self.signals.log_message.emit("Pausing for animation...")
                model.is_animating_sample = True
                ids_anim, _, txt_anim = self._anim_info
                with torch.no_grad(): model(ids_anim.unsqueeze(0).to(DEVICE), current_sample_text=txt_anim)
                model.is_animating_sample = False
                self._anim_info = None;
                self._pause_anim = False
                self.signals.log_message.emit("Animation finished.");
                QThread.msleep(50)

            try:
                in_ids, tgt_ids, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader); in_ids, tgt_ids, _ = next(train_iter)
            in_ids, tgt_ids = in_ids.to(DEVICE), tgt_ids.to(DEVICE)

            model.train()
            ent_coef_val = max(
                self.hps['router_entropy_coef'] * (1 - self.step / max(1, self.hps.get('total_steps', TOTAL_STEPS))),
                0.001)

            logits, router_ent_val, paths_taken, aux_loss_val = model(in_ids, target_ids=tgt_ids)
            task_loss_val = criterion(logits, tgt_ids) if logits.numel() > 0 and tgt_ids.numel() > 0 else torch.tensor(
                0.0, device=DEVICE)
            total_loss = (task_loss_val - ent_coef_val * router_ent_val + aux_loss_val) / self.grad_accum_steps
            total_loss.backward()
            accum_grads_count += 1

            if accum_grads_count >= self.grad_accum_steps:
                clip_val = self.hps.get('gradient_clip_norm', GRADIENT_CLIP_NORM)
                grad_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val).item()

                # Enhancement 4: Dynamic LR adjustment respecting initial group LRs
                if grad_norm_val > 2.5 * clip_val and current_lr_factor > 0.1:
                    current_lr_factor = max(0.1, current_lr_factor * LR_FACTOR)
                    self.signals.log_message.emit(
                        f"High grad norm: {grad_norm_val:.2f}, LR factor: {current_lr_factor:.2f}")

                for i, group in enumerate(optimizer.param_groups):
                    group['lr'] = self.initial_lrs[i] * current_lr_factor

                optimizer.step();
                optimizer.zero_grad()
                if self.step >= WARMUP_STEPS: scheduler.step()
                accum_grads_count = 0

            recent_losses.append(task_loss_val.item())
            if len(recent_losses) == SMOOTHING_WINDOW_LOSS: self.train_loss_hist.append(
                sum(recent_losses) / SMOOTHING_WINDOW_LOSS)
            for p_item in paths_taken: self.path_freq[tuple(p_item)] += 1; self.expert_usage.update(
                el for el in p_item if isinstance(el, int))
            if logits.numel() > 0 and tgt_ids.numel() > 0:
                self.reward_hist.append(-F.cross_entropy(logits.detach(), tgt_ids,
                                                         reduction='none').mean().item())  # Log original LM reward before penalty

            self.step += 1
            if self.step % self.hps['train_steps_per_eval'] == 0:
                model.eval()
                val_loss_accum, n_val_samples = 0.0, 0.0
                path_lens_eval, router_ents_eval = [], []
                with torch.no_grad():
                    for val_in_ids, val_tgt_ids, _ in val_loader:
                        val_in_ids, val_tgt_ids = val_in_ids.to(DEVICE), val_tgt_ids.to(DEVICE)
                        val_logits, val_ent_i, val_paths_i, _ = model(val_in_ids, target_ids=val_tgt_ids)
                        if val_logits.numel() > 0 and val_tgt_ids.numel() > 0:
                            val_loss_accum += criterion(val_logits, val_tgt_ids).item() * val_in_ids.size(0)
                            n_val_samples += val_in_ids.size(0)
                        path_lens_eval.extend(
                            len([el for el in p_i if el not in ('T', 'T_max')]) for p_i in val_paths_i)
                        router_ents_eval.append(val_ent_i.item())

                avg_val_loss = val_loss_accum / n_val_samples if n_val_samples > 0 else float('inf')
                val_ppl = math.exp(min(avg_val_loss, 700))
                self.val_loss_hist.append(avg_val_loss);
                self.val_ppl_hist.append(val_ppl)
                self.path_len_hist.append(sum(path_lens_eval) / len(path_lens_eval) if path_lens_eval else 0.0)
                self.router_ent_hist.append(sum(router_ents_eval) / len(router_ents_eval) if router_ents_eval else 0.0)

                self.signals.stats_update.emit({
                    'global_step': self.step, 'train_loss_hist': list(self.train_loss_hist),
                    'val_loss_hist': list(self.val_loss_hist), 'val_perp_hist': list(self.val_ppl_hist),
                    'path_len_hist': list(self.path_len_hist), 'router_entropy_hist': list(self.router_ent_hist),
                    'pathway_freq': dict(self.path_freq), 'expert_usage': dict(self.expert_usage),
                    'current_val_loss': avg_val_loss, 'current_perplexity': val_ppl, 'grad_norm': grad_norm_val,
                    'reward_mean': sum(self.reward_hist) / len(self.reward_hist) if self.reward_hist else 0.0,
                    'current_lr_factor': current_lr_factor
                })
            QThread.msleep(1)

        self.signals.log_message.emit("Worker finished.")
        self.signals.learning_stopped.emit()

    def stop_learning(self):
        self._is_running = False

    def request_animation(self, info: Tuple[torch.Tensor, torch.Tensor, str]):
        self._anim_info, self._pause_anim = info, True


# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph-of-Experts LM (Enhanced)")
        self.setGeometry(30, 30, 1400, 900)
        self.hps = self._get_default_hps()
        self.vocab: Optional[Vocabulary] = None
        self.train_ds, self.val_ds, self.train_dl, self.val_dl = None, None, None, None
        self.model: Optional[GoEModel] = None
        self.optim, self.crit = None, None
        self.worker: Optional[ContinuousLearningWorker] = None
        self.plot_cache_lock = Lock()
        self.plot_cache: Optional[Dict[str, Any]] = None

        self.anim_signals = AnimationSignals()
        self.learn_signals = ContinuousLearningSignals()
        self._init_ui()
        self._connect_signals()
        self.init_system_data_and_model()

        if PYQTGRAPH_AVAILABLE:
            self.plot_timer = QTimer()
            self.plot_timer.timeout.connect(self.update_plots_from_cache)

    def _create_spinbox(self, min_v, max_v, def_v, step=None, dec=None):
        sb = QSpinBox() if isinstance(def_v, int) and dec is None else QDoubleSpinBox()
        sb.setRange(min_v, max_v);
        sb.setValue(def_v)
        if step: sb.setSingleStep(step)
        if dec: sb.setDecimals(dec)
        return sb

    def _get_default_hps(self) -> Dict[str, Any]:
        return {'embed_dim': EMBED_DIM_DEFAULT, 'num_experts': NUM_EXPERTS_DEFAULT,
                'expert_nhead': EXPERT_NHEAD_DEFAULT, 'expert_dim_feedforward': EXPERT_DIM_FEEDFORWARD_DEFAULT,
                'expert_layers': EXPERT_LAYERS_DEFAULT, 'router_hidden_dim': ROUTER_HIDDEN_DIM_DEFAULT,
                'max_path_len': MAX_PATH_LEN_DEFAULT, 'max_visits_per_expert': MAX_VISITS_PER_EXPERT_DEFAULT,
                'gumbel_tau': GUMBEL_TAU_INIT, 'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE, 'router_entropy_coef': ROUTER_ENTROPY_COEF,
                'animation_delay': ANIMATION_DELAY_DEFAULT, 'train_steps_per_eval': TRAIN_STEPS_PER_EVAL,
                # New HPs for enhancements
                'path_reward_penalty_coef': PATH_REWARD_PENALTY_COEF_DEFAULT,
                'router_lr_mult': ROUTER_LR_MULT_DEFAULT,
                'expert_lr_mult': EXPERT_LR_MULT_DEFAULT,
                'total_steps': TOTAL_STEPS  # Make total steps configurable via HPs
                }

    def _get_current_hps_from_ui(self) -> Dict[str, Any]:
        ui_hps = {name: spin.value() for name, spin in self.hps_spins.items()}
        ui_hps.update({
            'max_lm_seq_len': MAX_LM_SEQ_LEN, 'weight_decay': WEIGHT_DECAY,
            'gradient_clip_norm': GRADIENT_CLIP_NORM,
            'global_step': self.worker.step if self.worker else 0
        })
        return ui_hps

    def validate_hps(self, hps_to_validate: Dict[str, Any]) -> List[str]:
        errs = []
        if hps_to_validate['embed_dim'] % hps_to_validate['expert_nhead'] != 0: errs.append("EmbDim % E.NHead != 0")
        if hps_to_validate['max_path_len'] < 1: errs.append("MaxPath < 1")
        if self.train_ds and hps_to_validate['batch_size'] > len(self.train_ds.sequences):
            errs.append(f"BatchSz > train seqs ({len(self.train_ds.sequences)})")
        if EFFECTIVE_BATCH_SIZE % hps_to_validate['batch_size'] != 0:
            errs.append(f"BatchSz must divide EffBatchSz ({EFFECTIVE_BATCH_SIZE})")
        return errs

    def _init_ui(self):
        main_w = QWidget();
        self.setCentralWidget(main_w)
        top_layout = QHBoxLayout(main_w)
        left_scroll_area = QScrollArea();
        left_scroll_area.setWidgetResizable(True)
        left_panel_widget_container = QWidget();
        left_panel_layout = QVBoxLayout(left_panel_widget_container)
        left_panel_layout.setSpacing(10)

        hps_group = QGroupBox("Hyperparameters");
        hps_form = QFormLayout()
        self.hps_spins = {
            'num_experts': self._create_spinbox(1, 16, NUM_EXPERTS_DEFAULT),
            'embed_dim': self._create_spinbox(64, 1024, EMBED_DIM_DEFAULT, step=32),
            'expert_nhead': self._create_spinbox(1, 16, EXPERT_NHEAD_DEFAULT),
            'expert_dim_feedforward': self._create_spinbox(256, 4096, EXPERT_DIM_FEEDFORWARD_DEFAULT, step=128),
            'expert_layers': self._create_spinbox(1, 8, EXPERT_LAYERS_DEFAULT),
            'router_hidden_dim': self._create_spinbox(64, 1024, ROUTER_HIDDEN_DIM_DEFAULT, step=64),
            'max_path_len': self._create_spinbox(1, 16, MAX_PATH_LEN_DEFAULT),
            'max_visits_per_expert': self._create_spinbox(1, 5, MAX_VISITS_PER_EXPERT_DEFAULT),
            'gumbel_tau': self._create_spinbox(0.1, 5.0, GUMBEL_TAU_INIT, 0.1, 2),
            'learning_rate': self._create_spinbox(1e-5, 1e-3, LEARNING_RATE, None, 6),
            'batch_size': self._create_spinbox(4, min(EFFECTIVE_BATCH_SIZE, 128), BATCH_SIZE, step=4),
            'router_entropy_coef': self._create_spinbox(0.0, 0.1, ROUTER_ENTROPY_COEF, 0.001, 4),
            'animation_delay': self._create_spinbox(0.0, 1.0, ANIMATION_DELAY_DEFAULT, 0.05, 2),
            'train_steps_per_eval': self._create_spinbox(5, 200, TRAIN_STEPS_PER_EVAL, step=5),
            # New HP spinboxes
            'path_reward_penalty_coef': self._create_spinbox(0.0, 0.1, PATH_REWARD_PENALTY_COEF_DEFAULT, 0.001, 4),
            'router_lr_mult': self._create_spinbox(0.1, 2.0, ROUTER_LR_MULT_DEFAULT, 0.1, 2),
            'expert_lr_mult': self._create_spinbox(0.1, 2.0, EXPERT_LR_MULT_DEFAULT, 0.1, 2),
            'total_steps': self._create_spinbox(100, 100000, TOTAL_STEPS, step=100),
        }
        lbls = {"num_experts": "Experts", "embed_dim": "EmbDim", "expert_nhead": "E.NHead",
                "expert_dim_feedforward": "E.FFDim", "expert_layers": "E.Layers", "router_hidden_dim": "RtrHid",
                "max_path_len": "MaxPath", "max_visits_per_expert": "MaxVisits", "gumbel_tau": "GumbelTau",
                "learning_rate": "LR", "batch_size": "BatchSz", "router_entropy_coef": "EntrCoef",
                "animation_delay": "AnimDelay", "train_steps_per_eval": "EvalSteps",
                "path_reward_penalty_coef": "PathPenCoef", "router_lr_mult": "RtrLRMult",
                "expert_lr_mult": "ExpLRMult", "total_steps": "TotalSteps"}
        for name, spin in self.hps_spins.items(): hps_form.addRow(f"{lbls.get(name, name)}:", spin)
        hps_group.setLayout(hps_form);
        left_panel_layout.addWidget(hps_group)

        actions_group = QGroupBox("System Control");
        actions_layout = QVBoxLayout()
        self.reinit_btn = QPushButton("Apply & Reinitialize");
        self.start_btn = QPushButton("Start Learning")
        self.stop_btn = QPushButton("Stop Learning", enabled=False);
        self.anim_btn = QPushButton("Animate Sample", enabled=False)
        for btn in [self.reinit_btn, self.start_btn, self.stop_btn, self.anim_btn]: actions_layout.addWidget(btn)
        actions_group.setLayout(actions_layout);
        left_panel_layout.addWidget(actions_group)

        self.diagram = DiagramWidget(num_experts=self.hps['num_experts'])
        left_panel_layout.addWidget(self.diagram);
        left_panel_layout.addStretch(1)
        left_scroll_area.setWidget(left_panel_widget_container);
        top_layout.addWidget(left_scroll_area, stretch=1)

        right_panel_layout = QVBoxLayout();
        self.tabs = QTabWidget()
        plots_tab = QWidget();
        plots_layout = QVBoxLayout(plots_tab)
        if PYQTGRAPH_AVAILABLE:
            pg_palette = self.palette();
            pg.setConfigOptions(antialias=True, background=pg_palette.color(pg_palette.Background),
                                foreground=pg_palette.color(pg_palette.Foreground))
            legend_brush = pg.mkBrush(200, 200, 200, 100);
            legend_text_color = pg_palette.color(pg_palette.Text)
            plot_cfgs = [
                ("loss_plot", "Loss Curves", [("train_loss_curve", 'b', "Train"), ("val_loss_curve", 'r', "Val")],
                 "Loss", "Eval Steps"),
                ("ppl_plot", "Val Perplexity", [("ppl_curve", 'g', "PPL")], "PPL", "Eval Steps"),
                ("aux_plot", "Aux Metrics",
                 [("path_len_curve", 'c', "PathLen"), ("router_ent_curve", 'm', "RouterEnt")], "Value", "Eval Steps")]
            for w_name, title, curves, y_lbl, x_lbl in plot_cfgs:
                plot_w = pg.PlotWidget(title=title);
                setattr(self, w_name, plot_w)
                if len(curves) > 1: plot_w.addLegend(offset=(-10, 10), brush=legend_brush,
                                                     labelTextColor=legend_text_color)
                for c_attr, c_col, c_name in curves: setattr(self, c_attr,
                                                             plot_w.plot(pen=pg.mkPen(c_col, width=2), name=c_name))
                plot_w.setLabel('left', y_lbl);
                plot_w.setLabel('bottom', x_lbl);
                plots_layout.addWidget(plot_w)
            self.path_viz = GLViewWidget();
            self.path_viz.setCameraPosition(distance=40, elevation=20, azimuth=45)
            plots_layout.addWidget(self.path_viz, stretch=1)
        else:
            plots_layout.addWidget(QLabel("pyqtgraph not installed. Plots and 3D view unavailable."))
        self.tabs.addTab(plots_tab, "Live Metrics")

        stats_tab = QWidget();
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.addWidget(QLabel("Expert Usage Freq:"))
        if PYQTGRAPH_AVAILABLE:
            self.expert_usage_plot = pg.PlotWidget(); self.expert_bars = None; stats_layout.addWidget(
                self.expert_usage_plot, stretch=1)
        else:
            stats_layout.addWidget(QLabel("pyqtgraph needed for chart."))
        stats_layout.addWidget(QLabel("Pathway Freq:"));
        self.path_stats_text = QTextEdit(readOnly=True);
        stats_layout.addWidget(self.path_stats_text, stretch=1)
        self.tabs.addTab(stats_tab, "Analysis")

        log_tab = QWidget();
        log_layout = QVBoxLayout(log_tab)
        log_layout.addWidget(QLabel("Log:"));
        self.log_edit = QTextEdit(readOnly=True);
        self.log_edit.setFixedHeight(200);
        log_layout.addWidget(self.log_edit)
        log_layout.addWidget(QLabel("Sample Info:"));
        self.sample_edit = QTextEdit(readOnly=True);
        self.sample_edit.setFixedHeight(120);
        log_layout.addWidget(self.sample_edit)
        log_layout.addStretch(1);
        self.tabs.addTab(log_tab, "Log & Sample")
        right_panel_layout.addWidget(self.tabs);
        top_layout.addLayout(right_panel_layout, stretch=2)
        self.hps_spins['embed_dim'].valueChanged.connect(self.update_nhead_options);
        self.update_nhead_options()

    def update_nhead_options(self):
        emb_dim_val = self.hps_spins['embed_dim'].value();
        nhead_spin = self.hps_spins['expert_nhead']
        max_allowable_nhead = emb_dim_val
        valid_nheads = [i for i in range(1, min(nhead_spin.maximum(), max_allowable_nhead) + 1) if emb_dim_val % i == 0]
        current_nhead_val = nhead_spin.value()
        if valid_nheads:
            nhead_spin.setRange(min(valid_nheads), max(valid_nheads))
            if current_nhead_val not in valid_nheads:
                nhead_spin.setValue(
                    min(valid_nheads, key=lambda x: abs(x - current_nhead_val)) if current_nhead_val in range(
                        min(valid_nheads), max(valid_nheads) + 1) else valid_nheads[0])
        else:
            nhead_spin.setRange(1, 1); nhead_spin.setValue(1)

    def _connect_signals(self):
        self.reinit_btn.clicked.connect(self.init_system_data_and_model);
        self.start_btn.clicked.connect(self.start_continuous_learning)
        self.stop_btn.clicked.connect(self.stop_continuous_learning);
        self.anim_btn.clicked.connect(self.trigger_one_sample_animation)
        asig = self.anim_signals
        asig.signal_input_text.connect(self.on_input_text_anim);
        asig.signal_embedded.connect(self.on_embedded_anim)
        asig.signal_to_router.connect(self.on_to_router_anim);
        asig.signal_router_output.connect(self.on_router_output_anim)
        asig.signal_to_expert.connect(self.on_to_expert_anim);
        asig.signal_expert_output.connect(self.on_expert_output_anim)
        asig.signal_to_output_layer.connect(self.on_to_output_layer_anim);
        asig.signal_final_prediction_lm.connect(self.on_final_prediction_lm_anim)
        lsig = self.learn_signals
        lsig.log_message.connect(self.log_message);
        lsig.stats_update.connect(self.handle_stats_update)
        lsig.learning_stopped.connect(self.on_learning_stopped_ui_update)

    def init_system_data_and_model(self):
        self.log_message("Initializing system...")
        if self.worker and self.worker.isRunning():
            self.log_message("Stopping active worker...");
            self.stop_continuous_learning()
            if not self.worker.wait(3000): self.log_message(
                "Warn: Worker did not terminate gracefully. Forcing."); self.worker.terminate(); self.worker.wait()
            self.worker = None

        self.hps = self._get_current_hps_from_ui()
        if (errors := self.validate_hps(self.hps)): self.log_message(f"ERROR: Invalid HPs: {'; '.join(errors)}"); return

        tokens = tokenize_text(WIKITEXT_SAMPLE)
        for _ in range(3): tokens.extend(tokenize_text(augment_text(WIKITEXT_SAMPLE)))
        split_idx = int(len(tokens) * TRAIN_SPLIT_RATIO);
        train_tok, val_tok = tokens[:split_idx], tokens[split_idx:]

        self.vocab = Vocabulary(MAX_VOCAB_SIZE, MIN_FREQ)
        self.train_ds = LanguageModelDataset(train_tok, self.vocab, self.hps['max_lm_seq_len'], "train")
        self.anim_signals.vocab = self.vocab
        self.val_ds = LanguageModelDataset(val_tok, self.vocab, self.hps['max_lm_seq_len'], "val")

        if not self.train_ds.sequences or not self.val_ds.sequences:
            self.log_message("ERROR: No sequences generated. Check MAX_LM_SEQ_LEN/data.");
            self.start_btn.setEnabled(False);
            return

        dl_args = {'num_workers': 2 if sys.platform != "win32" else 0,
                   'pin_memory': True if DEVICE.type == 'cuda' else False}
        self.train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=self.hps['batch_size'], shuffle=True,
                                                    drop_last=True, **dl_args)
        self.val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size=self.hps['batch_size'], shuffle=False,
                                                  **dl_args)
        self.log_message(f"Data: Train {len(self.train_ds)}, Val {len(self.val_ds)} seqs. Vocab: {len(self.vocab)}.")

        self.model = GoEModel(len(self.vocab), self.hps['embed_dim'], self.hps['num_experts'], self.hps['expert_nhead'],
                              self.hps['expert_dim_feedforward'], self.hps['expert_layers'],
                              self.hps['router_hidden_dim'],
                              self.hps['max_path_len'], self.hps['max_visits_per_expert'], self.hps['max_lm_seq_len'],
                              self.hps['gumbel_tau'], self.hps['path_reward_penalty_coef'])  # Pass new HP
        self.model.to(DEVICE);
        self.model.animation_signals_obj = self.anim_signals

        # Enhancement 4: Optimizer with parameter groups for different LRs
        router_params = [p for n, p in self.model.named_parameters() if 'router.' in n and p.requires_grad]
        expert_params = [p for n, p in self.model.named_parameters() if 'experts.' in n and p.requires_grad]

        base_param_names = set(n for n, p in self.model.named_parameters() if p.requires_grad) \
                           - set(n for n, p in self.model.named_parameters() if 'router.' in n) \
                           - set(n for n, p in self.model.named_parameters() if 'experts.' in n)
        base_params = [p for n, p in self.model.named_parameters() if n in base_param_names and p.requires_grad]

        param_groups = []
        if base_params: param_groups.append({'params': base_params, 'lr': self.hps['learning_rate']})
        if router_params: param_groups.append(
            {'params': router_params, 'lr': self.hps['learning_rate'] * self.hps['router_lr_mult']})
        if expert_params: param_groups.append(
            {'params': expert_params, 'lr': self.hps['learning_rate'] * self.hps['expert_lr_mult']})

        if not param_groups:  # Should not happen with this model, but as a safeguard
            self.log_message("Warning: No parameters found for optimizer.")
            param_groups = [{'params': list(self.model.parameters())}]

        self.optim = optim.AdamW(param_groups, lr=self.hps['learning_rate'],
                                 # Default LR (used if a group doesn't specify)
                                 weight_decay=self.hps['weight_decay'], eps=1e-8)
        self.crit = nn.CrossEntropyLoss(ignore_index=self.vocab.word_to_idx["<pad>"]).to(DEVICE)

        self.log_message(f"Model initialized ({self.hps['num_experts']} experts) on {DEVICE}.")
        self.diagram.update_num_experts(self.hps['num_experts'])
        self.path_stats_text.clear();
        self.sample_edit.clear();
        self.reset_plots()
        self.start_btn.setEnabled(True);
        self.stop_btn.setEnabled(False);
        self.anim_btn.setEnabled(False)
        self.learn_signals.model_initialization_done.emit();
        self.log_message("Initialization complete.")

    def reset_plots(self):
        if not PYQTGRAPH_AVAILABLE: return
        plot_attrs = ["train_loss_curve", "val_loss_curve", "ppl_curve", "path_len_curve", "router_ent_curve"]
        for attr in plot_attrs:
            if hasattr(self, attr): curve = getattr(self, attr); curve.clear() if curve else None
        if hasattr(self, 'expert_bars') and self.expert_bars:
            if self.expert_usage_plot: self.expert_usage_plot.removeItem(self.expert_bars); self.expert_bars = None
        if hasattr(self, 'path_viz') and self.path_viz: self.path_viz.clear()
        with self.plot_cache_lock:
            self.plot_cache = None

    def start_continuous_learning(self):
        if not all([self.model, self.train_dl, self.val_dl, self.optim, self.crit]):
            self.log_message("System not fully initialized. Cannot start.");
            return
        self.log_message("Starting learning worker...")
        self.hps = self._get_current_hps_from_ui()  # Refresh HPs from UI before starting worker
        self.worker = ContinuousLearningWorker(lambda: (self.model, self.optim, self.crit),
                                               lambda: (self.train_dl, self.val_dl), self.hps, self.learn_signals)
        self.worker.start()
        if PYQTGRAPH_AVAILABLE and hasattr(self,
                                           'plot_timer') and not self.plot_timer.isActive(): self.plot_timer.start(
            PLOT_UPDATE_INTERVAL_MS)
        self.start_btn.setEnabled(False);
        self.stop_btn.setEnabled(True);
        self.anim_btn.setEnabled(True);
        self.reinit_btn.setEnabled(False)

    def stop_continuous_learning(self):
        if self.worker and self.worker.isRunning(): self.log_message(
            "Stopping learning worker..."); self.worker.stop_learning()
        if PYQTGRAPH_AVAILABLE and hasattr(self, 'plot_timer') and self.plot_timer.isActive(): self.plot_timer.stop()

    def on_learning_stopped_ui_update(self):
        self.start_btn.setEnabled(True);
        self.stop_btn.setEnabled(False);
        self.anim_btn.setEnabled(False);
        self.reinit_btn.setEnabled(True)
        self.log_message("Worker has stopped.")

    def trigger_one_sample_animation(self):
        if not (self.worker and self.worker.isRunning()): self.log_message(
            "Worker not running. Cannot animate."); return
        if not (self.val_ds and self.val_ds.sequences): self.log_message(
            "Validation data for animation missing."); return
        self.diagram.reset_highlights();
        self.sample_edit.clear()
        idx = random.randint(0, len(self.val_ds) - 1)
        in_ids_anim, tgt_id_anim, snippet_anim = self.val_ds[idx]
        self.log_message(f"Requesting animation for sample {idx}: {snippet_anim[:50]}...")
        current_hps_for_anim = self._get_current_hps_from_ui()
        self.model.current_hps = current_hps_for_anim
        self.worker.request_animation((in_ids_anim, tgt_id_anim, snippet_anim))

    def log_message(self, msg: str):
        self.log_edit.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def handle_stats_update(self, stats: Dict[str, Any]):
        with self.plot_cache_lock:
            self.plot_cache = stats.copy()
        self.path_stats_text.clear()
        path_freq = stats.get('pathway_freq', {});
        total_paths = sum(path_freq.values())
        self.path_stats_text.append(f"--- Pathway Frequencies (Total: {total_paths}) ---")
        sorted_paths = sorted(path_freq.items(), key=lambda x: x[1], reverse=True)
        for i, (path_tuple, count) in enumerate(sorted_paths[:20]): self.path_stats_text.append(
            f"{'->'.join(map(str, path_tuple))}: {count} ({count / max(1, total_paths):.2%})")
        if len(sorted_paths) > 20: self.path_stats_text.append("...")
        self.path_stats_text.append("\n--- Compilation Candidates (>5% usage) ---")
        found_candidates = False
        for p_tuple, c_val in sorted_paths:
            if total_paths > 0 and c_val / total_paths > 0.05: self.path_stats_text.append(
                f"MERGE: {'->'.join(map(str, p_tuple))} ({c_val / total_paths:.2%})"); found_candidates = True
        if not found_candidates: self.path_stats_text.append("None.")

        lr_factor_msg = f", LRFactor: {stats.get('current_lr_factor', 1.0):.3f}" if 'current_lr_factor' in stats else ""
        self.log_message(
            f"Step {stats.get('global_step', 0)}: VL={stats.get('current_val_loss', float('nan')):.4f}, PPL={stats.get('current_perplexity', float('nan')):.2f}, GN={stats.get('grad_norm', float('nan')):.2f}, R={stats.get('reward_mean', float('nan')):.2f}{lr_factor_msg}")
        self.diagram.set_expert_usage_text(stats.get('expert_usage', {}))

        if PYQTGRAPH_AVAILABLE and hasattr(self, 'path_viz') and self.path_viz:
            self.path_viz.clear()
            num_exp_viz = self.hps.get('num_experts', NUM_EXPERTS_DEFAULT)
            node_coords = [[0, 0, -2]]
            for i in range(num_exp_viz): angle = 2 * math.pi * i / num_exp_viz - math.pi / 2; node_coords.append(
                [15 * math.cos(angle), 15 * math.sin(angle), 0])
            node_coords.append([0, 0, 2]);
            node_pos_np = np.array(node_coords, dtype=np.float32)
            colors = [(1, 1, 0, 1)] + [(0, 1, 1, 1)] * num_exp_viz + [(0, 1, 0, 1)]
            self.path_viz.addItem(GLScatterPlotItem(pos=node_pos_np, size=1.5, color=np.array(colors), pxMode=False))
            output_node_idx = num_exp_viz + 1
            for path_tuple, count_val in sorted_paths[:5]:
                current_path_indices = [0]
                for step_item in path_tuple:
                    if isinstance(step_item, int):
                        current_path_indices.append(step_item + 1)
                    elif step_item in ('T', 'T_max'):
                        current_path_indices.append(output_node_idx); break
                if len(current_path_indices) > 1:
                    line_segments_pos = [
                        [node_pos_np[current_path_indices[i_seg]], node_pos_np[current_path_indices[i_seg + 1]]] for
                        i_seg in range(len(current_path_indices) - 1)]
                    if line_segments_pos:
                        line_segments_np = np.array(line_segments_pos, dtype=np.float32)
                        line_alpha = min(1.0,
                                         0.2 + 0.8 * (count_val / max(1, sorted_paths[0][1] if sorted_paths else 1)))
                        self.path_viz.addItem(
                            GLLinePlotItem(pos=line_segments_np, color=(0.7, 0.7, 0.7, line_alpha), width=1.5))

    def update_plots_from_cache(self):
        if not PYQTGRAPH_AVAILABLE or not self.plot_cache: return
        with self.plot_cache_lock:
            stats = self.plot_cache.copy(); self.plot_cache = None
        if not stats: return
        x_eval_steps = list(range(len(stats.get('val_loss_hist', []))))
        plot_map = {"train_loss_curve": "train_loss_hist", "val_loss_curve": "val_loss_hist",
                    "ppl_curve": "val_perp_hist", "path_len_curve": "path_len_hist",
                    "router_ent_curve": "router_ent_hist"}
        for curve_attr, data_key in plot_map.items():
            if hasattr(self, curve_attr):
                curve_widget = getattr(self, curve_attr);
                data_y = stats.get(data_key, [])
                data_x = list(range(len(data_y))) if "train_loss" in data_key else x_eval_steps[:len(data_y)]
                if curve_widget: curve_widget.setData(x=data_x, y=data_y)
        if hasattr(self, 'expert_usage_plot') and self.expert_usage_plot:
            usage_data = stats.get('expert_usage', {});
            num_exp_plot = self.hps.get('num_experts', NUM_EXPERTS_DEFAULT)
            ticks_list = [[(i, f"E{i}") for i in range(num_exp_plot)]];
            heights_list = [usage_data.get(i, 0) for i in range(num_exp_plot)]
            if self.expert_bars: self.expert_usage_plot.removeItem(self.expert_bars)
            self.expert_bars = pg.BarGraphItem(x=list(range(num_exp_plot)), height=heights_list, width=0.6,
                                               brush='teal')
            self.expert_usage_plot.addItem(self.expert_bars);
            self.expert_usage_plot.getAxis('bottom').setTicks(ticks_list)
            self.expert_usage_plot.setLabel('left', "Usage Count");
            self.expert_usage_plot.setTitle("Expert Usage Frequency")

    def on_input_text_anim(self, t: str):
        self.diagram.reset_highlights(); self.diagram.highlight_node("INPUT"); self.sample_edit.setText(
            f"Input: {t[:100]}...")

    def on_embedded_anim(self, r: str):
        self.diagram.highlight_node("INPUT", False); self.diagram.highlight_connection("INPUT",
                                                                                       "EMBED"); self.diagram.highlight_node(
            "EMBED"); self.diagram.set_representation_text(r)

    def on_to_router_anim(self, r: str, p: List[Any]):
        prev_node = "EMBED" if not p else (f"EXPERT_{p[-1]}" if isinstance(p[-1], int) else "ROUTER")
        self.diagram.highlight_node(prev_node, False);
        self.diagram.highlight_connection(prev_node, "ROUTER");
        self.diagram.highlight_node("ROUTER")
        self.diagram.set_representation_text(r);
        self.diagram.set_current_path_text(p)

    def on_router_output_anim(self, probs: List[float], action_idx: int, is_term: bool):
        self.diagram.set_router_probs_text(probs);
        self.diagram.highlight_node("ROUTER", False)
        to_node_name = "LM_HEAD" if is_term else f"EXPERT_{action_idx}"
        self.diagram.highlight_connection("ROUTER", to_node_name)
        if is_term: self.diagram.highlight_node("LM_HEAD")

    def on_to_expert_anim(self, idx: int, r: str):
        self.diagram.highlight_node(f"EXPERT_{idx}"); self.diagram.set_representation_text(r)

    def on_expert_output_anim(self, idx: int, r: str):
        self.diagram.set_representation_text(r)

    def on_to_output_layer_anim(self, r: str):
        current_path_str = self.diagram.current_path_text;
        last_proc_node = "ROUTER"
        if "Path: " in current_path_str:
            elements = current_path_str.replace("Path: ", "").split("->")
            if elements:
                relevant_elements = [el for el in elements if el not in ('T', 'T_max')]
                if relevant_elements and relevant_elements[-1].isdigit():
                    last_proc_node = f"EXPERT_{relevant_elements[-1]}"
                elif not relevant_elements and "EMBED" in self.diagram.active_elements:
                    last_proc_node = "ROUTER"
                elif len(elements) == 1 and elements[0] == 'T':
                    if "ROUTER" in self.diagram.active_elements and self.diagram.active_elements["ROUTER"] != QColor(
                        "lightblue"): last_proc_node = "ROUTER"
        self.diagram.highlight_node(last_proc_node, False);
        self.diagram.highlight_connection(last_proc_node, "LM_HEAD", active=True)
        self.diagram.highlight_node("LM_HEAD", True);
        self.diagram.set_representation_text(r)

    def on_final_prediction_lm_anim(self, pred: str, path: List[Any]):
        self.diagram.highlight_node("LM_HEAD", True);
        self.diagram.set_current_path_text(path)
        self.sample_edit.setText(f"{self.sample_edit.toPlainText()}\nPred: '{pred}', Path: {'->'.join(map(str, path))}")

    def closeEvent(self, event):
        self.log_message("Close event received. Shutting down worker...");
        self.stop_continuous_learning()
        if self.worker and self.worker.isRunning():
            if not self.worker.wait(5000): self.log_message(
                "Worker did not stop in time. Terminating."); self.worker.terminate(); self.worker.wait()
        super().closeEvent(event)


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    if not PYQTGRAPH_AVAILABLE:
        try:
            from PyQt5.QtWidgets import QMessageBox

            msg_box = QMessageBox();
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setText("Critical Dependency Error: pyqtgraph (and pyopengl for 3D) is required.")
            msg_box.setInformativeText("Please install it (e.g., 'pip install pyqtgraph pyopengl') and try again.")
            msg_box.setWindowTitle("Dependency Error");
            msg_box.exec_()
        except ImportError:
            print("ERROR: PyQt5 is required.")
        sys.exit(1)
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required."); sys.exit(1)
    mainWin = MainWindow();
    mainWin.show();
    sys.exit(app.exec_())