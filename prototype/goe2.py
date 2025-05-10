import math
import sys
import time
import random
import collections
from typing import List, Tuple, Dict, Any

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
# Data params
MAX_SEQ_LEN = 20
VOCAB = {"<pad>": 0, "<unk>": 1, "this": 2, "is": 3, "a": 4, "great": 5, "sentence": 6, "about": 7,
         "positive": 8, "things": 9, "very": 10, "bad": 11, "example": 12, "of": 13, "negative": 14,
         "stuff": 15, "expert": 16, "one": 17, "two": 18, "three": 19, "specific": 20, "input": 21,
         "for": 22, "task": 23, "analyze": 24, "syntax": 25, "understand": 26, "semantics": 27,
         "process": 28, "context": 29, "general": 30, "query": 31}
VOCAB_SIZE = len(VOCAB)
WORD_TO_IDX = VOCAB
IDX_TO_WORD = {i: w for w, i in VOCAB.items()}

# Model HPs
EMBED_DIM = 32
EXPERT_HIDDEN_DIM = 64
ROUTER_HIDDEN_DIM = 64
NUM_EXPERTS = 3
NUM_CLASSES = 3
MAX_PATH_LEN_DEFAULT = 5
MAX_VISITS_PER_EXPERT_IN_PATH_DEFAULT = 1

# Training HPs
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
NUM_EPOCHS = 10
ROUTER_ENTROPY_COEF = 0.01
WEIGHT_DECAY = 1e-4

# Animation
ANIMATION_DELAY_DEFAULT = 0.7

# Global variables to be updated by UI
MAX_PATH_LEN = MAX_PATH_LEN_DEFAULT
MAX_VISITS_PER_EXPERT_IN_PATH = MAX_VISITS_PER_EXPERT_IN_PATH_DEFAULT
ANIMATION_DELAY = ANIMATION_DELAY_DEFAULT


# --- PyTorch Model Components ---

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, expert_id):
        super().__init__()
        self.expert_id = expert_id
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.specialization_tag = nn.Parameter(torch.randn(1, output_dim) * 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x + self.specialization_tag


class RoutingController(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_experts + 1)

    def forward(self, x, current_visit_counts_for_batch_item, max_visits_per_expert):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)

        # Mask experts based on visit counts for the current sample path
        # current_visit_counts_for_batch_item is (num_experts) for a single sample
        # or a list/tensor of such for a batch
        if x.dim() == 1:  # Single sample in batch (during animation) or unbatched
            for expert_idx in range(self.num_experts):
                if current_visit_counts_for_batch_item[expert_idx] >= max_visits_per_expert:
                    logits[expert_idx] = -float('inf')
        elif x.dim() == 2:  # Batched processing
            # current_visit_counts_for_batch_item here is actually current_visit_counts for the whole active batch
            # Shape: (current_active_batch_size, num_experts)
            for i in range(logits.size(0)):  # Iterate over batch items
                for expert_idx in range(self.num_experts):
                    if current_visit_counts_for_batch_item[i, expert_idx] >= max_visits_per_expert:
                        logits[i, expert_idx] = -float('inf')
        return logits


class GoEModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, expert_hidden_dim, router_hidden_dim,
                 num_experts, num_classes):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_classes = num_classes

        # These will be updated by UI
        self.max_path_len = MAX_PATH_LEN
        self.max_visits_per_expert_in_path = MAX_VISITS_PER_EXPERT_IN_PATH

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=WORD_TO_IDX["<pad>"])
        self.experts = nn.ModuleList([
            Expert(embed_dim, embed_dim, expert_hidden_dim, i) for i in range(num_experts)
        ])
        self.router = RoutingController(embed_dim, router_hidden_dim, num_experts)
        self.output_classifier = nn.Linear(embed_dim, num_classes)

        self.animation_signals = None
        self.is_animating = False

    def _emit_signal(self, signal_type: str, *args):
        if self.is_animating and self.animation_signals:
            getattr(self.animation_signals, signal_type).emit(*args)
            time.sleep(ANIMATION_DELAY)  # Use global ANIMATION_DELAY

    def forward(self, input_ids, current_sample_text_for_anim=None):
        batch_size = input_ids.size(0)

        if self.is_animating and current_sample_text_for_anim:
            self._emit_signal("signal_input_text", current_sample_text_for_anim)

        embedded_input = self.embedding(input_ids)
        current_representation_batch = embedded_input.mean(dim=1)

        if self.is_animating:
            self._emit_signal("signal_embedded", repr_tensor(current_representation_batch[0]))

        total_router_entropy = torch.tensor(0.0, device=input_ids.device)
        paths_taken_indices = [[] for _ in range(batch_size)]
        current_visit_counts_batch = torch.zeros((batch_size, self.num_experts), dtype=torch.long,
                                                 device=input_ids.device)

        active_indices_original = torch.arange(batch_size, device=input_ids.device)
        final_representations_batch = torch.zeros_like(current_representation_batch)

        # For loop operates on a potentially shrinking batch of active samples
        current_active_representations = current_representation_batch
        current_active_indices = active_indices_original.clone()

        for step in range(self.max_path_len):
            if not current_active_indices.numel():
                break

            # For animation, we focus on the first sample of the original batch if it's still active
            anim_sample_original_idx = 0
            is_anim_sample_active = anim_sample_original_idx in current_active_indices
            anim_sample_repr_for_signal = None
            anim_sample_current_path_for_signal = None

            if self.is_animating and is_anim_sample_active:
                # Find where anim_sample_original_idx is in current_active_indices
                anim_sample_current_batch_idx = \
                (current_active_indices == anim_sample_original_idx).nonzero(as_tuple=True)[0].item()
                anim_sample_repr_for_signal = repr_tensor(current_active_representations[anim_sample_current_batch_idx])
                anim_sample_current_path_for_signal = [str(p) for p in paths_taken_indices[anim_sample_original_idx]]
                self._emit_signal("signal_to_router", anim_sample_repr_for_signal, anim_sample_current_path_for_signal)

            # Router processes only active samples
            # Pass visit counts for the currently active samples
            router_logits = self.router(current_active_representations,
                                        current_visit_counts_batch[current_active_indices],
                                        self.max_visits_per_expert_in_path)

            router_probs = F.softmax(router_logits, dim=1)
            step_entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-9), dim=1).mean()
            total_router_entropy += step_entropy

            if self.training:
                one_hot_decision = F.gumbel_softmax(router_logits, tau=1.0, hard=True)
            else:
                decision_indices = torch.argmax(router_logits, dim=1)
                one_hot_decision = F.one_hot(decision_indices, num_classes=self.num_experts + 1).float()

            terminate_decision = one_hot_decision[:, self.num_experts]
            expert_selection_one_hot = one_hot_decision[:, :self.num_experts]

            if self.is_animating and is_anim_sample_active:
                anim_sample_current_batch_idx = \
                (current_active_indices == anim_sample_original_idx).nonzero(as_tuple=True)[0].item()
                flat_probs_anim = router_probs[anim_sample_current_batch_idx].tolist()
                selected_idx_anim = one_hot_decision[anim_sample_current_batch_idx].argmax().item()
                is_terminate_anim = (selected_idx_anim == self.num_experts)
                self._emit_signal("signal_router_output", flat_probs_anim, selected_idx_anim, is_terminate_anim)

            # Handle terminated samples
            terminated_mask_in_active = (terminate_decision == 1)
            if terminated_mask_in_active.any():
                original_indices_of_terminated_now = current_active_indices[terminated_mask_in_active]
                final_representations_batch[original_indices_of_terminated_now] = current_active_representations[
                    terminated_mask_in_active]
                for orig_idx in original_indices_of_terminated_now.tolist():
                    paths_taken_indices[orig_idx].append('T')

            # Prepare for next step: only non-terminated samples continue
            chose_expert_mask_in_active = (terminate_decision == 0)
            if not chose_expert_mask_in_active.any():  # All remaining samples chose to terminate
                current_active_indices = torch.tensor([], dtype=torch.long, device=input_ids.device)  # Empty active set
                break

                # Update active set for next iteration
            next_active_indices = current_active_indices[chose_expert_mask_in_active]

            # Filter representations and selections for those continuing
            current_active_representations = current_active_representations[chose_expert_mask_in_active]
            active_expert_selection_one_hot = expert_selection_one_hot[chose_expert_mask_in_active]
            chosen_expert_indices_for_active = active_expert_selection_one_hot.argmax(dim=1)

            # Update visit counts and paths_taken for samples that chose an expert
            for i, orig_idx in enumerate(next_active_indices.tolist()):
                expert_idx_chosen = chosen_expert_indices_for_active[i].item()
                paths_taken_indices[orig_idx].append(expert_idx_chosen)
                current_visit_counts_batch[orig_idx, expert_idx_chosen] += 1

            # Process with experts for the (newly filtered) active samples
            next_step_representations_for_active = torch.zeros_like(current_active_representations)
            for i, expert_idx_val in enumerate(chosen_expert_indices_for_active.tolist()):
                is_anim_sample_processing_expert = (self.is_animating and is_anim_sample_active and
                                                    next_active_indices[i].item() == anim_sample_original_idx)

                if is_anim_sample_processing_expert:
                    self._emit_signal("signal_to_expert", expert_idx_val,
                                      repr_tensor(current_active_representations[i]))

                expert_output = self.experts[expert_idx_val](current_active_representations[i].unsqueeze(0))
                next_step_representations_for_active[i] = expert_output.squeeze(0)

                if is_anim_sample_processing_expert:
                    self._emit_signal("signal_expert_output", expert_idx_val,
                                      repr_tensor(next_step_representations_for_active[i]))

            current_active_representations = next_step_representations_for_active
            current_active_indices = next_active_indices

        # Handle samples that reached max_path_len without explicit termination
        if current_active_indices.numel() > 0:
            final_representations_batch[current_active_indices] = current_active_representations
            for orig_idx in current_active_indices.tolist():
                paths_taken_indices[orig_idx].append('T_max')

        if self.is_animating:  # Assuming anim sample is always the first one (index 0)
            self._emit_signal("signal_to_output_layer", repr_tensor(final_representations_batch[0]))

        logits = self.output_classifier(final_representations_batch)

        if self.is_animating:
            self._emit_signal("signal_final_prediction", logits[0].argmax().item(),
                              paths_taken_indices[0] if paths_taken_indices else ["N/A"])

        return logits, total_router_entropy, paths_taken_indices


# --- Dataset and Preprocessing ---
DUMMY_DATASET = [
    ("this is a great sentence about positive things", 0),
    ("very bad example of negative stuff", 1),
    ("expert one specific input for task one", 0),
    ("expert two specific input for task two", 1),
    ("expert three specific input for task three", 2),
    ("analyze syntax for this great query", 0),
    ("understand semantics of bad stuff", 1),
    ("process context for expert three task", 2),
    ("general query positive things", 0),
    ("another sentence for negative task expert two", 1),
    ("this specific input is for task three", 2),
    ("great positive example", 0),
    ("bad negative example", 1),
    ("task three general input", 2)
]


def preprocess_text(text: str, word_to_idx: Dict[str, int], max_len: int) -> torch.Tensor:
    tokens = text.lower().split()
    indexed_tokens = [word_to_idx.get(token, word_to_idx["<unk>"]) for token in tokens]
    if len(indexed_tokens) > max_len:
        indexed_tokens = indexed_tokens[:max_len]
    else:
        indexed_tokens += [word_to_idx["<pad>"]] * (max_len - len(indexed_tokens))
    return torch.tensor(indexed_tokens, dtype=torch.long)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, word_to_idx, max_len):
        self.data = data
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        processed_text = preprocess_text(text, self.word_to_idx, self.max_len)
        return processed_text, torch.tensor(label, dtype=torch.long), text

    # --- PyQt Components ---


def repr_tensor(tensor_data, num_elems=5):
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
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 500)
        self.node_positions = {}
        self.active_elements = {}
        self.connections_to_draw = []
        self.current_representation_text = ""
        self.router_probs_text = ""
        self.current_path_text = ""
        self._setup_node_positions()

    def _setup_node_positions(self):
        width = self.width() if self.width() > 0 else 600
        height = self.height() if self.height() > 0 else 500

        self.node_size = QRectF(0, 0, 120, 40)

        self.node_positions["INPUT"] = QPointF(width / 2, height * 0.08)
        self.node_positions["EMBED"] = QPointF(width / 2, height * 0.22)
        self.node_positions["ROUTER"] = QPointF(width / 2, height * 0.40)

        expert_y = height * 0.60
        self.node_positions["EXPERT_0"] = QPointF(width * 0.25, expert_y)
        self.node_positions["EXPERT_1"] = QPointF(width * 0.5, expert_y)
        self.node_positions["EXPERT_2"] = QPointF(width * 0.75, expert_y)

        self.node_positions["CLASSIFIER"] = QPointF(width / 2, height * 0.85)

        self.base_connections = [
            ("INPUT", "EMBED"), ("EMBED", "ROUTER"),
            ("ROUTER", "EXPERT_0"), ("ROUTER", "EXPERT_1"), ("ROUTER", "EXPERT_2"),
            ("EXPERT_0", "ROUTER"), ("EXPERT_1", "ROUTER"), ("EXPERT_2", "ROUTER"),
            ("ROUTER", "CLASSIFIER")
        ]
        self.reset_highlights()

    def resizeEvent(self, event):
        self._setup_node_positions()  # Recalculate positions on resize
        super().resizeEvent(event)

    def reset_highlights(self):
        self.active_elements.clear()
        self.connections_to_draw.clear()
        for src, dest in self.base_connections:
            self.connections_to_draw.append((src, dest, QColor("lightgray"), False))  # color, is_active
        self.current_representation_text = ""
        self.router_probs_text = ""
        self.current_path_text = "Path: "
        self.update()

    def highlight_node(self, name, active=True):
        if active:
            self.active_elements[name] = QColor("yellow")
        else:
            if name in self.active_elements:
                del self.active_elements[name]
        self.update()

    def highlight_connection(self, from_name, to_name, active=True):
        # Find and update or add connection
        updated = False
        for i, (s, d, c, was_active) in enumerate(self.connections_to_draw):
            if (s == from_name and d == to_name):  # Only one direction for now
                self.connections_to_draw[i] = (s, d, QColor("orange") if active else QColor("lightgray"), active)
                updated = True
                break
        if not updated:  # Should not happen if base_connections are set up correctly
            self.connections_to_draw.append(
                (from_name, to_name, QColor("orange") if active else QColor("lightgray"), active))

        # Deactivate other connections from 'from_name' if this one is active
        if active:
            for i, (s, d, c, was_active_flag) in enumerate(self.connections_to_draw):
                if s == from_name and d != to_name:
                    self.connections_to_draw[i] = (s, d, QColor("lightgray"), False)

        self.update()

    def set_representation_text(self, text):
        self.current_representation_text = f"Repr: {text}"
        self.update()

    def set_router_probs_text(self, probs_list):
        if not probs_list:
            self.router_probs_text = ""
        else:
            formatted_probs = [f"{p:.2f}" for p in probs_list]
            self.router_probs_text = f"Router Probs (E0,E1,E2,Term): [{', '.join(formatted_probs)}]"
        self.update()

    def set_current_path_text(self, path_list):
        self.current_path_text = "Path: " + " -> ".join(map(str, path_list))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for from_name, to_name, color, is_active_conn in self.connections_to_draw:
            if from_name in self.node_positions and to_name in self.node_positions:
                p1 = self.node_positions[from_name]
                p2 = self.node_positions[to_name]
                pen = QPen(color, 2.5 if is_active_conn else 1.5, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(p1, p2)

                # Arrowhead
                line_vec = p2 - p1
                length = pow(line_vec.x() ** 2 + line_vec.y() ** 2, 0.5)
                if length == 0: continue

                norm_line_vec = line_vec / length
                angle_rad = math.atan2(norm_line_vec.y(), norm_line_vec.x())

                arrow_size = 10.0
                arrow_p1 = p2 - QPointF(arrow_size * torch.cos(torch.tensor(angle_rad - torch.pi / 6)).item(),
                                        arrow_size * torch.sin(torch.tensor(angle_rad - torch.pi / 6)).item())
                arrow_p2 = p2 - QPointF(arrow_size * torch.cos(torch.tensor(angle_rad + torch.pi / 6)).item(),
                                        arrow_size * torch.sin(torch.tensor(angle_rad + torch.pi / 6)).item())

                arrow_head = QPolygonF([p2, arrow_p1, arrow_p2])
                painter.setBrush(QBrush(color))
                painter.drawPolygon(arrow_head)

        node_font = QFont("Arial", 8)
        painter.setFont(node_font)
        for name, pos in self.node_positions.items():
            rect = QRectF(pos.x() - self.node_size.width() / 2, pos.y() - self.node_size.height() / 2,
                          self.node_size.width(), self.node_size.height())

            color = self.active_elements.get(name, QColor("lightblue"))
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.black))
            painter.drawRoundedRect(rect, 5, 5)
            painter.drawText(rect, Qt.AlignCenter, name.replace("_", " "))

        info_font = QFont("Arial", 9)
        painter.setFont(info_font)
        painter.setPen(Qt.black)
        text_y_start = 15
        painter.drawText(QPointF(10, text_y_start), self.current_representation_text)
        painter.drawText(QPointF(10, text_y_start + 20), self.router_probs_text)
        painter.drawText(QPointF(10, text_y_start + 40), self.current_path_text)


class AnimationSignals(QWidget):
    signal_input_text = pyqtSignal(str)
    signal_embedded = pyqtSignal(str)
    signal_to_router = pyqtSignal(str, list)
    signal_router_output = pyqtSignal(list, int, bool)
    signal_to_expert = pyqtSignal(int, str)
    signal_expert_output = pyqtSignal(int, str)
    signal_to_output_layer = pyqtSignal(str)
    signal_final_prediction = pyqtSignal(int, list)

    log_message = pyqtSignal(str)
    epoch_completed = pyqtSignal(int, float, float, dict)
    inference_completed = pyqtSignal(str, str, str, list)


class Worker(QThread):
    def __init__(self, model, data_loader, optimizer, criterion, mode="train", sample_to_infer=None):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.mode = mode
        self.sample_to_infer = sample_to_infer

        self.signals = AnimationSignals()
        self.model.animation_signals = self.signals
        self.pathway_stats = collections.Counter()

    def run(self):
        self.model.is_animating = True
        if self.mode == "train":
            self._train_epoch()
        elif self.mode == "infer":
            self._infer_sample()
        self.model.is_animating = False

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        has_animated_this_epoch = False

        for batch_idx, (input_ids, labels, original_texts) in enumerate(self.data_loader):
            current_sample_text_for_anim = None
            if not has_animated_this_epoch:
                self.model.is_animating = True  # Animate this first sample
                current_sample_text_for_anim = original_texts[0]
            else:
                self.model.is_animating = False  # Disable animation signals for rest of epoch

            self.optimizer.zero_grad()

            logits, router_entropy, paths = self.model(input_ids,
                                                       current_sample_text_for_anim=current_sample_text_for_anim)

            task_loss = self.criterion(logits, labels)
            loss = task_loss - ROUTER_ENTROPY_COEF * router_entropy

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += input_ids.size(0)

            for path in paths:
                self.pathway_stats[tuple(path)] += 1

            if not has_animated_this_epoch:
                self.signals.log_message.emit(
                    f"Animated sample loss: {loss.item():.4f}. True: {labels[0].item()}, Pred: {preds[0].item()}")
                has_animated_this_epoch = True

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        self.signals.epoch_completed.emit(0, avg_loss, accuracy, dict(self.pathway_stats))

    def _infer_sample(self):
        self.model.eval()
        self.model.is_animating = True

        input_ids, original_text, true_label = self.sample_to_infer
        input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            logits, _, paths = self.model(input_ids, current_sample_text_for_anim=original_text)

        pred_idx = logits.argmax(dim=1).item()
        path_taken = paths[0]
        self.pathway_stats[tuple(path_taken)] += 1

        pred_label_str = f"Class {pred_idx}"
        true_label_str = f"Class {true_label.item()}" if true_label is not None else "N/A"

        self.signals.inference_completed.emit(original_text, pred_label_str, true_label_str, path_taken)
        self.signals.log_message.emit(f"Inferred. Path: {path_taken}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph of Experts Demo")
        self.setGeometry(100, 100, 1100, 750)

        self.model = GoEModel(VOCAB_SIZE, EMBED_DIM, EXPERT_HIDDEN_DIM, ROUTER_HIDDEN_DIM,
                              NUM_EXPERTS, NUM_CLASSES)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.criterion = nn.CrossEntropyLoss()

        self.dataset = TextDataset(DUMMY_DATASET, WORD_TO_IDX, MAX_SEQ_LEN)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.current_epoch = 0
        self.pathway_frequencies = collections.Counter()

        self._init_ui()
        self._connect_signals_basic()
        self.update_model_hps()  # Initialize model HPs from spinboxes

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        self.diagram_widget = DiagramWidget()
        layout.addWidget(self.diagram_widget, stretch=2)

        controls_layout = QVBoxLayout()
        controls_group = QGroupBox("Controls")
        form_layout = QFormLayout()
        self.train_button = QPushButton("Train 1 Epoch")
        self.infer_button = QPushButton("Infer One Sample")
        self.reset_anim_button = QPushButton("Reset Animation/Highlights")

        self.max_path_len_spin = QSpinBox()
        self.max_path_len_spin.setRange(1, 10)
        self.max_path_len_spin.setValue(MAX_PATH_LEN_DEFAULT)

        self.max_visits_spin = QSpinBox()
        self.max_visits_spin.setRange(1, 5)
        self.max_visits_spin.setValue(MAX_VISITS_PER_EXPERT_IN_PATH_DEFAULT)

        self.anim_delay_spin = QDoubleSpinBox()
        self.anim_delay_spin.setRange(0.0, 5.0)  # Allow 0 for no delay
        self.anim_delay_spin.setSingleStep(0.1)
        self.anim_delay_spin.setValue(ANIMATION_DELAY_DEFAULT)

        form_layout.addRow(self.train_button)
        form_layout.addRow(self.infer_button)
        form_layout.addRow(self.reset_anim_button)
        form_layout.addRow("Max Path Len:", self.max_path_len_spin)
        form_layout.addRow("Max Visits/Expert (Cycle):", self.max_visits_spin)
        form_layout.addRow("Animation Delay (s):", self.anim_delay_spin)
        controls_group.setLayout(form_layout)
        controls_layout.addWidget(controls_group)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True);
        self.log_text_edit.setFixedHeight(150)
        controls_layout.addWidget(QLabel("Log:"));
        controls_layout.addWidget(self.log_text_edit)

        self.stats_text_edit = QTextEdit()
        self.stats_text_edit.setReadOnly(True);
        self.stats_text_edit.setFixedHeight(180)
        controls_layout.addWidget(QLabel("Epoch Stats & Pathway Frequencies (for Compilation):"))
        controls_layout.addWidget(self.stats_text_edit)

        self.current_sample_text_edit = QTextEdit()
        self.current_sample_text_edit.setReadOnly(True);
        self.current_sample_text_edit.setFixedHeight(100)
        controls_layout.addWidget(QLabel("Current Sample Info:"))
        controls_layout.addWidget(self.current_sample_text_edit)

        layout.addLayout(controls_layout, stretch=1)

    def update_model_hps(self):
        global MAX_PATH_LEN, MAX_VISITS_PER_EXPERT_IN_PATH, ANIMATION_DELAY  # Update globals
        MAX_PATH_LEN = self.max_path_len_spin.value()
        MAX_VISITS_PER_EXPERT_IN_PATH = self.max_visits_spin.value()
        ANIMATION_DELAY = self.anim_delay_spin.value()

        self.model.max_path_len = MAX_PATH_LEN
        self.model.max_visits_per_expert_in_path = MAX_VISITS_PER_EXPERT_IN_PATH

        self.log_message(
            f"Updated HPs: MaxPath={MAX_PATH_LEN}, MaxVisits={MAX_VISITS_PER_EXPERT_IN_PATH}, AnimDelay={ANIMATION_DELAY:.1f}s")

    def _connect_signals_basic(self):
        self.train_button.clicked.connect(self.start_training)
        self.infer_button.clicked.connect(self.start_inference)
        self.reset_anim_button.clicked.connect(self.reset_animation_display)
        self.max_path_len_spin.valueChanged.connect(self.update_model_hps)
        self.max_visits_spin.valueChanged.connect(self.update_model_hps)
        self.anim_delay_spin.valueChanged.connect(self.update_model_hps)

    def _connect_worker_signals(self, worker_signals):
        worker_signals.signal_input_text.connect(self.on_input_text)
        worker_signals.signal_embedded.connect(self.on_embedded)
        worker_signals.signal_to_router.connect(self.on_to_router)
        worker_signals.signal_router_output.connect(self.on_router_output)
        worker_signals.signal_to_expert.connect(self.on_to_expert)
        worker_signals.signal_expert_output.connect(self.on_expert_output)
        worker_signals.signal_to_output_layer.connect(self.on_to_output_layer)
        worker_signals.signal_final_prediction.connect(self.on_final_prediction_anim)
        worker_signals.log_message.connect(self.log_message)
        worker_signals.epoch_completed.connect(self.on_epoch_completed)
        worker_signals.inference_completed.connect(self.on_inference_completed)

    def reset_animation_display(self):
        self.diagram_widget.reset_highlights()
        self.current_sample_text_edit.clear()
        self.log_message("Animation display reset.")

    def start_training(self):
        self.reset_animation_display()
        self.log_message(f"Starting training epoch {self.current_epoch + 1}...")
        self.train_button.setEnabled(False);
        self.infer_button.setEnabled(False)

        self.worker = Worker(self.model, self.data_loader, self.optimizer, self.criterion, mode="train")
        self._connect_worker_signals(self.worker.signals)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def start_inference(self):
        self.reset_animation_display()
        sample_idx = random.randint(0, len(self.dataset) - 1)
        input_ids, true_label, original_text = self.dataset[sample_idx]

        self.log_message(f"Starting inference for: {original_text}")
        self.current_sample_text_edit.setText(f"Input: {original_text}\nTrue Label: Class {true_label.item()}")

        self.train_button.setEnabled(False);
        self.infer_button.setEnabled(False)

        self.worker = Worker(self.model, None, None, None, mode="infer",
                             sample_to_infer=(input_ids, original_text, true_label))
        self._connect_worker_signals(self.worker.signals)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self):
        self.log_message("Worker finished.")
        self.train_button.setEnabled(True);
        self.infer_button.setEnabled(True)
        if hasattr(self.worker, 'pathway_stats') and self.worker.pathway_stats:
            self.pathway_frequencies.update(self.worker.pathway_stats)
            self.display_pathway_stats()

    def log_message(self, msg):
        self.log_text_edit.append(msg)

    def display_pathway_stats(self):
        path_stats_str = f"--- Pathway Frequencies (Total {sum(self.pathway_frequencies.values())}) ---\n"
        sorted_paths = sorted(self.pathway_frequencies.items(), key=lambda item: item[1], reverse=True)
        for path, count in sorted_paths[:10]:
            path_str = " -> ".join(map(str, path))
            path_stats_str += f"{path_str}: {count}\n"

        total_paths_logged = sum(self.pathway_frequencies.values())
        if total_paths_logged > 0:
            path_stats_str += "\n--- Potential Compilation Candidates (>{:.0f}% usage) ---\n".format(0.1 * 100)
            for path, count in sorted_paths:
                if count / total_paths_logged > 0.10:
                    path_str = " -> ".join(map(str, path))
                    path_stats_str += f"MERGE CANDIDATE: {path_str} ({count / total_paths_logged:.2%})\n"

        current_stats = self.stats_text_edit.toPlainText().split("--- Pathway Frequencies ---")[0]
        self.stats_text_edit.setText(current_stats.strip() + "\n" + path_stats_str)

    def on_input_text(self, text):
        self.diagram_widget.reset_highlights()
        self.diagram_widget.highlight_node("INPUT")
        self.current_sample_text_edit.setText(f"Input: {text}")

    def on_embedded(self, repr_preview):
        self.diagram_widget.highlight_node("INPUT", False)
        self.diagram_widget.highlight_connection("INPUT", "EMBED")
        self.diagram_widget.highlight_node("EMBED")
        self.diagram_widget.set_representation_text(repr_preview)

    def on_to_router(self, repr_preview, current_path):
        self.diagram_widget.highlight_node("EMBED", False)  # Assume came from embed or last expert
        # De-highlight previous expert if path is not empty
        if current_path:
            last_elem = current_path[-1]
            if isinstance(last_elem, int):  # It's an expert index
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
            self.diagram_widget.highlight_connection("ROUTER", "CLASSIFIER")
        else:
            self.diagram_widget.highlight_connection("ROUTER", f"EXPERT_{selected_index}")

    def on_to_expert(self, expert_idx, repr_preview):
        self.diagram_widget.highlight_node(f"EXPERT_{expert_idx}")
        self.diagram_widget.set_representation_text(repr_preview)

    def on_expert_output(self, expert_idx, repr_preview):
        # Keep expert highlighted until data moves to router
        self.diagram_widget.set_representation_text(repr_preview)

    def on_to_output_layer(self, repr_preview):
        self.diagram_widget.highlight_node("ROUTER", False)
        # De-highlight last expert if path ended by max_len
        # This part is tricky as we don't get full path here easily.
        # Simplification: assume termination came from router.
        self.diagram_widget.highlight_node("CLASSIFIER")
        self.diagram_widget.set_representation_text(repr_preview)

    def on_final_prediction_anim(self, prediction_idx, path_taken):
        self.diagram_widget.highlight_node("CLASSIFIER", True)
        path_str = " -> ".join(map(str, path_taken))
        self.diagram_widget.set_current_path_text(path_taken)
        current_text = self.current_sample_text_edit.toPlainText()
        self.current_sample_text_edit.setText(f"{current_text}\nPredicted: Class {prediction_idx}\nPath: {path_str}")

    def on_epoch_completed(self, epoch_num_ignored, loss, acc, pathway_freq_update):
        self.current_epoch += 1
        self.log_message(f"Epoch {self.current_epoch} completed. Loss: {loss:.4f}, Acc: {acc:.4f}")
        self.pathway_frequencies.update(pathway_freq_update)
        epoch_summary = f"Epoch {self.current_epoch}: Loss={loss:.4f}, Accuracy={acc:.4f}\n"
        existing_stats = self.stats_text_edit.toPlainText().split("--- Pathway Frequencies ---")[0]
        self.stats_text_edit.setText(epoch_summary + existing_stats.strip())
        self.display_pathway_stats()

    def on_inference_completed(self, input_text, pred_label, true_label, path_taken):
        self.log_message(
            f"Inference done for: {input_text}. Pred: {pred_label}, True: {true_label}, Path: {path_taken}")
        path_str = " -> ".join(map(str, path_taken))
        self.current_sample_text_edit.setText(
            f"Input: {input_text}\nTrue Label: {true_label}\nPredicted: {pred_label}\nPath: {path_str}")
        self.pathway_frequencies.update({tuple(path_taken): 1})
        self.display_pathway_stats()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        import torch
    except ImportError:
        # Fallback for systems without Qt message box easily available before app.exec_
        print("CRITICAL: PyTorch not found. Please install PyTorch to run this demo.")
        sys.exit(1)

    if torch.cuda.is_available():
        print("CUDA is available. Demo runs on CPU.")
    else:
        print("CUDA not available. Demo runs on CPU.")
    torch.set_default_device('cpu')  # Explicitly set to CPU

    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())