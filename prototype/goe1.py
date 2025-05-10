# Single-file Graph of Experts (GoE) Demo with PyTorch & PyQt5

import sys
import os
import re
import math
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QGroupBox, QTextEdit, QSpinBox, QDoubleSpinBox,
                             QFormLayout, QProgressBar, QSplitter)
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QTimer, QSizeF

# --- Configuration & Global Variables ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1
VOCAB = None  # To be built: {word: index}
ITOS = None  # To be built: [word1, word2, ...]

# --- Custom Dataset and Preprocessing (Replaces TorchText) ---
# (World, Sports, Business, Sci/Tech) -> Labels 0, 1, 2, 3
CUSTOM_DATASET_RAW = [
    ("UN chief urges global calm amid rising tensions politics", 0),
    ("Local team wins championship in thrilling final match", 1),
    ("Stock market hits new record high on tech optimism economy", 2),
    ("New breakthrough in AI could revolutionize computing science", 3),
    ("Peace talks scheduled next week in neutral venue diplomacy", 0),
    ("Star player scores hat trick leading team to victory game", 1),
    ("Company announces strong quarterly earnings growth report", 2),
    ("Scientists discover new exoplanet with potential for life astronomy", 3),
    ("International summit to discuss climate change policies environment", 0),
    ("Olympic trials begin for aspiring athletes competition", 1),
    ("Tech giant unveils its latest smartphone model innovation business", 2),  # Mix of Sci/Tech & Business
    ("Researchers develop faster algorithm for data processing tech", 3),
    ("Election results show a shift in political landscape vote", 0),
    ("Injury forces key player out of the upcoming game medical", 1),
    ("Startup secures major funding for innovative project investment", 2),
    ("Quantum computer achieves new computational milestone physics", 3),
    ("Global markets react to new trade agreement international", 0),
    ("Team prepares for crucial semi final tournament", 1),
    ("New software update promises enhanced security features", 2),  # Could be Sci/Tech too
    ("Exploring the mysteries of deep space exploration", 3)
]
NUM_CLASSES_CUSTOM = 4

# Shuffle dataset for better train/test split (optional, but good practice)
# import random
# random.seed(42)
# random.shuffle(CUSTOM_DATASET_RAW) # Shuffling here for consistency

TRAIN_DATA_RAW = CUSTOM_DATASET_RAW[:16]
TEST_DATA_RAW = CUSTOM_DATASET_RAW[16:]


def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Keep only alphanumeric and spaces
    return text.split()


def build_vocab_from_texts(texts, tokenizer_fn, min_freq=1):
    global VOCAB, ITOS, PAD_IDX, UNK_IDX
    token_counts = Counter()
    for text in texts:
        token_counts.update(tokenizer_fn(text))

    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    ITOS = [PAD_TOKEN, UNK_TOKEN]
    for token, freq in sorted_tokens:
        if freq >= min_freq:
            ITOS.append(token)

    VOCAB = {token: i for i, token in enumerate(ITOS)}
    PAD_IDX = VOCAB[PAD_TOKEN]
    UNK_IDX = VOCAB[UNK_TOKEN]
    return VOCAB, ITOS


class CustomTextDataset(Dataset):
    def __init__(self, raw_data, tokenizer_fn, vocab, max_seq_len):
        self.raw_data = raw_data
        self.tokenizer_fn = tokenizer_fn
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        text, label = self.raw_data[idx]
        tokens = self.tokenizer_fn(text)
        token_ids = [self.vocab.get(token, UNK_IDX) for token in tokens]
        token_ids = token_ids[:self.max_seq_len]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_batch_custom(batch):
    texts, labels = zip(*batch)
    labels_tensor = torch.stack(labels)
    texts_padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    return texts_padded.to(DEVICE), labels_tensor.to(DEVICE)


# --- PyTorch Model Components ---
class ExpertModule(nn.Module):
    def __init__(self, shared_dim, hidden_dim_multiplier=2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(shared_dim, shared_dim * hidden_dim_multiplier),
            nn.ReLU(),
            nn.Linear(shared_dim * hidden_dim_multiplier, shared_dim),
            nn.LayerNorm(shared_dim)
        )

    def forward(self, x):
        return x + self.layer(x)  # Residual connection


class RoutingController(nn.Module):
    def __init__(self, shared_dim, num_experts, hidden_dim_multiplier=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(shared_dim, shared_dim * hidden_dim_multiplier),
            nn.ReLU(),
            nn.Linear(shared_dim * hidden_dim_multiplier, num_experts + 1)  # +1 for termination
        )

    def forward(self, x):
        return self.layer(x)


# (Keep all other code the same, only replace the GoEModel.forward method)

# (Keep all other code the same, only replace the GoEModel.forward method)

class GoEModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_experts, num_classes,
                 max_steps=5, router_hidden_mult=1, expert_hidden_mult=2,
                 dropout_rate=0.1, gumbel_tau=1.0, padding_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.max_steps = max_steps
        self.gumbel_tau = gumbel_tau
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.padding_idx)
        self.embed_dropout = nn.Dropout(dropout_rate)

        self.experts = nn.ModuleList(
            [ExpertModule(embed_dim, expert_hidden_mult) for _ in range(num_experts)]
        )
        self.router = RoutingController(embed_dim, num_experts, router_hidden_mult)

        self.output_layer = nn.Linear(embed_dim, num_classes)

        self.current_visualization_pathway = []
        self.current_visualization_router_probs = []
        self.current_visualization_shared_repr_snippets = []

    def forward(self, input_ids, visualize_step_callback=None):
        batch_size = input_ids.size(0)

        # 1. Initial Embedding & Representation
        embedded = self.embedding(input_ids)
        mask = (input_ids != self.padding_idx).unsqueeze(-1).float()
        current_h = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        current_h = self.embed_dropout(current_h)  # (batch_size, embed_dim)

        if visualize_step_callback and batch_size == 1:
            self.current_visualization_pathway = []
            self.current_visualization_router_probs = []
            self.current_visualization_shared_repr_snippets = [current_h[0, :5].tolist()]
            visualize_step_callback(
                step_type="initial_representation",
                data={"representation_snippet": self.current_visualization_shared_repr_snippets[-1]}
            )

        all_router_logits_for_entropy = []
        pathway_taken_indices = [[-1] * self.max_steps for _ in range(batch_size)]

        # Mask for active samples (not yet terminated)
        # This mask will be recomputed cleanly each step, not modified inplace in a way that affects gradients
        active_samples_mask = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)

        # This will store the final representation for each sample once it terminates or hits max_steps
        final_output_representations = torch.zeros_like(current_h)

        for step in range(self.max_steps):
            if not torch.any(active_samples_mask):
                break  # All samples have terminated

            # Select representations of currently active samples
            active_h = current_h[active_samples_mask]  # This is a new tensor slice

            if active_h.size(0) == 0:  # Should be caught by torch.any(active_samples_mask)
                break

            # 2. Routing Decision for active samples
            router_logits = self.router(active_h)  # (num_active, num_experts + 1)
            all_router_logits_for_entropy.append(router_logits)  # For entropy regularization

            if self.training:
                # Gumbel-Softmax for differentiable selection
                action_probs_gumbel = F.gumbel_softmax(router_logits, tau=self.gumbel_tau, hard=True)
                chosen_actions_for_active = torch.argmax(action_probs_gumbel, dim=-1)  # (num_active)
            else:
                action_probs = F.softmax(router_logits, dim=-1)
                chosen_actions_for_active = torch.argmax(action_probs, dim=-1)  # (num_active)

                if visualize_step_callback and batch_size == 1:
                    # Find if the visualized sample is among the active ones
                    active_indices_global = torch.where(active_samples_mask)[0]
                    if 0 in active_indices_global:  # If the first sample (our viz sample) is active
                        idx_in_active = (active_indices_global == 0).nonzero(as_tuple=True)[0]
                        if idx_in_active.numel() > 0:  # Should always be true if 0 in active_indices_global
                            self.current_visualization_router_probs.append(action_probs[idx_in_active[0]].tolist())
                            visualize_step_callback(
                                step_type="router_decision",
                                data={"probs": self.current_visualization_router_probs[-1],
                                      "chosen_idx": chosen_actions_for_active[idx_in_active[0]].item()}
                            )

            # Update pathway log for all original batch samples that are currently active
            global_indices_of_active = torch.where(active_samples_mask)[0]
            for i, global_idx in enumerate(global_indices_of_active):
                pathway_taken_indices[global_idx][step] = chosen_actions_for_active[i].item()

            # 3. Process terminations and expert selections

            # Create a new tensor for the next step's representations
            next_h_parts = []  # Will gather parts to form the next current_h

            # Prepare a new active_mask for the *next* iteration based on current decisions
            next_active_samples_mask = active_samples_mask.clone()  # Start with current active state

            processed_expert_outputs = torch.zeros_like(active_h)  # To store outputs for those going to experts

            # Identify which of the *currently active* samples are terminating
            is_terminate_action_for_active = (chosen_actions_for_active == self.num_experts)

            # Update final_output_representations and next_active_samples_mask for those terminating
            terminating_global_indices = global_indices_of_active[is_terminate_action_for_active]
            if terminating_global_indices.numel() > 0:
                final_output_representations[terminating_global_indices] = active_h[is_terminate_action_for_active]
                next_active_samples_mask[terminating_global_indices] = False
                if visualize_step_callback and batch_size == 1 and 0 in terminating_global_indices:
                    self.current_visualization_pathway.append(self.num_experts)
                    visualize_step_callback(
                        step_type="expert_processing",
                        data={"expert_idx": self.num_experts, "is_terminate": True,
                              "pathway": self.current_visualization_pathway}
                    )

            # Identify which of the *currently active* samples are going to an expert
            to_expert_mask_in_active = ~is_terminate_action_for_active
            active_h_for_experts = active_h[to_expert_mask_in_active]
            chosen_expert_indices_for_step = chosen_actions_for_active[to_expert_mask_in_active]

            temp_next_h_for_active_non_terminating = torch.zeros_like(active_h_for_experts)

            if active_h_for_experts.size(0) > 0:
                for expert_j in range(self.num_experts):
                    # Select samples routed to expert_j from those going to experts
                    mask_for_this_expert = (chosen_expert_indices_for_step == expert_j)
                    if torch.any(mask_for_this_expert):
                        inputs_this_expert = active_h_for_experts[mask_for_this_expert]
                        outputs_this_expert = self.experts[expert_j](inputs_this_expert)
                        temp_next_h_for_active_non_terminating[mask_for_this_expert] = outputs_this_expert

                # Update the global current_h for samples that went through an expert
                # These are the samples that were active AND did not terminate
                global_indices_to_update = global_indices_of_active[to_expert_mask_in_active]
                current_h_temp_clone = current_h.clone()  # Clone to avoid inplace on original current_h for this step
                current_h_temp_clone[global_indices_to_update] = temp_next_h_for_active_non_terminating
                current_h = current_h_temp_clone  # current_h is now updated for the next iteration

                if visualize_step_callback and batch_size == 1:
                    if 0 in global_indices_to_update:  # If viz sample went to an expert
                        # Find its index within chosen_expert_indices_for_step
                        viz_sample_local_idx_in_to_expert_mask = \
                        (global_indices_of_active[to_expert_mask_in_active] == 0).nonzero(as_tuple=True)[0]
                        if viz_sample_local_idx_in_to_expert_mask.numel() > 0:
                            chosen_expert_for_viz = chosen_expert_indices_for_step[
                                viz_sample_local_idx_in_to_expert_mask[0]].item()
                            self.current_visualization_pathway.append(chosen_expert_for_viz)
                            self.current_visualization_shared_repr_snippets.append(
                                current_h[0, :5].tolist())  # Use updated current_h
                            visualize_step_callback(
                                step_type="expert_processing",
                                data={"expert_idx": chosen_expert_for_viz, "is_terminate": False,
                                      "representation_snippet": self.current_visualization_shared_repr_snippets[-1],
                                      "pathway": self.current_visualization_pathway}
                            )

            active_samples_mask = next_active_samples_mask  # This is now the mask for the next iteration

        # After loop, for any samples still active (hit max_steps), their final rep is the current_h
        if torch.any(active_samples_mask):
            final_output_representations[active_samples_mask] = current_h[active_samples_mask]
            if visualize_step_callback and batch_size == 1 and active_samples_mask[0]:
                self.current_visualization_pathway.append(self.num_experts)  # Forced
                self.current_visualization_shared_repr_snippets.append(current_h[0, :5].tolist())
                visualize_step_callback(
                    step_type="expert_processing",
                    data={"expert_idx": self.num_experts, "is_terminate": True, "forced": True,
                          "pathway": self.current_visualization_pathway,
                          "representation_snippet": self.current_visualization_shared_repr_snippets[-1]}
                )

        # 4. Final Output Layer
        output_logits = self.output_layer(final_output_representations)

        # Calculate entropy loss
        entropy_loss_val = 0  # Use different name
        if self.training and len(all_router_logits_for_entropy) > 0:
            step_entropies = []
            for step_logits in all_router_logits_for_entropy:
                if step_logits.numel() > 0:
                    probs = F.softmax(step_logits, dim=-1)
                    log_probs = F.log_softmax(step_logits, dim=-1)  # Use F.log_softmax for numerical stability
                    step_entropy = -torch.sum(probs * log_probs, dim=-1).mean()
                    step_entropies.append(step_entropy)
            if step_entropies:
                entropy_loss_val = torch.stack(step_entropies).mean()

        # Prepare pathways for output
        final_pathways_for_batch = []
        for sample_path in pathway_taken_indices:
            path = [p for p in sample_path if p != -1]
            if not path or path[-1] != self.num_experts:
                path.append(self.num_experts)
            final_pathways_for_batch.append(path)

        if visualize_step_callback and batch_size == 1:
            visualize_step_callback(
                step_type="final_output",
                data={"output_logits": output_logits[0].tolist(),
                      "final_pathway_from_model": final_pathways_for_batch[0]}  # Ensure this key matches GUI
            )
        return output_logits, entropy_loss_val, final_pathways_for_batch

# --- PyQt5 GUI Components ---
NODE_RADIUS = 30
ROUTER_NODE_RADIUS = 40
NODE_SPACING_X = 120
NODE_SPACING_Y = 100
TERMINATE_NODE_RADIUS = 25


class GoEWidget(QWidget):
    def __init__(self, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.setMinimumSize(600, 450)
        self.current_input_text = ""
        self.active_expert_idx = -1
        self.pathway_taken = []
        self.final_output_text = ""
        self.representation_snippet_text = ""

        self.expert_nodes_pos = []
        self.router_node_pos = QPointF(0, 0)
        self.terminate_node_pos = QPointF(0, 0)
        self.input_node_pos = QPointF(0, 0)
        self.output_node_pos = QPointF(0, 0)

        self.animation_step_data = None
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._perform_animation_step)
        self.animation_queue = []

    def update_config(self, num_experts):
        self.num_experts = num_experts
        self._calculate_node_positions()
        self.reset_visualization()
        self.update()

    def _calculate_node_positions(self):
        self.expert_nodes_pos = []
        center_x = self.width() / 2
        center_y = self.height() / 2

        self.input_node_pos = QPointF(center_x, center_y - NODE_SPACING_Y * 2)
        self.router_node_pos = QPointF(center_x, center_y - NODE_SPACING_Y * 0.8)

        total_expert_width = (self.num_experts - 1) * NODE_SPACING_X
        start_x_experts = center_x - total_expert_width / 2
        for i in range(self.num_experts):
            self.expert_nodes_pos.append(QPointF(start_x_experts + i * NODE_SPACING_X, center_y + NODE_SPACING_Y * 0.5))

        # Place terminate node to the right of experts
        if self.num_experts > 0:
            terminate_x = self.expert_nodes_pos[-1].x() + NODE_SPACING_X
        else:  # No experts, place it near router
            terminate_x = self.router_node_pos.x() + NODE_SPACING_X
        self.terminate_node_pos = QPointF(terminate_x, center_y + NODE_SPACING_Y * 0.5)

        self.output_node_pos = QPointF(center_x, center_y + NODE_SPACING_Y * 2.2)

    def resizeEvent(self, event):
        self._calculate_node_positions()
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.white)

        if not self.expert_nodes_pos and self.num_experts > 0:  # Check num_experts too
            self._calculate_node_positions()
        elif not self.router_node_pos.x():  # If not initialized at all
            self._calculate_node_positions()

        font = QFont("Arial", 10)
        painter.setFont(font)

        # Draw Input
        painter.setBrush(QColor("#e6f7ff"))  # Lighter blue
        painter.drawEllipse(self.input_node_pos, NODE_RADIUS, NODE_RADIUS)
        painter.drawText(
            QRectF(self.input_node_pos - QPointF(NODE_RADIUS, NODE_RADIUS), QSizeF(NODE_RADIUS * 2, NODE_RADIUS * 2)),
            Qt.AlignCenter, "Input")
        if self.current_input_text:
            painter.drawText(self.input_node_pos + QPointF(-NODE_RADIUS * 1.5, NODE_RADIUS + 5),
                             self.current_input_text[:20] + ("..." if len(self.current_input_text) > 20 else ""))

        # Draw Router
        router_color = QColor("#fff0b3")  # Light Gold
        if self.animation_step_data and self.animation_step_data.get("step_type") == "router_decision":
            router_color = QColor("#FFD700")  # Gold highlight
        painter.setBrush(router_color)
        painter.drawEllipse(self.router_node_pos, ROUTER_NODE_RADIUS, ROUTER_NODE_RADIUS)
        painter.drawText(QRectF(self.router_node_pos - QPointF(ROUTER_NODE_RADIUS, ROUTER_NODE_RADIUS),
                                QSizeF(ROUTER_NODE_RADIUS * 2, ROUTER_NODE_RADIUS * 2)), Qt.AlignCenter, "Router")
        if self.representation_snippet_text:
            painter.drawText(self.router_node_pos + QPointF(-ROUTER_NODE_RADIUS * 1.5, ROUTER_NODE_RADIUS + 15),
                             f"Rep: {self.representation_snippet_text}")

        # Draw Experts
        for i, pos in enumerate(self.expert_nodes_pos):
            expert_color = QColor("#d9ead3")  # Light green
            if self.active_expert_idx == i:
                expert_color = QColor("#90EE90")  # Active green
            if i in self.pathway_taken and self.active_expert_idx != i:
                expert_color = expert_color.darker(110)
            painter.setBrush(expert_color)
            painter.drawEllipse(pos, NODE_RADIUS, NODE_RADIUS)
            painter.drawText(QRectF(pos - QPointF(NODE_RADIUS, NODE_RADIUS), QSizeF(NODE_RADIUS * 2, NODE_RADIUS * 2)),
                             Qt.AlignCenter, f"E{i + 1}")

        # Draw Terminate Node
        terminate_color = QColor("#f8d7da")  # Light Pink
        if self.active_expert_idx == self.num_experts:
            terminate_color = QColor("#FF69B4")  # Hot Pink
        if self.num_experts in self.pathway_taken and self.active_expert_idx != self.num_experts:
            terminate_color = terminate_color.darker(110)
        painter.setBrush(terminate_color)
        painter.drawEllipse(self.terminate_node_pos, TERMINATE_NODE_RADIUS, TERMINATE_NODE_RADIUS)
        painter.drawText(QRectF(self.terminate_node_pos - QPointF(TERMINATE_NODE_RADIUS, TERMINATE_NODE_RADIUS),
                                QSizeF(TERMINATE_NODE_RADIUS * 2, TERMINATE_NODE_RADIUS * 2)), Qt.AlignCenter, "End")

        # Draw Output Node
        painter.setBrush(QColor("#e6f7ff"))
        painter.drawEllipse(self.output_node_pos, NODE_RADIUS, NODE_RADIUS)
        painter.drawText(
            QRectF(self.output_node_pos - QPointF(NODE_RADIUS, NODE_RADIUS), QSizeF(NODE_RADIUS * 2, NODE_RADIUS * 2)),
            Qt.AlignCenter, "Output")
        if self.final_output_text:
            painter.drawText(self.output_node_pos + QPointF(-NODE_RADIUS * 1.5, NODE_RADIUS + 15),
                             self.final_output_text)

        # Draw Connections
        pen = QPen(Qt.black)
        pen.setWidth(1)
        painter.setPen(pen)
        self._draw_arrow(painter, self.input_node_pos, self.router_node_pos, NODE_RADIUS, ROUTER_NODE_RADIUS)

        # Router to Experts/Terminate (Probabilities)
        if self.animation_step_data and self.animation_step_data.get("step_type") == "router_decision":
            probs = self.animation_step_data["data"]["probs"]
            chosen_idx = self.animation_step_data["data"]["chosen_idx"]

            targets = self.expert_nodes_pos + [self.terminate_node_pos]
            target_radii = [NODE_RADIUS] * self.num_experts + [TERMINATE_NODE_RADIUS]

            for i, prob in enumerate(probs):
                if i >= len(targets): continue  # Safety for num_experts change mismatch
                target_pos = targets[i]
                target_rad = target_radii[i]

                prob_pen = QPen(QColor(0, 0, 0, int(max(30, 255 * (prob ** 0.5)))))
                prob_pen.setWidth(1 + int(prob * 4))
                if i == chosen_idx:
                    prob_pen.setColor(Qt.red)
                    prob_pen.setWidth(max(2, prob_pen.width()))
                painter.setPen(prob_pen)
                self._draw_arrow(painter, self.router_node_pos, target_pos, ROUTER_NODE_RADIUS, target_rad)
                mid_point = (self.router_node_pos + target_pos) / 2
                painter.drawText(mid_point + QPointF(5, -5), f"{prob:.2f}")

        # Draw Pathway Taken
        path_pen = QPen(Qt.blue)
        path_pen.setWidth(2)  # Thinner pathway

        last_node_in_path_pos = self.router_node_pos  # Start from router for initial connection
        last_node_in_path_rad = ROUTER_NODE_RADIUS

        for k_step, expert_idx_in_path in enumerate(self.pathway_taken):
            painter.setPen(path_pen)  # Reset pen for each segment in case it was changed by prob_pen

            current_target_pos = None
            current_target_rad = None

            if expert_idx_in_path < self.num_experts:  # To an expert
                current_target_pos = self.expert_nodes_pos[expert_idx_in_path]
                current_target_rad = NODE_RADIUS
            else:  # To termination
                current_target_pos = self.terminate_node_pos
                current_target_rad = TERMINATE_NODE_RADIUS

            # Draw from last_node_pos to current_target_pos
            self._draw_arrow(painter, last_node_in_path_pos, current_target_pos, last_node_in_path_rad,
                             current_target_rad)

            if expert_idx_in_path < self.num_experts:  # If it went to an expert, next arrow is from this expert back to router
                # Draw dashed line from expert back to router
                dashed_pen = QPen(Qt.gray, 1, Qt.DashLine)
                painter.setPen(dashed_pen)
                self._draw_arrow(painter, current_target_pos, self.router_node_pos, NODE_RADIUS, ROUTER_NODE_RADIUS)
                last_node_in_path_pos = self.router_node_pos  # Next decision comes from router
                last_node_in_path_rad = ROUTER_NODE_RADIUS
            else:  # Terminated
                last_node_in_path_pos = current_target_pos  # Store terminate node as last
                last_node_in_path_rad = current_target_rad
                break  # Path ends here

        # Final connection to Output Node if pathway terminated
        if self.pathway_taken and self.pathway_taken[-1] == self.num_experts:
            painter.setPen(path_pen)  # Solid blue for final output connection
            # Draw from where it terminated (conceptually the router made the decision based on final state)
            # or from the termination node itself to the output. Let's use router.
            self._draw_arrow(painter, self.router_node_pos, self.output_node_pos, ROUTER_NODE_RADIUS, NODE_RADIUS)

    def _draw_arrow(self, painter, start_point, end_point, start_radius, end_radius):
        original_pen = painter.pen()  # Save to restore style if changed (e.g. for dashed lines)

        direction = end_point - start_point
        distance = math.sqrt(direction.x() ** 2 + direction.y() ** 2)
        if distance < 1e-6: return  # Too close, don't draw

        unit_direction = direction / distance

        adjusted_start = start_point + unit_direction * start_radius
        adjusted_end = end_point - unit_direction * end_radius

        # Check if points are too close after adjustment (can happen if nodes overlap)
        if (adjusted_end - adjusted_start).manhattanLength() < 2: return

        painter.drawLine(adjusted_start, adjusted_end)

        arrow_size = 8  # Smaller arrowhead
        angle = math.atan2(-unit_direction.y(), -unit_direction.x())

        p1 = adjusted_end + QPointF(math.sin(angle - math.pi / 6) * arrow_size,
                                    math.cos(angle - math.pi / 6) * arrow_size)
        p2 = adjusted_end + QPointF(math.sin(angle + math.pi / 6) * arrow_size,
                                    math.cos(angle + math.pi / 6) * arrow_size)

        arrow_head = QPolygonF()
        arrow_head.append(adjusted_end)
        arrow_head.append(p1)
        arrow_head.append(p2)

        # Save brush, set to pen color for filled arrowhead, then restore
        original_brush = painter.brush()
        painter.setBrush(painter.pen().color())
        painter.drawPolygon(arrow_head)
        painter.setBrush(original_brush)

        painter.setPen(original_pen)  # Restore original pen style

    def reset_visualization(self):
        self.current_input_text = ""
        self.active_expert_idx = -1
        self.pathway_taken = []
        self.final_output_text = ""
        self.representation_snippet_text = ""
        self.animation_step_data = None
        self.animation_queue = []
        if self.animation_timer.isActive():
            self.animation_timer.stop()
        self.update()

    def queue_animation_step(self, step_type, data):
        self.animation_queue.append({"step_type": step_type, "data": data})
        if not self.animation_timer.isActive():
            self.animation_timer.start(700)  # Animation step duration

    def _perform_animation_step(self):
        if not self.animation_queue:
            self.animation_timer.stop()
            if self.animation_step_data and self.animation_step_data.get("step_type") == "final_output":
                self.active_expert_idx = -1
                # Keep pathway_taken to show the final path
            # self.animation_step_data = None # Don't clear, so last probabilities remain visible
            self.update()
            return

        self.animation_step_data = self.animation_queue.pop(0)
        step_type = self.animation_step_data["step_type"]
        data = self.animation_step_data["data"]

        if step_type == "initial_representation":
            self.representation_snippet_text = str(["{:.2f}".format(x) for x in data["representation_snippet"]])
            self.active_expert_idx = -1
            self.pathway_taken = []  # Reset pathway at start of new inference
        elif step_type == "router_decision":
            self.active_expert_idx = -1  # Router is deciding
        elif step_type == "expert_processing":
            self.active_expert_idx = data["expert_idx"]
            self.pathway_taken = data["pathway"]  # Update pathway
            if not data["is_terminate"]:
                self.representation_snippet_text = str(["{:.2f}".format(x) for x in data["representation_snippet"]])
            else:
                self.representation_snippet_text = "Terminated."
                if data.get("forced"): self.representation_snippet_text = "Max steps!"
        elif step_type == "final_output":
            predicted_class = torch.tensor(data['output_logits']).argmax().item()
            self.final_output_text = f"Class: {predicted_class} (Logits: {[f'{x:.1f}' for x in data['output_logits'][:4]]})"
            if "final_pathway_from_model" in data:  # Ensure pathway is updated if model provided it
                self.pathway_taken = data["final_pathway_from_model"]
            self.active_expert_idx = self.num_experts  # Visually mark termination/end

        self.update()
        if not self.animation_queue:
            self.animation_timer.setInterval(1200)
        else:
            self.animation_timer.setInterval(700)


# Worker thread for PyTorch operations
class WorkerThread(QThread):
    progress_signal = pyqtSignal(int, float, float)
    log_signal = pyqtSignal(str)
    training_finished_signal = pyqtSignal(str)
    animation_step_signal = pyqtSignal(str, dict)
    inference_ready_signal = pyqtSignal()
    vocab_built_signal = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None
        self._is_training = False
        self._is_inferring = False
        self.text_to_infer = None

    def run(self):
        if self._is_training:
            self._run_training()
        elif self._is_inferring:
            self._run_inference_and_animation()

    def _prepare_vocab_and_data(self):
        global VOCAB, ITOS, PAD_IDX, UNK_IDX  # Ensure we're modifying globals
        if VOCAB is None:
            self.log_signal.emit("Building vocabulary from custom dataset...")
            all_train_texts = [text for text, label in TRAIN_DATA_RAW]
            VOCAB, ITOS = build_vocab_from_texts(all_train_texts, simple_tokenizer)
            PAD_IDX = VOCAB[PAD_TOKEN]  # Ensure these are set from the built vocab
            UNK_IDX = VOCAB[UNK_TOKEN]
            self.log_signal.emit(f"Vocabulary built. Size: {len(VOCAB)}")
            self.vocab_built_signal.emit()

        train_dataset = CustomTextDataset(TRAIN_DATA_RAW, simple_tokenizer, VOCAB, self.config["max_seq_len"])
        test_dataset = CustomTextDataset(TEST_DATA_RAW, simple_tokenizer, VOCAB, self.config["max_seq_len"])

        self.train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"],
                                       shuffle=True, collate_fn=collate_batch_custom)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config["batch_size"],
                                      collate_fn=collate_batch_custom)
        self.log_signal.emit("DataLoaders created.")

    def _setup_model_for_training(self):
        try:
            self._prepare_vocab_and_data()

            self.model = GoEModel(
                vocab_size=len(VOCAB),
                embed_dim=self.config["embed_dim"],
                num_experts=self.config["num_experts"],
                num_classes=NUM_CLASSES_CUSTOM,
                max_steps=self.config["max_steps"],
                gumbel_tau=self.config["gumbel_tau"],
                padding_idx=PAD_IDX
            ).to(DEVICE)
            self.log_signal.emit(f"Model created on {DEVICE}.")

            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=1e-4)
            self.criterion = nn.CrossEntropyLoss()
            return True
        except Exception as e:
            self.log_signal.emit(f"Error during model setup: {e}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
            self.training_finished_signal.emit(f"Error: {e}")
            return False

    def start_training(self):
        self._is_training = True
        self._is_inferring = False
        if not self.isRunning():
            self.start()

    def _run_training(self):
        if not self._setup_model_for_training():
            return

        self.log_signal.emit("Starting training...")
        entropy_weight = self.config["entropy_reg_weight"]

        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            total_loss, total_correct, total_samples = 0, 0, 0

            for i, (texts, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output_logits, entropy_loss, _ = self.model(texts)
                task_loss = self.criterion(output_logits, labels)
                loss = task_loss + entropy_weight * entropy_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item() * texts.size(0)  # loss.item is avg loss for batch
                total_correct += (output_logits.argmax(dim=1) == labels).sum().item()
                total_samples += texts.size(0)

                if (i + 1) % 5 == 0:  # Log more frequently for small dataset
                    avg_loss = total_loss / total_samples if total_samples > 0 else 0
                    accuracy = total_correct / total_samples if total_samples > 0 else 0
                    self.log_signal.emit(
                        f"E{epoch + 1}, B{i + 1}/{len(self.train_loader)}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, EntL: {entropy_loss.item():.4f}")

            avg_epoch_loss = total_loss / total_samples if total_samples > 0 else 0
            epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
            self.progress_signal.emit(epoch + 1, avg_epoch_loss, epoch_accuracy)

            val_loss, val_acc = self._evaluate()
            self.log_signal.emit(f"E{epoch + 1} Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        model_path = "goe_model_custom.pth"
        torch.save(self.model.state_dict(), model_path)
        self.training_finished_signal.emit(model_path)
        self._is_training = False

    def _evaluate(self):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for texts, labels in self.test_loader:  # Use self.test_loader
                output_logits, _, _ = self.model(texts)
                loss = self.criterion(output_logits, labels)

                total_loss += loss.item() * texts.size(0)
                total_correct += (output_logits.argmax(dim=1) == labels).sum().item()
                total_samples += texts.size(0)
        if total_samples == 0: return 0, 0
        return total_loss / total_samples, total_correct / total_samples

    def load_model(self, path):
        try:
            if VOCAB is None:
                self.log_signal.emit("Vocab not built. Building default vocab for model loading...")
                self._prepare_vocab_and_data()  # This will build/ensure VOCAB

            self.model = GoEModel(
                vocab_size=len(VOCAB),
                embed_dim=self.config["embed_dim"],
                num_experts=self.config["num_experts"],
                num_classes=NUM_CLASSES_CUSTOM,
                max_steps=self.config["max_steps"],
                padding_idx=PAD_IDX
            ).to(DEVICE)
            self.model.load_state_dict(torch.load(path, map_location=DEVICE))
            self.model.eval()
            self.log_signal.emit(f"Model loaded from {path} to {DEVICE}")
            self.inference_ready_signal.emit()
            return True
        except Exception as e:
            self.log_signal.emit(f"Error loading model: {e}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
            return False

    def start_inference_animation(self, text):
        self.text_to_infer = text
        self._is_inferring = True
        self._is_training = False
        if not self.isRunning():
            self.start()
        # else: if already running, this call will be queued or handled when current op finishes.

    def _run_inference_and_animation(self):
        if not self.model:
            self.log_signal.emit("Model not loaded for inference.")
            self.animation_step_signal.emit("final_output",
                                            {"output_logits": [0] * NUM_CLASSES_CUSTOM, "error": "Model not loaded"})
            self._is_inferring = False
            return

        if not self.text_to_infer or VOCAB is None:
            self.log_signal.emit("Text or vocab not available.")
            self.animation_step_signal.emit("final_output",
                                            {"output_logits": [0] * NUM_CLASSES_CUSTOM, "error": "Input/Vocab error"})
            self._is_inferring = False
            return

        self.model.eval()
        tokens = simple_tokenizer(self.text_to_infer)
        token_ids = [VOCAB.get(token, UNK_IDX) for token in tokens][:self.config["max_seq_len"]]
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(DEVICE)

        def animation_callback(step_type, data):
            self.animation_step_signal.emit(step_type, data)
            # QThread.msleep(50) # Optional small delay, GUI timer is main pacer

        with torch.no_grad():
            output_logits, _, pathway = self.model(input_tensor, visualize_step_callback=animation_callback)

        self._is_inferring = False
        self.text_to_infer = None
        self.inference_ready_signal.emit()  # Signal that it's ready for another inference


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph of Experts Demo (Single File)")
        self.setGeometry(100, 100, 1200, 800)

        self.config = {
            "num_experts": 3, "embed_dim": 32, "max_steps": 4,  # Smaller for faster demo
            "max_seq_len": 32, "batch_size": 8, "learning_rate": 2e-3,
            "num_epochs": 15, "entropy_reg_weight": 0.005, "gumbel_tau": 1.0
        }

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()

        form_layout = QFormLayout()
        self.num_experts_spin = QSpinBox();
        self.num_experts_spin.setRange(1, 5);
        self.num_experts_spin.setValue(self.config["num_experts"])
        form_layout.addRow("Num Experts:", self.num_experts_spin)
        self.embed_dim_spin = QSpinBox();
        self.embed_dim_spin.setRange(16, 128);
        self.embed_dim_spin.setValue(self.config["embed_dim"])
        form_layout.addRow("Embedding Dim:", self.embed_dim_spin)
        self.max_steps_spin = QSpinBox();
        self.max_steps_spin.setRange(1, 10);
        self.max_steps_spin.setValue(self.config["max_steps"])
        form_layout.addRow("Max Steps:", self.max_steps_spin)
        self.lr_spin = QDoubleSpinBox();
        self.lr_spin.setRange(1e-4, 1e-2);
        self.lr_spin.setDecimals(4);
        self.lr_spin.setSingleStep(1e-4);
        self.lr_spin.setValue(self.config["learning_rate"])
        form_layout.addRow("Learning Rate:", self.lr_spin)
        self.epochs_spin = QSpinBox();
        self.epochs_spin.setRange(1, 100);
        self.epochs_spin.setValue(self.config["num_epochs"])
        form_layout.addRow("Epochs:", self.epochs_spin)
        self.entropy_spin = QDoubleSpinBox();
        self.entropy_spin.setRange(0, 0.1);
        self.entropy_spin.setDecimals(4);
        self.entropy_spin.setSingleStep(0.001);
        self.entropy_spin.setValue(self.config["entropy_reg_weight"])
        form_layout.addRow("Entropy Reg:", self.entropy_spin)
        controls_layout.addLayout(form_layout)

        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        controls_layout.addWidget(self.train_button)

        self.load_model_button = QPushButton("Load Model (goe_model_custom.pth)")
        self.load_model_button.clicked.connect(self.load_model)
        controls_layout.addWidget(self.load_model_button)
        self.progress_bar = QProgressBar()
        controls_layout.addWidget(self.progress_bar)

        inference_group = QGroupBox("Inference & Animation");
        inference_layout = QVBoxLayout()
        self.input_text_edit = QLineEdit("New AI model released by tech company")
        inference_layout.addWidget(QLabel("Input Text:"));
        inference_layout.addWidget(self.input_text_edit)
        self.infer_button = QPushButton("Infer & Animate");
        self.infer_button.clicked.connect(self.run_inference_animation)
        self.infer_button.setEnabled(False)
        inference_layout.addWidget(self.infer_button)
        controls_layout.addWidget(inference_group)
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)

        right_splitter = QSplitter(Qt.Vertical)
        self.goe_widget = GoEWidget(self.config["num_experts"])
        right_splitter.addWidget(self.goe_widget)
        self.log_text_edit = QTextEdit();
        self.log_text_edit.setReadOnly(True)
        right_splitter.addWidget(self.log_text_edit)
        right_splitter.setSizes([550, 250])

        main_layout.addWidget(controls_group, 1)
        main_layout.addWidget(right_splitter, 3)

        self.worker_thread = WorkerThread(self.config)  # Pass initial config
        self.worker_thread.log_signal.connect(self.log_message)
        self.worker_thread.progress_signal.connect(self.update_progress)
        self.worker_thread.training_finished_signal.connect(self.training_finished)
        self.worker_thread.animation_step_signal.connect(self.goe_widget.queue_animation_step)
        self.worker_thread.inference_ready_signal.connect(lambda: self.infer_button.setEnabled(True))
        self.worker_thread.vocab_built_signal.connect(self._on_vocab_built)

        self.num_experts_spin.valueChanged.connect(self._update_config_from_gui_and_widget)
        # Other spins could also call _update_config_from_gui if their change needs immediate reflection outside training start

        self._update_config_from_gui()  # Call once to sync worker_thread.config

    def _on_vocab_built(self):
        self.log_message("MainWindow notified: Vocab built.")
        # Potentially enable inference button if a model might be loadable now
        # For now, loading explicitly builds vocab if needed.

    def _update_config_from_gui(self):
        self.config["num_experts"] = self.num_experts_spin.value()
        self.config["embed_dim"] = self.embed_dim_spin.value()
        self.config["max_steps"] = self.max_steps_spin.value()
        self.config["learning_rate"] = self.lr_spin.value()
        self.config["num_epochs"] = self.epochs_spin.value()
        self.config["entropy_reg_weight"] = self.entropy_spin.value()
        if self.worker_thread:  # Update worker's config
            self.worker_thread.config = self.config.copy()

    def _update_config_from_gui_and_widget(self):
        old_num_experts = self.config.get("num_experts")
        self._update_config_from_gui()
        if self.config["num_experts"] != old_num_experts:
            self.goe_widget.update_config(self.config["num_experts"])

    def log_message(self, message):
        self.log_text_edit.append(message)

    def update_progress(self, epoch, loss, acc):
        self.progress_bar.setValue(int((epoch / self.config["num_epochs"]) * 100))
        self.log_message(f"E{epoch} Done - AvgLoss: {loss:.4f}, AvgAcc: {acc:.4f}")

    def start_training(self):
        self._update_config_from_gui()
        self.train_button.setEnabled(False);
        self.load_model_button.setEnabled(False);
        self.infer_button.setEnabled(False)
        self.log_message("Training initiated...");
        self.progress_bar.setValue(0)
        self.worker_thread.start_training()

    def training_finished(self, msg):
        self.log_message(f"Training finished: {msg}")
        self.train_button.setEnabled(True);
        self.load_model_button.setEnabled(True)
        if not msg.startswith("Error"): self.infer_button.setEnabled(True)
        self.progress_bar.setValue(100 if not msg.startswith("Error") else 0)

    def load_model(self):
        self._update_config_from_gui()  # Ensure config used for model loading matches GUI
        model_path = "goe_model_custom.pth"
        self.log_message(f"Attempting to load model {model_path}...")
        self.infer_button.setEnabled(False)  # Disable while attempting load
        if self.worker_thread.load_model(model_path):
            pass  # inference_ready_signal will enable button
        else:
            self.log_message("Model loading failed (check logs).")
            # Re-enable load button, but infer stays disabled if load failed
            self.load_model_button.setEnabled(True)

    def run_inference_animation(self):
        self.goe_widget.reset_visualization()
        input_text = self.input_text_edit.text()
        if not input_text: self.log_message("Enter text for inference."); return

        self.goe_widget.current_input_text = input_text
        self.goe_widget.update()

        self.log_message(f"Inferring: \"{input_text}\"")
        self.infer_button.setEnabled(False)
        self.worker_thread.start_inference_animation(input_text)
        # inference_ready_signal from worker will re-enable button

    def closeEvent(self, event):
        if self.worker_thread.isRunning():
            self.log_message("Waiting for worker thread to finish...")
            self.worker_thread.quit()  # Request termination
            if not self.worker_thread.wait(3000):  # Wait up to 3s
                self.log_message("Worker thread did not stop in time. Forcing.")
                self.worker_thread.terminate()  # Force if necessary
                self.worker_thread.wait()  # Wait for forced termination
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())