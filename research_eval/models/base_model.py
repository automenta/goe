import abc
import torch.nn as nn


class EvaluableModel(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_parameter_count(self) -> int:
        pass

    @abc.abstractmethod
    def get_model_name(self) -> str:
        pass

    def get_auxiliary_loss(self):  # Optional for models like MoE, GoE
        return 0.0

    def get_model_specific_metrics(self):  # Optional for model-specific stats
        return {}

    def get_trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)