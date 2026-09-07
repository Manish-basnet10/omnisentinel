import torch
import torch.nn as nn

class PartialFlowMLP(nn.Module):
    """
    Fallback MLP for partial feature conditions.
    This model acts as a safe deterministic fallback when the required 60 features 
    cannot be constructed (even via feature engineering).
    """
    def __init__(self, input_size: int = 20, hidden_size: int = 64, num_classes: int = 15, dropout: float = 0.2):
        super(PartialFlowMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        x = self.relu1(self.fc1(x))
        x = self.drop(x)
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
