"""
FedAvg Training - CUDA Version
Uses centralized baseline model as initialization
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
"""
Advanced Model Architectures for AHU Fault Detection
LSTM + CNN + Multi-Head Attention + Residual Connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Multi-head self-attention layer"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class ResidualBlock(nn.Module):
    """Residual block with layer normalization"""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.norm2(self.fc2(x))
        x = self.dropout(x)
        return self.activation(x + residual)


class HybridLSTMCNNAttention(nn.Module):
    """
    Hybrid model: LSTM + CNN + Multi-Head Attention
    Target: Beat 96% F1-score
    ~5-10M parameters, CPU-friendly
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        embed_dim=256,
        cnn_channels=[128, 256, 384],
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        num_attention_heads=8,
        num_residual_blocks=2,
        dropout=0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # CNN Branch - Extract local patterns
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        for out_channels in cnn_channels:
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels

        # Global pooling for CNN
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)

        # LSTM Branch - Capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_dim = lstm_hidden_dim * 2  # bidirectional

        # Multi-head attention on LSTM output
        self.attention = AttentionLayer(
            embed_dim=lstm_output_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # Feature projection
        cnn_out_dim = cnn_channels[-1]
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.lstm_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Fusion with residual blocks
        fusion_dim = embed_dim * 2  # CNN + LSTM
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Residual blocks for deep processing
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(512, dropout=dropout) for _ in range(num_residual_blocks)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Input projection
        x_embed = self.input_proj(x)  # (batch, embed_dim)

        # CNN Branch
        x_cnn = x.unsqueeze(1)  # (batch, 1, input_dim)
        for cnn_layer in self.cnn_layers:
            x_cnn = cnn_layer(x_cnn)
        x_cnn = self.cnn_pool(x_cnn).squeeze(-1)  # (batch, cnn_channels[-1])
        x_cnn = self.cnn_proj(x_cnn)  # (batch, embed_dim)

        # LSTM Branch
        x_lstm = x_embed.unsqueeze(1)  # (batch, 1, embed_dim)
        lstm_out, _ = self.lstm(x_lstm)  # (batch, 1, lstm_output_dim)

        # Apply attention
        x_lstm = self.attention(lstm_out)  # (batch, 1, lstm_output_dim)
        x_lstm = x_lstm.squeeze(1)  # (batch, lstm_output_dim)
        x_lstm = self.lstm_proj(x_lstm)  # (batch, embed_dim)

        # Fusion
        x_fused = torch.cat([x_cnn, x_lstm], dim=1)  # (batch, embed_dim*2)
        x_fused = self.fusion(x_fused)  # (batch, 512)

        # Apply residual blocks
        for res_block in self.residual_blocks:
            x_fused = res_block(x_fused)

        # Classification
        logits = self.classifier(x_fused)

        return logits


# For backward compatibility and simpler usage
def create_model(input_dim, num_classes=4, model_type='hybrid', dropout=0.3):
    """
    Factory function to create models

    Args:
        input_dim: number of input features
        num_classes: number of output classes
        model_type: 'simple' or 'hybrid'
        dropout: dropout rate

    Returns:
        PyTorch model
    """
    if model_type == 'simple':
        # Simple MLP
        from train_simple_stable import SimpleStableModel
        return SimpleStableModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=[256, 128, 64],
            dropout=dropout
        )
    elif model_type == 'hybrid':
        # Hybrid LSTM+CNN+Attention
        return HybridLSTMCNNAttention(
            input_dim=input_dim,
            num_classes=num_classes,
            embed_dim=256,
            cnn_channels=[128, 256, 384],
            lstm_hidden_dim=256,
            lstm_num_layers=2,
            num_attention_heads=8,
            num_residual_blocks=2,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    input_dim = 15  # Typical number of features
    num_classes = 4

    model = HybridLSTMCNNAttention(input_dim=input_dim, num_classes=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    # Test forward pass
    x = torch.randn(32, input_dim)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Model works!")
# ============================================================================
# CONFIG
# ============================================================================

SPLITS_DIR = 'processed_splits_advanced/federated'
TEST_DIR = 'processed_splits_advanced/centralized'
BASELINE_MODEL_PATH = 'best_model_centralized_baseline.pth'

NUM_ROUNDS = 40
LOCAL_EPOCHS = 5
BATCH_SIZE = 128
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3


# ============================================================================
# DATASET
# ============================================================================

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.X = X.values.astype(np.float32)
        else:
            self.X = X.astype(np.float32)

        if isinstance(y, pd.Series):
            self.y = y.values.astype(np.int64)
        else:
            self.y = y.astype(np.int64)

        self.X = np.nan_to_num(self.X, nan=0.0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# ============================================================================
# MODEL
# ============================================================================
# Model is imported from hybrid_lstm.py (HybridLSTMCNNAttention)


# ============================================================================
# FUNCTIONS
# ============================================================================

def train_local(model, loader, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds.extend(outputs.max(1)[1].cpu().numpy())
            labels.extend(y.cpu().numpy())
    return preds, labels


def average_states(states, weights):
    total = sum(weights)
    avg = {}
    for key in states[0].keys():
        avg[key] = sum(states[i][key] * (weights[i] / total) for i in range(len(states)))
    return avg


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\nLoading data...")

    # Get AHUs
    ahus = sorted([d for d in os.listdir(SPLITS_DIR) if os.path.isdir(os.path.join(SPLITS_DIR, d))])
    print(f"AHUs: {ahus}")

    # Load
    clients = {}
    input_dim = None
    num_classes = None

    for ahu in ahus:
        train_df = pd.read_csv(os.path.join(SPLITS_DIR, ahu, 'train.csv'))
        val_df = pd.read_csv(os.path.join(SPLITS_DIR, ahu, 'val.csv'))

        # Drop label and any metadata columns
        ahu_cols = [col for col in train_df.columns if col.startswith('ahu_')]
        drop_cols = ['label'] + ahu_cols
        extra_cols = ['id', 'label_full', 'AHU_name']
        for col in extra_cols:
            if col in train_df.columns and col not in drop_cols:
                drop_cols.append(col)
        
        X_tr = train_df.drop(columns=drop_cols, errors='ignore')
        y_tr = train_df['label'].values
        X_val = val_df.drop(columns=drop_cols, errors='ignore')
        y_val = val_df['label'].values

        if input_dim is None:
            input_dim = X_tr.shape[1]
        
        if num_classes is None:
            num_classes = len(np.unique(y_tr))

        print(f"{ahu}: train={len(X_tr)}, val={len(X_val)}")

        clients[ahu] = {
            'train': DataLoader(SimpleDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=NUM_WORKERS, pin_memory=True),
            'val': DataLoader(SimpleDataset(X_val, y_val), batch_size=BATCH_SIZE, 
                             num_workers=NUM_WORKERS, pin_memory=True),
            'size': len(X_tr)
        }

    # Test
    test_df = pd.read_csv(os.path.join(TEST_DIR, 'test.csv'))
    # Drop same columns as training data
    ahu_cols = [col for col in test_df.columns if col.startswith('ahu_')]
    drop_cols = ['label'] + ahu_cols
    extra_cols = ['id', 'label_full', 'AHU_name']
    for col in extra_cols:
        if col in test_df.columns and col not in drop_cols:
            drop_cols.append(col)
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df['label'].values
    test_loader = DataLoader(SimpleDataset(X_test, y_test), batch_size=BATCH_SIZE, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test: {len(X_test)}")

    # Model - Use same architecture as centralized baseline
    print(f"\nCreating model...")
    
    # Try to infer num_classes from baseline model if it exists
    baseline_num_classes = None
    if os.path.exists(BASELINE_MODEL_PATH):
        print(f"Checking baseline model at {BASELINE_MODEL_PATH}...")
        try:
            state_dict = torch.load(BASELINE_MODEL_PATH, map_location=DEVICE)
            # Infer num_classes from the classifier output layer
            if 'classifier.8.weight' in state_dict:
                baseline_num_classes = state_dict['classifier.8.weight'].shape[0]
                print(f"  Baseline model has {baseline_num_classes} classes")
            elif 'classifier.2.weight' in state_dict:
                # Alternative: check if it's a different layer structure
                baseline_num_classes = state_dict['classifier.2.weight'].shape[0]
                print(f"  Baseline model has {baseline_num_classes} classes")
        except Exception as e:
            print(f"  Could not read baseline model: {e}")
    
    # Use baseline num_classes if available, otherwise use data's num_classes
    if baseline_num_classes is not None:
        model_num_classes = baseline_num_classes
        print(f"Using num_classes={model_num_classes} from baseline model")
    else:
        model_num_classes = num_classes
        print(f"Using num_classes={model_num_classes} from data")
    
    print(f"Input dim: {input_dim}, Num classes: {model_num_classes}")
    model = HybridLSTMCNNAttention(
        input_dim=input_dim,
        num_classes=model_num_classes,
        embed_dim=256,
        cnn_channels=[128, 256, 384],
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        num_attention_heads=8,
        num_residual_blocks=2,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Load centralized baseline model weights
    if os.path.exists(BASELINE_MODEL_PATH):
        print(f"Loading baseline model weights from {BASELINE_MODEL_PATH}...")
        try:
            state_dict = torch.load(BASELINE_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict, strict=True)
            print("✓ Baseline model loaded successfully!")
        except Exception as e:
            print(f"⚠ Warning: Could not load baseline model: {e}")
            print("  Starting from random initialization...")
    else:
        print(f"⚠ Warning: Baseline model not found at {BASELINE_MODEL_PATH}")
        print("  Starting from random initialization...")
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    print(f"\nTraining FedAvg for {NUM_ROUNDS} rounds...")

    best_f1 = 0
    best_state = None

    for r in range(NUM_ROUNDS):
        print(f"\nRound {r+1}/{NUM_ROUNDS}")

        states, weights = [], []

        for ahu, data in clients.items():
            # Local model - same architecture as global
            local = HybridLSTMCNNAttention(
                input_dim=input_dim,
                num_classes=model_num_classes,
                embed_dim=256,
                cnn_channels=[128, 256, 384],
                lstm_hidden_dim=256,
                lstm_num_layers=2,
                num_attention_heads=8,
                num_residual_blocks=2,
                dropout=DROPOUT
            ).to(DEVICE)
            local.load_state_dict(model.state_dict())

            # Train
            train_local(local, data['train'], LOCAL_EPOCHS)

            # Val
            val_preds, val_labels = evaluate(local, data['val'])
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
            print(f"  {ahu}: F1={val_f1*100:.1f}%")

            states.append(copy.deepcopy(local.state_dict()))
            weights.append(data['size'])
            del local

        # Aggregate
        model.load_state_dict(average_states(states, weights))

        # Test
        test_preds, test_labels = evaluate(model, test_loader)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
        test_f1_m = f1_score(test_labels, test_preds, average='macro', zero_division=0)

        print(f"  Global: Acc={test_acc*100:.2f}%, F1={test_f1*100:.2f}%, F1(M)={test_f1_m*100:.2f}%")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ✓ Best: {best_f1*100:.2f}%")

    # Final
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print("="*80)

    model.load_state_dict(best_state)
    preds, labels = evaluate(model, test_loader)

    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average='weighted', zero_division=0)
    f1_m = f1_score(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds)

    per_class = {}
    # Use model_num_classes which might be different from data's num_classes
    actual_num_classes = len(np.unique(labels))
    for i in range(actual_num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        per_class[f'class_{i}'] = {
            'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
            'precision': float(TP/(TP+FP)) if TP+FP>0 else 0.0,
            'recall': float(TP/(TP+FN)) if TP+FN>0 else 0.0,
            'f1': float(2*TP/(2*TP+FP+FN)) if 2*TP+FP+FN>0 else 0.0
        }

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 (Weighted): {f1_w*100:.2f}%")
    print(f"F1 (Macro): {f1_m*100:.2f}%")

    print(f"\nConfusion Matrix:")
    print(cm)

    print(f"\nPer-Class:")
    for cn, m in sorted(per_class.items()):
        print(f"  {cn}: TP={m['TP']:4d} TN={m['TN']:4d} FP={m['FP']:3d} FN={m['FN']:3d} | "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    print(f"\n{classification_report(labels, preds, zero_division=0)}")

    # Save
    results = {
        'final_metrics': {
            'accuracy': float(acc),
            'f1_weighted': float(f1_w),
            'f1_macro': float(f1_m),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class
        },
        'config': {'num_rounds': NUM_ROUNDS, 'local_epochs': LOCAL_EPOCHS}
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results_fedavg_{ts}.json', 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(best_state, f'model_fedavg_{ts}.pth')

    print(f"\n✓ Saved!")
    print("="*80)


if __name__ == "__main__":
    # Set seed for reproducibility
    import random
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {SEED} for reproducibility\n")
    
    print("="*80)
    print("FEDAVG - CUDA VERSION")
    print("="*80)

    # Use CUDA if available
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    main()

