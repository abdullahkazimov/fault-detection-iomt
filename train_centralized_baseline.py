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



"""
SIMPLE & STABLE Training Script
No NaN BS - Just results
Test on small dataset first, then scale up
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix



# ============================================================================
# CONFIGURATION
# ============================================================================

# Data
USE_SUBSET = False  # Start with subset for testing
SUBSET_FRACTION = 0.1  # Use 10% of data for testing
SPLITS_DIR = 'processed_splits_advanced'

# Training
BATCH_SIZE = 128
NUM_WORKERS = 4
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10

# Model - SIMPLE & STABLE
HIDDEN_DIMS = [256, 128, 64]  # Simple MLP
DROPOUT = 0.3

# Loss
USE_CLASS_WEIGHTS = True
WEIGHT_POWER = 1.5  # Conservative


# ============================================================================
# DATASET
# ============================================================================

class SimpleDataset(Dataset):
    """Simple, stable dataset"""
    def __init__(self, X, y):
        # Convert to float32 explicitly
        if isinstance(X, pd.DataFrame):
            self.X = X.values.astype(np.float32)
        else:
            self.X = X.astype(np.float32)

        if isinstance(y, pd.Series):
            self.y = y.values.astype(np.int64)
        else:
            self.y = y.astype(np.int64)

        # Check for NaN/Inf in data
        if np.isnan(self.X).any():
            print("WARNING: NaN in features, replacing with 0")
            self.X = np.nan_to_num(self.X, nan=0.0)

        if np.isinf(self.X).any():
            print("WARNING: Inf in features, clipping")
            self.X = np.clip(self.X, -1e6, 1e6)

        print(f"Dataset: {len(self.y)} samples, {self.X.shape[1]} features")
        print(f"  X range: [{self.X.min():.2f}, {self.X.max():.2f}]")
        print(f"  Class distribution: {dict(Counter(self.y))}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# ============================================================================
# MODEL - HYBRID LSTM + CNN + ATTENTION
# ============================================================================
# Model is imported from models.py


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def calculate_class_weights(y, power=1.5):
    """Calculate class weights"""
    class_counts = Counter(y)
    total = len(y)
    num_classes = len(class_counts)

    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = (total / (num_classes * count)) ** power
        weights.append(weight)

    # Normalize
    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes

    print(f"\nClass weights (power={power}):")
    for i, (w, count) in enumerate(zip(weights, [class_counts.get(i, 0) for i in range(num_classes)])):
        print(f"  Class {i}: weight={w:.3f}, count={count} ({count/total*100:.1f}%)")

    return torch.FloatTensor(weights)


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(x)

        # Check for NaN
        if torch.isnan(outputs).any():
            print(f"  ERROR: NaN in outputs at batch {batch_idx}")
            print(f"    Input range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"    Skipping batch...")
            continue

        loss = criterion(outputs, y)

        if torch.isnan(loss):
            print(f"  ERROR: NaN loss at batch {batch_idx}")
            continue

        # Backward
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Stats
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, f1_weighted, f1_macro, all_preds, all_labels


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print("="*80)

    # Load data
    cent_dir = os.path.join(SPLITS_DIR, 'centralized')

    train_df = pd.read_csv(os.path.join(cent_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(cent_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(cent_dir, 'test.csv'))

    # Use subset if testing
    if USE_SUBSET:
        print(f"\n⚠️  USING {SUBSET_FRACTION*100:.0f}% SUBSET FOR TESTING")
        train_df = train_df.sample(frac=SUBSET_FRACTION, random_state=42)
        val_df = val_df.sample(frac=SUBSET_FRACTION, random_state=42)
        test_df = test_df.sample(frac=SUBSET_FRACTION, random_state=42)

    # Drop label AND one-hot AHU columns (no data leakage!)
    ahu_cols = [col for col in train_df.columns if col.startswith('ahu_')]
    drop_cols = ['label'] + ahu_cols

    # Also drop other metadata if present
    extra_cols = ['id', 'label_full', 'AHU_name']
    for col in extra_cols:
        if col in train_df.columns and col not in drop_cols:
            drop_cols.append(col)

    print(f"\nDropping columns: {drop_cols}")

    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df['label'].values

    X_val = val_df.drop(columns=drop_cols, errors='ignore')
    y_val = val_df['label'].values

    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df['label'].values

    print(f"Actual features used: {list(X_train.columns)}")
    print(f"Number of features: {X_train.shape[1]}")

    print(f"\nData sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    # Create datasets
    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)
    test_dataset = SimpleDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Model
    print(f"\n{'='*80}")
    print("CREATING MODEL")
    print("="*80)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = HybridLSTMCNNAttention(
        input_dim=input_dim,
        num_classes=num_classes,
        embed_dim=256,
        cnn_channels=[128, 256, 384],
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        num_attention_heads=8,
        num_residual_blocks=2,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters (~{total_params/1e6:.1f}M)")
    print(f"Architecture: Hybrid LSTM + CNN + Attention")
    print(f"  - Input: {input_dim} features")
    print(f"  - CNN channels: [128, 256, 384]")
    print(f"  - LSTM: 256 hidden (bidirectional, 2 layers)")
    print(f"  - Attention heads: 8")
    print(f"  - Output: {num_classes} classes")

    # Loss & Optimizer
    if USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(y_train, power=WEIGHT_POWER).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    # Training
    print(f"\n{'='*80}")
    print("TRAINING")
    print("="*80)
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Patience: {PATIENCE}")

    best_f1 = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1_weighted': [],
        'val_f1_macro': []
    }

    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )

        # Validate
        val_loss, val_acc, val_f1_weighted, val_f1_macro, _, _ = evaluate(
            model, val_loader, criterion, DEVICE
        )

        # Update LR
        scheduler.step(val_f1_weighted)
        current_lr = optimizer.param_groups[0]['lr']

        # Print
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, "
              f"F1(W): {val_f1_weighted*100:.2f}%, F1(M): {val_f1_macro*100:.2f}%")
        print(f"LR: {current_lr:.2e}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_weighted'].append(val_f1_weighted)
        history['val_f1_macro'].append(val_f1_macro)

        # Check improvement
        if val_f1_weighted > best_f1:
            best_f1 = val_f1_weighted
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model_centralized_baseline.pth')
            print(f"✓ New best F1: {best_f1*100:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Testing
    print(f"\n{'='*80}")
    print("TESTING")
    print("="*80)

    # Load best model
    model.load_state_dict(torch.load('best_model_centralized_baseline.pth'))

    test_loss, test_acc, test_f1_weighted, test_f1_macro, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print("="*80)
    print(f"Accuracy:  {test_acc*100:.2f}%")
    print(f"F1 (Weighted): {test_f1_weighted*100:.2f}%")
    print(f"F1 (Macro):    {test_f1_macro*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

        print(f"  Class {i}: TP={TP:4d}, TN={TN:4d}, FP={FP:3d}, FN={FN:3d} | "
              f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, zero_division=0))

    # Save results
    results = {
        'accuracy': float(test_acc),
        'f1_weighted': float(test_f1_weighted),
        'f1_macro': float(test_f1_macro),
        'confusion_matrix': cm.tolist(),
        'history': history,
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'hidden_dims': HIDDEN_DIMS,
            'dropout': DROPOUT,
            'use_subset': USE_SUBSET,
            'subset_fraction': SUBSET_FRACTION if USE_SUBSET else 1.0
        }
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'results_simple_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Model saved to: best_model_centralized_baseline.pth")

    print(f"\n{'='*80}")
    if USE_SUBSET:
        print("✓ TEST ON SUBSET COMPLETED SUCCESSFULLY!")
        print(f"\nTo train on FULL dataset, set USE_SUBSET=False at line 22")
    else:
        print("✓ FULL TRAINING COMPLETED SUCCESSFULLY!")
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
    print(f"Random seed set to {SEED} for reproducibility")
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {DEVICE}")
    # Import the hybrid model
    print("="*80)
    print("CENTRALIZED TRAINING - HYBRID LSTM+CNN+ATTENTION")
    print("="*80)
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

