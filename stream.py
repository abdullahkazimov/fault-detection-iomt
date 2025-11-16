"""
Real-Time Data Streaming Model Comparison Script
Compares Centralized Baseline, FedAvg, and Edge-Aware FL models
Shows predictions and running metrics per step
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ============================================================================
# MODEL ARCHITECTURE (Same as training)
# ============================================================================

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
    """Hybrid model: LSTM + CNN + Multi-Head Attention"""
    
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

        # CNN Branch
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

        self.cnn_pool = nn.AdaptiveAvgPool1d(1)

        # LSTM Branch
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_dim = lstm_hidden_dim * 2

        # Multi-head attention
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

        # Fusion
        fusion_dim = embed_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(512, dropout=dropout) for _ in range(num_residual_blocks)
        ])

        # Classifier
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

        self._init_weights()

    def _init_weights(self):
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
        # Input projection
        x_embed = self.input_proj(x)

        # CNN Branch
        x_cnn = x.unsqueeze(1)
        for cnn_layer in self.cnn_layers:
            x_cnn = cnn_layer(x_cnn)
        x_cnn = self.cnn_pool(x_cnn).squeeze(-1)
        x_cnn = self.cnn_proj(x_cnn)

        # LSTM Branch
        x_lstm = x_embed.unsqueeze(1)
        lstm_out, _ = self.lstm(x_lstm)
        x_lstm = self.attention(lstm_out)
        x_lstm = x_lstm.squeeze(1)
        x_lstm = self.lstm_proj(x_lstm)

        # Fusion
        x_fused = torch.cat([x_cnn, x_lstm], dim=1)
        x_fused = self.fusion(x_fused)

        # Residual blocks
        for res_block in self.residual_blocks:
            x_fused = res_block(x_fused)

        # Classification
        logits = self.classifier(x_fused)
        return logits


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths
CENTRALIZED_MODEL_PATH = 'best_model_centralized_baseline.pth'
FEDAVG_MODEL_PATH = 'model_fedavg_20251115_203130.pth'
EDGE_AWARE_MODEL_PATH = 'model_edge_aware_full_edge_aware_full_moderate_conservative_offload_network_based_20251115_194741.pth'

# Data path
TEST_DATA_PATH = 'processed_splits_advanced/centralized/test.csv'

# Model config
INPUT_DIM = 15  # Will be inferred from data
NUM_CLASSES = 4
DROPOUT = 0.3

# Streaming config
STREAM_BATCH_SIZE = 1  # Stream one sample at a time
UPDATE_INTERVAL = 100  # Print stats every N samples
DELAY_MS = 50  # Delay between samples (milliseconds) for visualization

# Labels
LABEL_NAMES = {
    0: 'Normal',
    1: 'RATSF (Return Air Temp Sensor Fault)',
    2: 'SFF (Supply Fan Fault)',
    3: 'VPF (Valve Position Fault)'
}


# ============================================================================
# MODEL LOADER
# ============================================================================

def load_model(model_path, input_dim, num_classes, device):
    """Load a trained model from checkpoint"""
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
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    """Track running metrics for each model"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.all_predictions = []
        self.all_labels = []
        self.correct = 0
        self.total = 0
        
    def update(self, pred, label):
        """Update with new prediction"""
        self.all_predictions.append(pred)
        self.all_labels.append(label)
        if pred == label:
            self.correct += 1
        self.total += 1
    
    def get_accuracy(self):
        """Get current accuracy"""
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def get_f1_weighted(self):
        """Get current weighted F1 score"""
        if self.total < 2:
            return 0.0
        return f1_score(self.all_labels, self.all_predictions, 
                       average='weighted', zero_division=0)
    
    def get_confusion_matrix(self):
        """Get confusion matrix"""
        if self.total < NUM_CLASSES:
            return None
        return confusion_matrix(self.all_labels, self.all_predictions)
    
    def get_per_class_metrics(self):
        """Get per-class precision, recall, F1"""
        if self.total < NUM_CLASSES:
            return {}
        
        cm = self.get_confusion_matrix()
        if cm is None:
            return {}
        
        metrics = {}
        for i in range(NUM_CLASSES):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - TP - FP - FN
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
            
            metrics[i] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(TP + FN)
            }
        
        return metrics


# ============================================================================
# REAL-TIME STREAMING
# ============================================================================

def print_header():
    """Print header for streaming output"""
    print("\n" + "="*120)
    print("REAL-TIME MODEL COMPARISON - AHU FAULT DETECTION")
    print("="*120)
    print(f"{'Sample':<8} {'True Label':<40} {'Centralized':<25} {'FedAvg':<25} {'Edge-Aware':<25}")
    print("-"*120)


def print_sample_prediction(idx, true_label, pred_centralized, pred_fedavg, pred_edge_aware):
    """Print prediction for a single sample"""
    true_name = LABEL_NAMES[true_label]
    
    # Color coding: ✓ for correct, ✗ for incorrect
    cent_symbol = "✓" if pred_centralized == true_label else "✗"
    fedavg_symbol = "✓" if pred_fedavg == true_label else "✗"
    edge_symbol = "✓" if pred_edge_aware == true_label else "✗"
    
    cent_name = LABEL_NAMES[pred_centralized]
    fedavg_name = LABEL_NAMES[pred_fedavg]
    edge_name = LABEL_NAMES[pred_edge_aware]
    
    print(f"{idx:<8} {true_name:<40} {cent_symbol} {cent_name:<22} {fedavg_symbol} {fedavg_name:<22} {edge_symbol} {edge_name:<22}")


def print_running_metrics(trackers, sample_idx):
    """Print running metrics for all models"""
    print("\n" + "="*120)
    print(f"RUNNING METRICS (after {sample_idx} samples)")
    print("="*120)
    
    # Table header
    print(f"{'Model':<25} {'Accuracy':<15} {'F1 (Weighted)':<15}")
    print("-"*120)
    
    for tracker in trackers.values():
        acc = tracker.get_accuracy() * 100
        f1 = tracker.get_f1_weighted() * 100
        print(f"{tracker.model_name:<25} {acc:>6.2f}%{'':<8} {f1:>6.2f}%{'':<8}")
    
    print("-"*120 + "\n")


def print_final_report(trackers):
    """Print final comprehensive report"""
    print("\n" + "="*120)
    print("FINAL REPORT - COMPLETE DATASET EVALUATION")
    print("="*120)
    
    for model_name, tracker in trackers.items():
        print(f"\n{model_name.upper()}")
        print("-"*120)
        
        acc = tracker.get_accuracy() * 100
        f1_weighted = tracker.get_f1_weighted() * 100
        
        print(f"Total Samples:     {tracker.total}")
        print(f"Accuracy:          {acc:.2f}%")
        print(f"F1 (Weighted):     {f1_weighted:.2f}%")
        
        # Per-class metrics
        per_class = tracker.get_per_class_metrics()
        if per_class:
            print(f"\nPer-Class Metrics:")
            print(f"{'Class':<8} {'Label':<45} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
            print("-"*120)
            for class_id, metrics in per_class.items():
                label_name = LABEL_NAMES[class_id]
                print(f"{class_id:<8} {label_name:<45} "
                      f"{metrics['precision']*100:>6.2f}%{'':<5} "
                      f"{metrics['recall']*100:>6.2f}%{'':<5} "
                      f"{metrics['f1']*100:>6.2f}%{'':<5} "
                      f"{metrics['support']:<10}")
        
        # Confusion matrix
        cm = tracker.get_confusion_matrix()
        if cm is not None:
            print(f"\nConfusion Matrix:")
            print(f"{'':>10}", end="")
            for i in range(NUM_CLASSES):
                print(f"Pred {i:<6}", end="")
            print()
            for i in range(NUM_CLASSES):
                print(f"True {i:<5}", end="")
                for j in range(NUM_CLASSES):
                    print(f"{cm[i, j]:<12}", end="")
                print()
    
    print("\n" + "="*120)


def main():
    """Main streaming function"""
    print("\n" + "="*120)
    print("INITIALIZING REAL-TIME MODEL COMPARISON")
    print("="*120)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load test data
    print(f"Loading test data from: {TEST_DATA_PATH}")
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Prepare features and labels
    ahu_cols = [col for col in test_df.columns if col.startswith('ahu_')]
    drop_cols = ['label'] + ahu_cols
    extra_cols = ['id', 'label_full', 'AHU_name']
    for col in extra_cols:
        if col in test_df.columns and col not in drop_cols:
            drop_cols.append(col)
    
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df['label'].values
    
    # Convert to numpy
    X_test = X_test.values.astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    input_dim = X_test.shape[1]
    num_samples = X_test.shape[0]
    
    print(f"Test samples:      {num_samples}")
    print(f"Input features:    {input_dim}")
    print(f"Number of classes: {NUM_CLASSES}")
    
    # Load models
    print(f"\nLoading models...")
    
    models = {}
    model_paths = {
        'Centralized Baseline': CENTRALIZED_MODEL_PATH,
        'FedAvg': FEDAVG_MODEL_PATH,
        'Edge-Aware FL': EDGE_AWARE_MODEL_PATH
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"  Loading {name}... ", end="")
            models[name] = load_model(path, input_dim, NUM_CLASSES, device)
            print("✓")
        else:
            print(f"  ✗ {name} not found at {path}")
            return
    
    # Initialize metrics trackers
    trackers = {name: MetricsTracker(name) for name in models.keys()}
    
    print(f"\n✓ All models loaded successfully!")
    print(f"\nStarting real-time streaming...")
    print(f"  - Streaming {num_samples} samples")
    print(f"  - Delay: {DELAY_MS}ms between samples")
    print(f"  - Update interval: every {UPDATE_INTERVAL} samples")
    
    # Print header
    print_header()
    
    # Stream data
    try:
        for idx in range(num_samples):
            # Get sample
            x = torch.from_numpy(X_test[idx:idx+1]).to(device)
            true_label = int(y_test[idx])
            
            # Get predictions from all models
            predictions = {}
            with torch.no_grad():
                for name, model in models.items():
                    output = model(x)
                    pred = output.max(1)[1].item()
                    predictions[name] = pred
                    trackers[name].update(pred, true_label)
            
            # Print prediction
            print_sample_prediction(
                idx + 1,
                true_label,
                predictions['Centralized Baseline'],
                predictions['FedAvg'],
                predictions['Edge-Aware FL']
            )
            
            # Print running metrics at intervals
            if (idx + 1) % UPDATE_INTERVAL == 0:
                print_running_metrics(trackers, idx + 1)
                print_header()  # Reprint header
            
            # Delay for visualization
            time.sleep(DELAY_MS / 1000.0)
    
    except KeyboardInterrupt:
        print("\n\nStreaming interrupted by user!")
    
    # Print final report
    print_final_report(trackers)
    
    print("\n✓ Real-time streaming completed!")
    print("="*120 + "\n")


if __name__ == "__main__":
    main()

