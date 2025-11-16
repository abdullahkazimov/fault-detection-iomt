"""
Real-Time Data Streaming Model Comparison Script with Progressive Plots
Compares Centralized Baseline, FedAvg, and Edge-Aware FL models
Generates plots at checkpoints during streaming
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for background plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class AttentionLayer(nn.Module):
    """Multi-head self-attention layer"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
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
    
    def __init__(self, input_dim, num_classes, embed_dim=256, cnn_channels=[128, 256, 384],
                 lstm_hidden_dim=256, lstm_num_layers=2, num_attention_heads=8,
                 num_residual_blocks=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout)
        )

        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        for out_channels in cnn_channels:
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels), nn.GELU(), nn.Dropout(dropout)
            ))
            in_channels = out_channels

        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers,
            batch_first=True, dropout=dropout if lstm_num_layers > 1 else 0, bidirectional=True
        )

        lstm_output_dim = lstm_hidden_dim * 2
        self.attention = AttentionLayer(embed_dim=lstm_output_dim, num_heads=num_attention_heads, dropout=dropout)

        cnn_out_dim = cnn_channels[-1]
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.lstm_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout)
        )

        fusion_dim = embed_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout)
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(512, dropout=dropout) for _ in range(num_residual_blocks)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
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
        x_embed = self.input_proj(x)
        x_cnn = x.unsqueeze(1)
        for cnn_layer in self.cnn_layers:
            x_cnn = cnn_layer(x_cnn)
        x_cnn = self.cnn_pool(x_cnn).squeeze(-1)
        x_cnn = self.cnn_proj(x_cnn)

        x_lstm = x_embed.unsqueeze(1)
        lstm_out, _ = self.lstm(x_lstm)
        x_lstm = self.attention(lstm_out)
        x_lstm = x_lstm.squeeze(1)
        x_lstm = self.lstm_proj(x_lstm)

        x_fused = torch.cat([x_cnn, x_lstm], dim=1)
        x_fused = self.fusion(x_fused)
        for res_block in self.residual_blocks:
            x_fused = res_block(x_fused)
        logits = self.classifier(x_fused)
        return logits


# ============================================================================
# CONFIGURATION
# ============================================================================

CENTRALIZED_MODEL_PATH = 'best_model_centralized_baseline.pth'
FEDAVG_MODEL_PATH = 'model_fedavg_20251115_203130.pth'
EDGE_AWARE_MODEL_PATH = 'model_edge_aware_full_edge_aware_full_moderate_conservative_offload_network_based_20251115_194741.pth'
TEST_DATA_PATH = 'test.csv'

INPUT_DIM = 15
NUM_CLASSES = 4
DROPOUT = 0.3
PLOT_DIR = 'plots'

UPDATE_INTERVAL = 1000  # Generate plots every 1000 samples
PRINT_INTERVAL = 1  # Print every sample (set to 10 for every 10th sample)
DELAY_MS = 0  # No delay for fast processing

LABEL_NAMES = {
    0: 'Normal',
    1: 'RATSF',
    2: 'SFF',
    3: 'VPF'
}

LABEL_NAMES_FULL = {
    0: 'Normal Condition',
    1: 'Return Air Temp Sensor Fault',
    2: 'Supply Fan Fault',
    3: 'Valve Position Fault'
}

MODEL_COLORS = {
    'Centralized Baseline': '#2E86AB',
    'FedAvg': '#A23B72',
    'Edge-Aware FL': '#F18F01'
}


# ============================================================================
# MODEL LOADER
# ============================================================================

def load_model(model_path, input_dim, num_classes, device):
    model = HybridLSTMCNNAttention(
        input_dim=input_dim, num_classes=num_classes, embed_dim=256,
        cnn_channels=[128, 256, 384], lstm_hidden_dim=256, lstm_num_layers=2,
        num_attention_heads=8, num_residual_blocks=2, dropout=DROPOUT
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.all_predictions = []
        self.all_labels = []
        self.correct = 0
        self.total = 0
        
        # Streaming history
        self.accuracy_history = []
        self.f1_history = []
        self.sample_indices = []
        
    def update(self, pred, label):
        self.all_predictions.append(pred)
        self.all_labels.append(label)
        if pred == label:
            self.correct += 1
        self.total += 1
        
        # Update history
        self.sample_indices.append(self.total)
        self.accuracy_history.append(self.correct / self.total)
        if self.total >= 2:
            f1 = f1_score(self.all_labels, self.all_predictions, 
                         average='weighted', zero_division=0)
            self.f1_history.append(f1)
        else:
            self.f1_history.append(0.0)
    
    def get_accuracy(self):
        return self.correct / self.total if self.total > 0 else 0.0
    
    def get_f1_weighted(self):
        if self.total < 2:
            return 0.0
        return f1_score(self.all_labels, self.all_predictions, 
                       average='weighted', zero_division=0)
    
    def get_f1_macro(self):
        if self.total < 2:
            return 0.0
        return f1_score(self.all_labels, self.all_predictions, 
                       average='macro', zero_division=0)
    
    def get_confusion_matrix(self):
        if self.total < NUM_CLASSES:
            return None
        return confusion_matrix(self.all_labels, self.all_predictions)
    
    def get_per_class_metrics(self):
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
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
            metrics[i] = {
                'precision': precision, 'recall': recall, 'f1': f1,
                'support': int(TP + FN)
            }
        return metrics


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_plots_directory():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(f'{PLOT_DIR}/checkpoints', exist_ok=True)
    print(f"✓ Plots directory: {PLOT_DIR}/")


def plot_checkpoint_curves(trackers, checkpoint_num, total_samples):
    """Plot real-time curves at checkpoint"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Accuracy
    ax1 = axes[0]
    for name, tracker in trackers.items():
        ax1.plot(tracker.sample_indices, 
                [a * 100 for a in tracker.accuracy_history],
                label=name, linewidth=2.5, alpha=0.8,
                color=MODEL_COLORS[name])
    
    ax1.set_xlabel('Number of Samples Processed', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Real-Time Streaming Accuracy (Checkpoint {checkpoint_num})', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 105])
    
    # F1 Score
    ax2 = axes[1]
    for name, tracker in trackers.items():
        ax2.plot(tracker.sample_indices,
                [f * 100 for f in tracker.f1_history],
                label=name, linewidth=2.5, alpha=0.8,
                color=MODEL_COLORS[name])
    
    ax2.set_xlabel('Number of Samples Processed', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Weighted F1 Score (%)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Real-Time Streaming F1 Score (Checkpoint {checkpoint_num})',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/checkpoints/checkpoint_{checkpoint_num:03d}_curves.png', 
                dpi=200, bbox_inches='tight')
    plt.close()


def plot_checkpoint_comparison(trackers, checkpoint_num):
    """Plot comparison bars at checkpoint"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    model_names = list(trackers.keys())
    
    # Accuracy
    ax1 = axes[0]
    accuracies = [trackers[name].get_accuracy() * 100 for name in model_names]
    bars = ax1.bar(model_names, accuracies,
                   color=[MODEL_COLORS[name] for name in model_names],
                   edgecolor='black', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Accuracy Comparison (Checkpoint {checkpoint_num})', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.tick_params(axis='x', rotation=15)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    # F1 Score
    ax2 = axes[1]
    f1_scores = [trackers[name].get_f1_weighted() * 100 for name in model_names]
    bars = ax2.bar(model_names, f1_scores,
                   color=[MODEL_COLORS[name] for name in model_names],
                   edgecolor='black', linewidth=2, alpha=0.8)
    ax2.set_ylabel('Weighted F1 (%)', fontsize=13, fontweight='bold')
    ax2.set_title(f'F1 Score Comparison (Checkpoint {checkpoint_num})', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim([0, 105])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', rotation=15)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/checkpoints/checkpoint_{checkpoint_num:03d}_comparison.png', 
                dpi=200, bbox_inches='tight')
    plt.close()


def plot_final_comprehensive(trackers):
    """Generate all final comprehensive plots"""
    
    # 1. Real-time curves with zoomed sections
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Main accuracy curve
    ax1 = fig.add_subplot(gs[0, :])
    for name, tracker in trackers.items():
        ax1.plot(tracker.sample_indices, 
                [a * 100 for a in tracker.accuracy_history],
                label=name, linewidth=3, alpha=0.8,
                color=MODEL_COLORS[name])
    ax1.set_xlabel('Samples Processed', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Complete Streaming Accuracy Evolution', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.legend(fontsize=13, loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Main F1 curve
    ax2 = fig.add_subplot(gs[1, :])
    for name, tracker in trackers.items():
        ax2.plot(tracker.sample_indices,
                [f * 100 for f in tracker.f1_history],
                label=name, linewidth=3, alpha=0.8,
                color=MODEL_COLORS[name])
    ax2.set_xlabel('Samples Processed', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Weighted F1 Score (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Complete Streaming F1 Score Evolution',
                  fontsize=18, fontweight='bold', pad=20)
    ax2.legend(fontsize=13, loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Final metrics bars
    ax3 = fig.add_subplot(gs[2, 0])
    model_names = list(trackers.keys())
    accuracies = [trackers[name].get_accuracy() * 100 for name in model_names]
    bars = ax3.bar(model_names, accuracies,
                   color=[MODEL_COLORS[name] for name in model_names],
                   edgecolor='black', linewidth=2, alpha=0.8)
    ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Final Accuracy', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 105])
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=15)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[2, 1])
    f1_scores = [trackers[name].get_f1_weighted() * 100 for name in model_names]
    bars = ax4.bar(model_names, f1_scores,
                   color=[MODEL_COLORS[name] for name in model_names],
                   edgecolor='black', linewidth=2, alpha=0.8)
    ax4.set_ylabel('Weighted F1 (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Final F1 Score', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 105])
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=15)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    plt.suptitle('Real-Time Model Comparison - Complete Analysis',
                fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/01_complete_streaming_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_complete_streaming_analysis.png")
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for idx, (name, tracker) in enumerate(trackers.items()):
        cm = tracker.get_confusion_matrix()
        if cm is None:
            continue
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'},
                   xticklabels=[LABEL_NAMES[i] for i in range(NUM_CLASSES)],
                   yticklabels=[LABEL_NAMES[i] for i in range(NUM_CLASSES)],
                   linewidths=1.5, linecolor='white')
        ax.set_title(f'{name}\nAcc: {tracker.get_accuracy()*100:.2f}% | F1: {tracker.get_f1_weighted()*100:.2f}%',
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
    
    plt.suptitle('Confusion Matrices - Final Results', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/02_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_confusion_matrices.png")
    
    # 3. Radar Chart
    categories = ['Accuracy', 'Weighted F1', 'Macro F1', 'Class 0 F1', 'Class 1 F1']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    for name, tracker in trackers.items():
        per_class = tracker.get_per_class_metrics()
        values = [
            tracker.get_accuracy() * 100,
            tracker.get_f1_weighted() * 100,
            tracker.get_f1_macro() * 100,
            per_class.get(0, {}).get('f1', 0) * 100,
            per_class.get(1, {}).get('f1', 0) * 100
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=3, label=name,
               color=MODEL_COLORS[name], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=MODEL_COLORS[name])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title('Multi-Dimensional Performance Analysis',
                fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=13, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/03_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_radar_chart.png")
    
    # 4. Per-Class Performance
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    for class_id in range(NUM_CLASSES):
        ax = axes[class_id]
        model_names = list(trackers.keys())
        
        precisions, recalls, f1s = [], [], []
        for name in model_names:
            metrics = trackers[name].get_per_class_metrics()
            class_metric = metrics.get(class_id, {})
            precisions.append(class_metric.get('precision', 0) * 100)
            recalls.append(class_metric.get('recall', 0) * 100)
            f1s.append(class_metric.get('f1', 0) * 100)
        
        x = np.arange(len(model_names))
        width = 0.25
        
        ax.bar(x - width, precisions, width, label='Precision',
              color='#6A4C93', alpha=0.8, edgecolor='black')
        ax.bar(x, recalls, width, label='Recall',
              color='#1982C4', alpha=0.8, edgecolor='black')
        ax.bar(x + width, f1s, width, label='F1 Score',
              color='#8AC926', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Class {class_id}: {LABEL_NAMES_FULL[class_id]}',
                    fontsize=13, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 105])
    
    plt.suptitle('Per-Class Performance Metrics', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/04_per_class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_per_class_performance.png")
    
    # 5. Model Agreement Analysis
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    model_names = list(trackers.keys())
    preds = {name: np.array(trackers[name].all_predictions) for name in model_names}
    labels = np.array(trackers[model_names[0]].all_labels)
    
    # Agreement matrix
    ax1 = axes[0]
    agreement_matrix = np.zeros((len(model_names), len(model_names)))
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                agreement_matrix[i, j] = 100.0
            else:
                agreement = np.mean(preds[name1] == preds[name2]) * 100
                agreement_matrix[i, j] = agreement
    
    sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
               xticklabels=model_names, yticklabels=model_names,
               ax=ax1, vmin=0, vmax=100, cbar_kws={'label': 'Agreement (%)'},
               linewidths=2, linecolor='white')
    ax1.set_title('Model Agreement Matrix', fontsize=14, fontweight='bold', pad=15)
    
    # Agreement stats
    ax2 = axes[1]
    correct_all = sum([np.all([preds[name][i] == labels[i] for name in model_names])
                      for i in range(len(labels))])
    correct_some = sum([np.any([preds[name][i] == labels[i] for name in model_names]) and
                       not np.all([preds[name][i] == labels[i] for name in model_names])
                       for i in range(len(labels))])
    incorrect_all = sum([np.all([preds[name][i] != labels[i] for name in model_names])
                        for i in range(len(labels))])
    
    categories = ['All Models\nCorrect', 'Some Models\nCorrect', 'All Models\nIncorrect']
    values = [correct_all, correct_some, incorrect_all]
    colors = ['#2E7D32', '#FFA726', '#C62828']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=2)
    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Agreement Analysis', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        percentage = (height / len(labels)) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Model Agreement and Consensus', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/05_model_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 05_model_agreement.png")
    
    # 6. Convergence Analysis
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Moving average
    ax1 = axes[0]
    window = 100
    for name, tracker in trackers.items():
        if len(tracker.accuracy_history) > window:
            moving_avg = pd.Series(tracker.accuracy_history).rolling(window=window).mean()
            ax1.plot(tracker.sample_indices, [a * 100 for a in moving_avg],
                    label=name, linewidth=2.5, alpha=0.8, color=MODEL_COLORS[name])
    
    ax1.set_xlabel('Samples', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Moving Avg Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Convergence (Window={window})', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Stability
    ax2 = axes[1]
    window = 500
    for name, tracker in trackers.items():
        if len(tracker.accuracy_history) > window:
            rolling_std = pd.Series(tracker.accuracy_history).rolling(window=window).std()
            ax2.plot(tracker.sample_indices, [v * 100 for v in rolling_std],
                    label=name, linewidth=2.5, alpha=0.8, color=MODEL_COLORS[name])
    
    ax2.set_xlabel('Samples', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Std Deviation (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Stability (Window={window})', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Convergence and Stability Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/06_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 06_convergence_analysis.png")


def create_summary_report(trackers):
    """Create text summary"""
    report_path = f'{PLOT_DIR}/summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("REAL-TIME MODEL COMPARISON - FINAL SUMMARY\n")
        f.write("AHU Fault Detection System\n")
        f.write("="*80 + "\n\n")
        
        for name, tracker in trackers.items():
            f.write(f"\n{name.upper()}\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Samples:       {tracker.total:,}\n")
            f.write(f"Correct Predictions: {tracker.correct:,}\n")
            f.write(f"Accuracy:            {tracker.get_accuracy()*100:.4f}%\n")
            f.write(f"Weighted F1:         {tracker.get_f1_weighted()*100:.4f}%\n")
            f.write(f"Macro F1:            {tracker.get_f1_macro()*100:.4f}%\n\n")
            
            f.write("Per-Class Metrics:\n")
            per_class = tracker.get_per_class_metrics()
            for class_id in range(NUM_CLASSES):
                metrics = per_class.get(class_id, {})
                f.write(f"  Class {class_id} ({LABEL_NAMES_FULL[class_id]}):\n")
                f.write(f"    Precision: {metrics.get('precision', 0)*100:.2f}%\n")
                f.write(f"    Recall:    {metrics.get('recall', 0)*100:.2f}%\n")
                f.write(f"    F1:        {metrics.get('f1', 0)*100:.2f}%\n")
                f.write(f"    Support:   {metrics.get('support', 0)}\n")
            
            cm = tracker.get_confusion_matrix()
            if cm is not None:
                f.write("\nConfusion Matrix:\n")
                for row in cm:
                    f.write("  " + "  ".join([f"{val:>6d}" for val in row]) + "\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Saved: summary_report.txt")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("REAL-TIME MODEL COMPARISON WITH PROGRESSIVE PLOTS")
    print("="*80)
    
    create_plots_directory()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print(f"Loading test data from: {TEST_DATA_PATH}")
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    ahu_cols = [col for col in test_df.columns if col.startswith('ahu_')]
    drop_cols = ['label'] + ahu_cols + ['id', 'label_full', 'AHU_name']
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df['label'].values
    X_test = np.nan_to_num(X_test.values.astype(np.float32), nan=0.0)
    
    input_dim = X_test.shape[1]
    num_samples = X_test.shape[0]
    
    print(f"Test samples:      {num_samples:,}")
    print(f"Input features:    {input_dim}")
    
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
            print(f"  ✗ {name} not found")
            return
    
    trackers = {name: MetricsTracker(name) for name in models.keys()}
    
    print(f"\n✓ All models loaded!")
    print(f"\nProcessing {num_samples:,} samples...")
    print(f"  - Showing predictions in real-time")
    print(f"  - Generating plots every {UPDATE_INTERVAL:,} samples\n")
    
    # Print header
    print("="*120)
    print(f"{'#':<8} {'True':<15} {'Centralized':<35} {'FedAvg':<35} {'Edge-Aware':<35}")
    print("-"*120)
    
    checkpoint_num = 0
    
    # Process all samples
    for idx in range(num_samples):
        x = torch.from_numpy(X_test[idx:idx+1]).to(device)
        true_label = int(y_test[idx])
        true_name = LABEL_NAMES[true_label]
        
        predictions = {}
        with torch.no_grad():
            for name, model in models.items():
                output = model(x)
                pred = output.max(1)[1].item()
                predictions[name] = pred
                trackers[name].update(pred, true_label)
        
        # Print prediction (every sample or at intervals)
        if (idx + 1) % PRINT_INTERVAL == 0:
            cent_pred = predictions['Centralized Baseline']
            fedavg_pred = predictions['FedAvg']
            edge_pred = predictions['Edge-Aware FL']
            
            cent_symbol = "✓" if cent_pred == true_label else "✗"
            fedavg_symbol = "✓" if fedavg_pred == true_label else "✗"
            edge_symbol = "✓" if edge_pred == true_label else "✗"
            
            cent_name = LABEL_NAMES[cent_pred]
            fedavg_name = LABEL_NAMES[fedavg_pred]
            edge_name = LABEL_NAMES[edge_pred]
            
            print(f"{idx+1:<8} {true_name:<15} {cent_symbol} {cent_name:<32} {fedavg_symbol} {fedavg_name:<32} {edge_symbol} {edge_name:<32}")
        
        # Generate checkpoint plots
        if (idx + 1) % UPDATE_INTERVAL == 0:
            checkpoint_num += 1
            print("\n" + "="*120)
            print(f"CHECKPOINT {checkpoint_num} - Processed {idx+1:,}/{num_samples:,} samples")
            
            # Print current metrics
            for name, tracker in trackers.items():
                acc = tracker.get_accuracy() * 100
                f1 = tracker.get_f1_weighted() * 100
                print(f"  {name:<25} Acc: {acc:>6.2f}%  F1: {f1:>6.2f}%")
            
            print(f"  Generating checkpoint plots...")
            plot_checkpoint_curves(trackers, checkpoint_num, num_samples)
            plot_checkpoint_comparison(trackers, checkpoint_num)
            print("="*120 + "\n")
            
            # Reprint header
            print(f"{'#':<8} {'True':<15} {'Centralized':<35} {'FedAvg':<35} {'Edge-Aware':<35}")
            print("-"*120)
    
    print(f"\n✓ Processing complete!")
    
    # Generate final comprehensive plots
    print(f"\n{'='*80}")
    print("GENERATING FINAL COMPREHENSIVE PLOTS")
    print("="*80 + "\n")
    
    plot_final_comprehensive(trackers)
    create_summary_report(trackers)
    
    print(f"\n{'='*80}")
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"✓ Plots saved in: {PLOT_DIR}/")
    print(f"  - Checkpoint plots: {PLOT_DIR}/checkpoints/")
    print(f"  - Final plots: {PLOT_DIR}/")
    print("="*80)
    
    # Print summary
    print("\nFINAL PERFORMANCE SUMMARY:")
    print("-"*80)
    for name, tracker in trackers.items():
        print(f"{name:<25} Acc: {tracker.get_accuracy()*100:>6.2f}%  "
              f"F1(W): {tracker.get_f1_weighted()*100:>6.2f}%  "
              f"F1(M): {tracker.get_f1_macro()*100:>6.2f}%")
    print("-"*80 + "\n")


if __name__ == "__main__":
    main()
