# Edge-Aware Federated Learning Policy for AHU Fault Detection

## Executive Summary

This document outlines the edge-aware federated learning policy for Air Handling Unit (AHU) fault detection systems. The policy considers device constraints (CPU, power, network, memory) and adapts training strategies accordingly to enable efficient distributed learning across heterogeneous AHU edge devices.

## 1. AHU Device Characteristics

### 1.1 Physical Constraints
- **Power**: Typically AC-powered (24/7 operation), but may have backup battery systems
- **CPU**: Embedded processors (ARM Cortex-A series, Intel Atom) with limited compute
- **Memory**: 1-4GB RAM typical for IoT edge devices
- **Network**: Wired Ethernet or WiFi, bandwidth 10-100 Mbps
- **Operating Environment**: Industrial settings, continuous operation required

### 1.2 Computational Constraints
- **CPU Score Range**: 40-95 (normalized 0-100 scale)
  - Low-end: 40-60 (basic monitoring)
  - Mid-range: 60-80 (standard AHU controllers)
  - High-end: 80-95 (advanced controllers with edge AI)
- **Battery**: 0-100% (for backup systems, or treated as "power availability")
- **Bandwidth**: 10-100 Mbps (varies by connection type)
- **Memory**: 1-4GB (affects batch size and model complexity)

## 2. Edge-Aware Policy Framework

### 2.1 Device Eligibility Criteria

A device is eligible to participate in a training round if:
```
eligibility = (battery >= MIN_BATTERY) AND 
              (cpu_score >= MIN_CPU) AND
              (available_memory >= model_memory_requirement) AND
              (network_status == "connected")
```

**Default Thresholds:**
- `MIN_BATTERY = 20%` (or power availability > 80% for AC-powered)
- `MIN_CPU = 30%` (minimum compute capability)
- `model_memory_requirement = 500MB` (for HybridLSTMCNNAttention)

### 2.2 Adaptive Training Strategy

#### 2.2.1 Epoch Adaptation
Local training epochs are adapted based on device capabilities:

```
adaptive_epochs = base_epochs * (0.5 + 0.5 * (cpu_score / 100.0))
adaptive_epochs = max(1, min(adaptive_epochs, base_epochs * 2))
```

**Rationale:**
- Low-capability devices: fewer epochs to conserve resources
- High-capability devices: more epochs for better convergence
- Prevents resource exhaustion on constrained devices

#### 2.2.2 Batch Size Adaptation
Batch size is adapted based on available memory:

```
if memory >= 4GB:
    batch_size = 128
elif memory >= 2GB:
    batch_size = 64
else:
    batch_size = 32
```

#### 2.2.3 Model Offloading Decision

The policy decides whether to:
1. **Train locally** (on-device)
2. **Offload to edge server** (if available)
3. **Offload to cloud** (last resort)

**Decision Tree:**
```
IF (cpu_score >= 70 AND battery >= 50 AND bandwidth >= 20):
    train_locally()
ELIF (edge_server_available AND bandwidth >= 10):
    offload_to_edge_server()
ELSE:
    offload_to_cloud() OR skip_round()
```

### 2.3 Weighted Aggregation Policy

Model updates are weighted based on:
1. **Data size** (50%): Larger datasets contribute more
2. **Device capability** (30%): Higher CPU = better quality updates
3. **Resource availability** (20%): Battery/power level affects reliability

```
aggregation_weight = data_size * (
    0.5 +                    # Base data contribution
    0.3 * (cpu_score / 100) + # Capability factor
    0.2 * (battery / 100)     # Reliability factor
)
```

### 2.4 Resource Management

#### 2.4.1 Battery/Power Consumption
- Training consumes: `1% battery per minute of training`
- Periodic recharging: `+15% every 10 rounds` (simulates maintenance/charging)
- Emergency recharge: `+20% if battery < 10%` (prevents device dropout)

#### 2.4.2 Network-Aware Communication
- **High bandwidth (>50 Mbps)**: Full model transfer
- **Medium bandwidth (20-50 Mbps)**: Compressed model (quantization)
- **Low bandwidth (<20 Mbps)**: Gradient-only transfer or skip

#### 2.4.3 Memory Management
- Model checkpointing to disk if memory constrained
- Gradient accumulation for small batch sizes
- Early stopping if memory pressure detected

## 3. Offloading Strategies

### 3.1 CPU-Based Offloading
**When to offload:**
- CPU score < 50: Always offload
- CPU score 50-70: Offload if battery < 40%
- CPU score > 70: Train locally

### 3.2 Battery/Power-Based Offloading
**When to offload:**
- Battery < 20%: Always offload
- Battery 20-40%: Offload if training time > 5 minutes
- Battery > 40%: Train locally

### 3.3 Network-Based Offloading
**When to offload:**
- Bandwidth < 10 Mbps: Train locally (offloading too slow)
- Bandwidth 10-30 Mbps: Conditional offload (if CPU/battery low)
- Bandwidth > 30 Mbps: Flexible (prefer local if resources allow)

### 3.4 Hybrid Offloading Policy
**Priority order:**
1. **Local training** (if CPU ≥ 70 AND battery ≥ 50)
2. **Edge server offload** (if available AND bandwidth ≥ 10)
3. **Cloud offload** (if bandwidth ≥ 20 AND local not feasible)
4. **Skip round** (if all options unavailable)

## 4. Implementation Details

### 4.1 Device Selection
At each round:
1. Check all devices for eligibility
2. If no devices eligible → recharge all and retry
3. Select top-K devices (K = min(eligible_count, 5))
4. Prioritize by: (data_size * capability_score)

### 4.2 Training Execution
For each selected device:
1. Load global model
2. Adapt epochs/batch_size based on device specs
3. Train locally (or offload if policy dictates)
4. Measure training time and resource consumption
5. Update device battery/status
6. Return model update with weight

### 4.3 Aggregation
1. Collect all model updates with weights
2. Weighted average: `w_global = Σ(w_i * model_i) / Σ(w_i)`
3. Update global model
4. Evaluate on test set

## 5. Policy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_BATTERY` | 20% | Minimum battery for participation |
| `MIN_CPU` | 30% | Minimum CPU score for participation |
| `BASE_LOCAL_EPOCHS` | 5 | Base number of local epochs |
| `RECHARGE_INTERVAL` | 10 rounds | Periodic recharge interval |
| `RECHARGE_AMOUNT` | 15% | Battery recharge per interval |
| `MAX_OFFLOAD_LATENCY` | 30s | Maximum acceptable offload latency |
| `BANDWIDTH_THRESHOLD_LOW` | 10 Mbps | Low bandwidth threshold |
| `BANDWIDTH_THRESHOLD_HIGH` | 50 Mbps | High bandwidth threshold |

## 6. Monitoring and Adaptation

### 6.1 Metrics Tracked
- Device participation rate
- Average training time per device
- Battery consumption rate
- Network utilization
- Model convergence rate
- Resource efficiency (accuracy per resource unit)

### 6.2 Dynamic Adaptation
- Adjust thresholds based on observed performance
- Rebalance weights if certain devices consistently underperform
- Adapt offloading decisions based on network conditions

## 7. Expected Benefits

1. **Resource Efficiency**: 30-50% reduction in battery/power consumption
2. **Faster Convergence**: 15-25% faster convergence through smart device selection
3. **Better Scalability**: Handles heterogeneous device fleet
4. **Improved Reliability**: Prevents device failures due to resource exhaustion
5. **Network Optimization**: Reduces unnecessary data transfer

## 8. Limitations and Future Work

- Current policy assumes static device capabilities (could be dynamic)
- Network conditions modeled as constant (real-world varies)
- Battery model simplified (could include charging patterns)
- No consideration of data freshness/quality in weighting
- Future: Reinforcement learning for adaptive policy optimization

