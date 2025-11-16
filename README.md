# fault-detection-iomt
Fault Detection in air handling units (AHU) of hospital

## Prerequisites
First, make sure you have installed Python 3.13.9.

## Training
Create virtual envvironment and activate it:
```bash
python -m venv venv
```

Install required libraries

```bash
pip install -r requirements.txt
```

Pre-precess the data

```bash
python preprocess.py
```

Split and prepare data for training processes

```bash
python train_centralized_baseline.py
```

```bash
python train_fedavg.py
```

```bash
python train_edge_aware_fl.py
```

## Data Streaming
Run the code below and see real-time predictions of all three models, including plots.
```bash
python stream.py
```

(Plots have been shared in this repository)
