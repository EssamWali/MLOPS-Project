# Federated Learning Setup Guide

## Overview

Federated Learning allows training models across multiple nodes (hospitals/cities) without sharing raw data. Each node trains on local data and only sends model updates to the server.

## Architecture

```
┌─────────────┐
│   Server    │  ← Aggregates model updates
└──────┬──────┘
       │
   ┌───┴───┬─────────┬─────────┐
   │       │         │         │
┌──▼──┐ ┌──▼──┐ ┌───▼──┐ ┌───▼──┐
│Node0│ │Node1│ │Node2│ │Node3│  ← Train on local data
└─────┘ └─────┘ └──────┘ └──────┘
```

## Step 1: Start the Server

**Terminal 1 - Start Server:**
```bash
cd /Users/faiqahmed/Desktop/Semesters/Semester7/MLOPS/PROJECT
source venv/bin/activate
python src/federated/federated_server.py
```

**Default Settings:**
- Server address: `0.0.0.0:8080`
- Expected clients: 5
- Rounds: 10

**Custom Settings:**
```bash
# Custom number of clients and rounds
python src/federated/federated_server.py 5 10
```

**Expected Output:**
```
Starting federated learning server on 0.0.0.0:8080
Expected clients: 5
Rounds: 10
Min fit clients: 3
Min available clients: 3
...
[Server started, waiting for clients...]
```

## Step 2: Start Clients

**Option A: Run in Separate Terminals (Recommended for Simulation)**

**Terminal 2 - Client 0:**
```bash
cd /Users/faiqahmed/Desktop/Semesters/Semester7/MLOPS/PROJECT
source venv/bin/activate
python -c "from src.federated.federated_client import start_client; start_client(0, 'localhost:8080')"
```

**Terminal 3 - Client 1:**
```bash
python -c "from src.federated.federated_client import start_client; start_client(1, 'localhost:8080')"
```

**Terminal 4 - Client 2:**
```bash
python -c "from src.federated.federated_client import start_client; start_client(2, 'localhost:8080')"
```

**Terminal 5 - Client 3:**
```bash
python -c "from src.federated.federated_client import start_client; start_client(3, 'localhost:8080')"
```

**Terminal 6 - Client 4:**
```bash
python -c "from src.federated.federated_client import start_client; start_client(4, 'localhost:8080')"
```

**Option B: Use Helper Script**

Create a simple script to start all clients:

```python
# start_clients.py
import subprocess
import time
from pathlib import Path

for node_id in range(5):
    cmd = f"""python -c "from src.federated.federated_client import start_client; start_client({node_id}, 'localhost:8080')" """
    print(f"Starting client {node_id}...")
    subprocess.Popen(cmd, shell=True)
    time.sleep(2)  # Wait between clients
```

## How It Works

1. **Server starts** and waits for clients to connect
2. **Clients connect** and load their local data (from `data/federated/wearable_node_X.csv`)
3. **Training Round:**
   - Server sends global model to clients
   - Each client trains on local data
   - Clients send model updates back to server
   - Server aggregates updates (Federated Averaging)
4. **Repeat** for specified number of rounds
5. **Evaluation:** Server evaluates aggregated model

## Data Files Required

Make sure federated data files exist:
```bash
ls data/federated/wearable_node_*.csv
```

Should show:
- `wearable_node_0.csv`
- `wearable_node_1.csv`
- `wearable_node_2.csv`
- `wearable_node_3.csv`
- `wearable_node_4.csv`

If missing, generate them:
```bash
python src/data_ingestion/collect_data.py
```

## Troubleshooting

### Issue: "Address already in use"
**Solution:** Port 8080 is already in use. Change port:
```bash
python src/federated/federated_server.py 5 10
# Then modify server_address in code or use different port
```

### Issue: "Data file not found"
**Solution:** Generate federated data:
```bash
python src/data_ingestion/collect_data.py
```

### Issue: Clients not connecting
**Solution:**
1. Make sure server is running first
2. Check server address matches: `localhost:8080`
3. Wait a few seconds between starting clients

### Issue: "ModuleNotFoundError: flwr"
**Solution:**
```bash
source venv/bin/activate
pip install flwr
```

## Understanding the Output

**Server Output:**
```
INFO flwr 2024-11-15 20:00:00,000 | server | Starting server...
INFO flwr 2024-11-15 20:00:05,000 | server | Client 0 connected
INFO flwr 2024-11-15 20:00:06,000 | server | Client 1 connected
...
INFO flwr 2024-11-15 20:01:00,000 | server | Round 1: aggregated metrics = {'accuracy': 0.85, ...}
```

**Client Output:**
```
Node 0: Loaded 600 training samples, 150 test samples
Training random_forest model...
FederatedClient(node_id=0) connecting to server...
```

## Next Steps

1. **Monitor Training:** Watch metrics improve across rounds
2. **Compare Models:** Compare federated vs centralized training
3. **Experiment:** Try different numbers of clients/rounds
4. **Production:** Deploy clients on different machines (hospitals/cities)

## Production Deployment

In production:
- **Server:** Runs on central ML infrastructure
- **Clients:** Run on individual hospital/city servers
- **Network:** Clients connect over secure network
- **Privacy:** Raw data never leaves local nodes

## Example Workflow

```bash
# Terminal 1: Start server
python src/federated/federated_server.py

# Terminal 2-6: Start clients (one per terminal)
python -c "from src.federated.federated_client import start_client; start_client(0, 'localhost:8080')"
python -c "from src.federated.federated_client import start_client; start_client(1, 'localhost:8080')"
# ... etc
```

Wait for training to complete! You'll see aggregated metrics after each round.




