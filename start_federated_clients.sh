#!/bin/bash
# Script to start multiple federated learning clients

cd /Users/faiqahmed/Desktop/Semesters/Semester7/MLOPS/PROJECT
source venv/bin/activate

SERVER_ADDRESS="localhost:8080"

echo "Starting federated learning clients..."
echo "Make sure the server is running first!"
echo ""

for i in {0..4}; do
    echo "Starting client $i..."
    python -c "from src.federated.federated_client import start_client; start_client($i, '$SERVER_ADDRESS')" &
    sleep 2  # Wait 2 seconds between clients
done

echo ""
echo "All clients started!"
echo "Check the server terminal to see connections and training progress."
echo ""
echo "To stop clients, press Ctrl+C or close the terminals"




