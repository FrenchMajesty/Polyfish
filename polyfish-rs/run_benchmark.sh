#!/bin/bash
set -e

# Configuration for Benchmark
NUM_GAMES=10
MCTS_ITERS=200
# Ensure batch inference is fully utilized
export RAYON_NUM_THREADS=24
export OMP_NUM_THREADS=1

echo "=============================================="
echo "Running PolyFish Benchmark"
echo "Games: $NUM_GAMES | MCTS Iters: $MCTS_ITERS | Threads: $RAYON_NUM_THREADS"
echo "=============================================="

# Build release binary if not already built
echo "Building release binary..."
cargo build --bin self_play --release --features cuda

echo "Starting Benchmark..."
START_TIME=$(date +%s)

# Run self_play
# Use tee to stream output to console AND file for parsing
./target/release/self_play --num-games $NUM_GAMES --mcts-iters $MCTS_ITERS --no-train 2>&1 | tee benchmark_output.log

# Extract Device
DEVICE_USED=$(grep "Using device:" benchmark_output.log | head -n 1)

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Calculate Games per Second
GPS=$(awk "BEGIN {printf \"%.2f\", $NUM_GAMES / $DURATION}")

echo "=============================================="
echo "Benchmark Complete!"
echo "Total Time: ${DURATION}s"
echo "Device: $DEVICE_USED"
echo "Throughput: $GPS games/second"
echo "=============================================="
