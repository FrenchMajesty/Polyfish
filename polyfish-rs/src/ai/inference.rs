use crate::ai::network::{PolicyOutput, PolyZeroNet, ValueOutput};
use candle_core::{Device, Tensor};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

pub struct InferenceRequest {
    pub spatial: Tensor, // [B, C, H, W]
    pub player: Tensor,  // [B, 10]
    pub reply: Sender<(PolicyOutput, ValueOutput)>,
}

pub struct InferenceServer {
    network: Arc<PolyZeroNet>,
    receiver: Receiver<InferenceRequest>,
    batch_size: usize,
    device: Device,
}

impl InferenceServer {
    pub fn new(
        network: Arc<PolyZeroNet>,
        receiver: Receiver<InferenceRequest>,
        batch_size: usize,
    ) -> Self {
        Self {
            device: network.device(),
            network,
            receiver,
            batch_size,
        }
    }

    pub fn run(&self) {
        let mut batch_queue: Vec<InferenceRequest> = Vec::with_capacity(self.batch_size);

        loop {
            // 1. Collect batch
            // Blocking receive for first item
            if let Ok(req) = self.receiver.recv() {
                batch_queue.push(req);
            } else {
                // Channel closed
                eprintln!("InferenceServer channel closed, shutting down.");
                break;
            }

            // eprintln!("InferenceServer: Got 1 request");

            // Greedy collect up to batch_size or timeout (simple greedy for now)
            // Ideally we use a timeout, but std mpsc doesn't support recv_timeout easily with mixed blocking
            // We can check try_recv loop
            let _start_time = Instant::now();
            while batch_queue.len() < self.batch_size {
                if let Ok(req) = self.receiver.try_recv() {
                    batch_queue.push(req);
                } else {
                    // Empty queue, maybe wait a tiny bit or just break if we have *something*
                    // If we rely on pure blocking, we maximize throughput but latency suffers for single games.
                    // For batch self-play, throughput is king.
                    // Implementation: If we have data, process it. Don't busy wait too long.
                    // Better: sleep briefly if queue is empty but batch not full?
                    // Actually, for self-play with 24 threads, we likely always have data.
                    // Simple logic: try_recv until empty.
                    break;
                }
            }

            if batch_queue.is_empty() {
                continue;
            }

            // 2. Collate Tensors
            // This is the tricky part: input tensors are already [B_local, ...], we need to stack them.
            // But `InferenceRequest` has `spatial` as Tensor.
            // We can just push them into a vec and cat.

            let mut spatials = Vec::with_capacity(batch_queue.len());
            let mut players = Vec::with_capacity(batch_queue.len());
            let mut sizes = Vec::with_capacity(batch_queue.len()); // To split output later

            for req in &batch_queue {
                spatials.push(req.spatial.clone());
                players.push(req.player.clone());
                sizes.push(req.spatial.dim(0).unwrap_or(1));
            }

            let batch_spatial = Tensor::cat(&spatials, 0).unwrap_or_else(|e| {
                panic!("Failed to cat batch spatial: {}", e);
            });
            let batch_player = Tensor::cat(&players, 0).unwrap_or_else(|e| {
                panic!("Failed to cat batch player: {}", e);
            });

            // Ensure tensors are on the correct device (cheap no-op if already matching)
            let batch_spatial = batch_spatial.to_device(&self.device).unwrap_or_else(|e| {
                panic!("Failed to move batch spatial to device: {}", e);
            });
            let batch_player = batch_player.to_device(&self.device).unwrap_or_else(|e| {
                panic!("Failed to move batch player to device: {}", e);
            });

            // 3. Forward Pass
            let (policy_out, value_out) =
                match self.network.forward_t(&batch_spatial, &batch_player, false) {
                    Ok(out) => out,
                    Err(e) => {
                        eprintln!("InferenceServer Panic: Network forward failed: {}", e);
                        eprintln!("Batch spatial shape: {:?}", batch_spatial.shape());
                        eprintln!("Batch player shape: {:?}", batch_player.shape());
                        panic!("Inference network error");
                    }
                };

            // 4. Split and Reply
            // We need to slice the outputs back to matches the requests
            // PolicyOutput fields are tensors [Total_B, ...]
            // ValueOutput fields are [Total_B, 1]

            let mut offset = 0;
            for (i, req) in batch_queue.drain(..).enumerate() {
                let size = sizes[i];

                // Helper to slice
                let slice_policy = PolicyOutput {
                    action_type: policy_out.action_type.narrow(0, offset, size).unwrap(),
                    source_spatial: policy_out.source_spatial.narrow(0, offset, size).unwrap(),
                    target_spatial: policy_out.target_spatial.narrow(0, offset, size).unwrap(),
                    move_option: policy_out.move_option.narrow(0, offset, size).unwrap(),
                };

                let slice_value = ValueOutput {
                    win_value: value_out.win_value.narrow(0, offset, size).unwrap(),
                    eco_value: value_out.eco_value.narrow(0, offset, size).unwrap(),
                    mil_value: value_out.mil_value.narrow(0, offset, size).unwrap(),
                };

                let _ = req.reply.send((slice_policy, slice_value));
                offset += size;
            }
        }
    }
}
