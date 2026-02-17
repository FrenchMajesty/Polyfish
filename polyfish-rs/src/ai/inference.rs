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
            let mut total_items = 0;
            // 1. Collect batch
            // Blocking receive for first item
            if let Ok(req) = self.receiver.recv() {
                let items = req.spatial.dim(0).unwrap_or(1);
                total_items += items;
                batch_queue.push(req);
            } else {
                // Channel closed
                eprintln!("InferenceServer channel closed, shutting down.");
                break;
            }

            // eprintln!("InferenceServer: Got 1 request");

            // Greedy collect up to batch_size items
            // We want to fill the batch but not exceed the limit too much.
            // Since we receive chunks of ~24, we might slightly overshoot if we check after receiving.
            // But better to check `total_items < self.batch_size`
            let _start_time = Instant::now();
            while total_items < self.batch_size {
                if let Ok(req) = self.receiver.try_recv() {
                    let items = req.spatial.dim(0).unwrap_or(1);
                    total_items += items;
                    batch_queue.push(req);
                } else {
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
            // 3. Forward Pass
            let forward_result = self.network.forward_t(&batch_spatial, &batch_player, false);

            let (policy_out, value_out) = match forward_result {
                Ok(out) => out,
                Err(e) => {
                    eprintln!("InferenceServer Error: Network forward failed: {}", e);
                    eprintln!("Batch spatial shape: {:?}", batch_spatial.shape());
                    eprintln!("Batch player shape: {:?}", batch_player.shape());
                    // Drop requests (triggering channel closed error on workers) and continue
                    batch_queue.clear();
                    continue;
                }
            };

            // 4. Split and Reply
            // We need to slice the outputs back to matches the requests
            // PolicyOutput fields are tensors [Total_B, ...]
            // ValueOutput fields are [Total_B, 1]

            let total_batch_size: usize = sizes.iter().sum();
            let output_batch_size = policy_out.action_type.dim(0).unwrap_or(0);

            if output_batch_size != total_batch_size {
                eprintln!(
                    "CRITICAL ERROR: Mismatch between input batch size and output batch size!"
                );
                eprintln!("Input batch size (sum of requests): {}", total_batch_size);
                eprintln!("Output batch size (from network): {}", output_batch_size);
                eprintln!("Batch spatial shape: {:?}", batch_spatial.shape());
            }

            let mut offset = 0;
            for (i, req) in batch_queue.drain(..).enumerate() {
                let size = sizes[i];

                // Helper to slice
                let slice_policy = PolicyOutput {
                    action_type: policy_out
                        .action_type
                        .narrow(0, offset, size)
                        .unwrap_or_else(|e| {
                            panic!(
                                "Slice action failed: {} offset {} size {} dim {:?}",
                                e,
                                offset,
                                size,
                                policy_out.action_type.shape()
                            )
                        }),
                    source_spatial: policy_out
                        .source_spatial
                        .narrow(0, offset, size)
                        .unwrap_or_else(|e| {
                            panic!(
                                "Slice source failed: {} offset {} size {} dim {:?}",
                                e,
                                offset,
                                size,
                                policy_out.source_spatial.shape()
                            )
                        }),
                    target_spatial: policy_out
                        .target_spatial
                        .narrow(0, offset, size)
                        .unwrap_or_else(|e| {
                            panic!(
                                "Slice target failed: {} offset {} size {} dim {:?}",
                                e,
                                offset,
                                size,
                                policy_out.target_spatial.shape()
                            )
                        }),
                    move_option: policy_out
                        .move_option
                        .narrow(0, offset, size)
                        .unwrap_or_else(|e| {
                            panic!(
                                "Slice option failed: {} offset {} size {} dim {:?}",
                                e,
                                offset,
                                size,
                                policy_out.move_option.shape()
                            )
                        }),
                };

                let slice_value =
                    ValueOutput {
                        win_value: value_out.win_value.narrow(0, offset, size).unwrap_or_else(
                            |e| {
                                panic!(
                                    "Slice win failed: {} offset {} size {} dim {:?}",
                                    e,
                                    offset,
                                    size,
                                    value_out.win_value.shape()
                                )
                            },
                        ),
                        eco_value: value_out.eco_value.narrow(0, offset, size).unwrap_or_else(
                            |e| {
                                panic!(
                                    "Slice eco failed: {} offset {} size {} dim {:?}",
                                    e,
                                    offset,
                                    size,
                                    value_out.eco_value.shape()
                                )
                            },
                        ),
                        mil_value: value_out.mil_value.narrow(0, offset, size).unwrap_or_else(
                            |e| {
                                panic!(
                                    "Slice mil failed: {} offset {} size {} dim {:?}",
                                    e,
                                    offset,
                                    size,
                                    value_out.mil_value.shape()
                                )
                            },
                        ),
                    };

                let _ = req.reply.send((slice_policy, slice_value));
                offset += size;
            }
        }
    }
}
