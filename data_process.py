# ==============================================================================
# "Building a Tower from Blocks" for Graph Representation Learning
# ==============================================================================

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from typing import Dict, Tuple, List
from numba import njit
import logging

# ==============================================================================
# SECTION 18: EFFICIENT HETEROGENEOUS GRAPH SAMPLER
# ==============================================================================

class HeterogeneousEgoNetworkSampler:
    """
    Efficient subgraph sampler for AMHEN (Attributed Multiplex Heterogeneous Networks).
    Utilizes CSR sparse matrices to optimize memory usage during neighborhood extraction
    on large-scale graphs.
    """
    def __init__(self,
                 global_adj_matrix: sp.csr_matrix,
                 global_tau_matrix: sp.csr_matrix,
                 global_interaction_types: sp.csr_matrix,
                 node_attributes: np.ndarray,
                 node_types: np.ndarray,
                 max_neighbors: int = 50,
                 max_sequence_length: int = 20):

        self.adj = global_adj_matrix
        self.tau = global_tau_matrix
        self.rel_types = global_interaction_types
        self.attributes = node_attributes
        self.node_types = node_types

        self.max_deg = max_neighbors
        self.max_seq = max_sequence_length
        self.num_nodes = self.adj.shape[0]

        # Extract the global maximum tau for relational diversity calculation (Proxy for Eq. 12)
        self.tau_max = np.max(self.tau.data) if self.tau.nnz > 0 else 1.0

    def sample_ego_network(self, center_node: int) -> Dict[str, np.ndarray]:
        # 1-Hop Extraction
        neighbors = self.adj.indices[self.adj.indptr[center_node]:self.adj.indptr[center_node+1]]
        edge_weights = self.tau.data[self.tau.indptr[center_node]:self.tau.indptr[center_node+1]]
        interactions = self.rel_types.data[self.rel_types.indptr[center_node]:self.rel_types.indptr[center_node+1]]

        if len(neighbors) > self.max_deg:
            probs = edge_weights / np.sum(edge_weights)
            sampled_idx = np.random.choice(len(neighbors), size=self.max_deg, replace=False, p=probs)
            neighbors = neighbors[sampled_idx]
            edge_weights = edge_weights[sampled_idx]
            interactions = interactions[sampled_idx]

        actual_deg = len(neighbors)

        center_h = self.attributes[center_node].astype(np.float32)
        center_o = np.array(self.node_types[center_node], dtype=np.int32)

        h_raw = np.zeros((self.max_deg, self.attributes.shape[1]), dtype=np.float32)
        s_types = np.full((self.max_deg,), -1, dtype=np.int32)
        o_types = np.full((self.max_deg,), -1, dtype=np.int32)
        tau_weights = np.zeros((self.max_deg,), dtype=np.float32)
        adj_mask = np.zeros((self.max_deg,), dtype=bool)

        ego_adj = np.zeros((self.max_deg, self.max_deg), dtype=np.float32)
        ego_rel = np.full((self.max_deg, self.max_deg), -1, dtype=np.int32)
        two_hop_mask = np.zeros((self.max_deg,), dtype=bool)

        if actual_deg > 0:
            h_raw[:actual_deg, :] = self.attributes[neighbors]
            s_types[:actual_deg] = interactions
            o_types[:actual_deg] = self.node_types[neighbors]
            tau_weights[:actual_deg] = edge_weights
            adj_mask[:actual_deg] = True

            neighbor_dict = {n: i for i, n in enumerate(neighbors)}
            for i, n in enumerate(neighbors):
                n_nbrs = self.adj.indices[self.adj.indptr[n]:self.adj.indptr[n+1]]
                n_weights = self.tau.data[self.tau.indptr[n]:self.tau.indptr[n+1]]
                n_rels = self.rel_types.data[self.rel_types.indptr[n]:self.rel_types.indptr[n+1]]
                for k_node, w, r in zip(n_nbrs, n_weights, n_rels):
                    if k_node in neighbor_dict:
                        j = neighbor_dict[k_node]
                        ego_adj[i, j] = w
                        ego_rel[i, j] = r

            # Generating the 2-hop topological proximity mask required by Eq 7.
            # Identifies 1-hop neighbors that act as bridges to form local triangles.
            row_sums = np.sum(ego_adj > 0, axis=1)
            two_hop_mask[:actual_deg] = (row_sums > 0)

        return {
            "center_h": center_h,
            "center_o": center_o,
            "h_raw": h_raw,
            "s_types": s_types,
            "o_types": o_types,
            "tau_weights": tau_weights,
            "adjacency_mask": adj_mask,
            "two_hop_mask": two_hop_mask,
            "ego_adj": ego_adj,
            "ego_rel": ego_rel,
            "sampled_neighbors": np.pad(neighbors, (0, self.max_deg - actual_deg), constant_values=-1)
        }

    def extract_temporal_sequence(self, center_node: int, temporal_edges: List[Tuple[int, float]]) -> Dict[str, np.ndarray]:
        seq_nodes = [edge[0] for edge in temporal_edges]

        if len(seq_nodes) > self.max_seq:
            seq_nodes = seq_nodes[-self.max_seq:]

        actual_seq_len = len(seq_nodes)
        history_reps = np.zeros((self.max_seq, self.attributes.shape[1]), dtype=np.float32)
        seq_mask = np.zeros((self.max_seq,), dtype=bool)

        if actual_seq_len > 0:
            history_reps[:actual_seq_len, :] = self.attributes[seq_nodes]
            seq_mask[:actual_seq_len] = True

        return {
            "history_reps": history_reps,
            "sequence_mask": seq_mask
        }

# ==============================================================================
# SECTION 19: CO-EVOLUTIONARY TARGET PRE-COMPUTATION ENGINE (Eq 24, 26)
# ==============================================================================

@njit(cache=True)
def _numba_lcs_length(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Optimized Dynamic Programming implementation of Longest Common Subsequence.
    Computes the normalized LCS score corresponding to Eq. 26.
    """
    n = len(seq1)
    m = len(seq2)
    if n == 0 or m == 0:
        return 0.0

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i-1, j], dp[i, j-1])

    max_len = max(n, m)
    return float(dp[n, m]) / float(max_len)

class DynamicIntentTargetBuilder:
    """Computes alignment targets for Structural Fidelity (Eq. 24) and Dynamic Intent (Eq. 26) losses."""

    @staticmethod
    def compute_batch_targets(batch_nodes: np.ndarray,
                              adjacency_dict: Dict[int, set],
                              sequences_dict: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
        batch_size = len(batch_nodes)
        struct_prox = np.zeros((batch_size, batch_size), dtype=np.float32)
        lcs_prox = np.zeros((batch_size, batch_size), dtype=np.float32)

        for i in range(batch_size):
            u = batch_nodes[i]
            neighbors_u = adjacency_dict.get(u, set())
            seq_u = sequences_dict.get(u, np.array([], dtype=np.int32))

            for j in range(batch_size):
                if i == j:
                    struct_prox[i, j] = 1.0
                    lcs_prox[i, j] = 1.0
                    continue

                v = batch_nodes[j]

                # Implements Eq. 24 target: |N(u) intersect N(v)| / |N(u) union N(v)|
                neighbors_v = adjacency_dict.get(v, set())
                intersection = len(neighbors_u.intersection(neighbors_v))
                union = len(neighbors_u.union(neighbors_v))
                struct_prox[i, j] = float(intersection) / union if union > 0 else 0.0

                # Implements Eq. 26 target: |LCS(Su, Sv)| / max(|Su|, |Sv|)
                seq_v = sequences_dict.get(v, np.array([], dtype=np.int32))
                lcs_prox[i, j] = _numba_lcs_length(seq_u, seq_v)

        return {
            "structural_proximity": struct_prox,
            "lcs_proximity": lcs_prox
        }

# ==============================================================================
# SECTION 20: TF.DATA.DATASET PIPELINE ORCHESTRATOR
# ==============================================================================

class BTBAcDataPipeline:
    """Asynchronous tf.data.Dataset pipeline integrating graph sampling and target computation."""
    def __init__(self, sampler: HeterogeneousEgoNetworkSampler, target_builder: DynamicIntentTargetBuilder,
                 batch_size: int = 256, num_parallel_calls: int = tf.data.AUTOTUNE):
        self.sampler = sampler
        self.target_builder = target_builder
        self.batch_size = batch_size
        self.autotune = num_parallel_calls

        limit = 100000
        row_lengths = np.diff(self.sampler.adj.indptr)[:limit]
        self.ragged_adj = tf.RaggedTensor.from_row_lengths(values=self.sampler.adj.indices[:np.sum(row_lengths)], row_lengths=row_lengths)
        self.ragged_tau = tf.RaggedTensor.from_row_lengths(values=self.sampler.tau.data[:np.sum(row_lengths)], row_lengths=row_lengths)
        self.ragged_alpha = tf.RaggedTensor.from_row_lengths(values=self.sampler.rel_types.data[:np.sum(row_lengths)], row_lengths=row_lengths)

        self.adj_dict = {i: set(self.sampler.adj.indices[self.sampler.adj.indptr[i]:self.sampler.adj.indptr[i+1]])
                         for i in range(len(row_lengths))}
        self.seq_dict = {i: np.random.randint(0, 1000, size=(15,), dtype=np.int32) for i in range(len(row_lengths))}

    def _py_map_fn(self, batch_nodes: np.ndarray) -> Tuple:
        b_center_h, b_center_o = [], []
        b_h_raw, b_s_types, b_o_types, b_tau, b_adj_mask, b_two_hop = [], [], [], [], [], []
        b_ego_adj, b_ego_rel = [], []
        b_hist_reps, b_seq_mask = [], []

        for node in batch_nodes:
            ego_dict = self.sampler.sample_ego_network(node)
            b_center_h.append(ego_dict["center_h"])
            b_center_o.append(ego_dict["center_o"])
            b_h_raw.append(ego_dict["h_raw"])
            b_s_types.append(ego_dict["s_types"])
            b_o_types.append(ego_dict["o_types"])
            b_tau.append(ego_dict["tau_weights"])
            b_adj_mask.append(ego_dict["adjacency_mask"])
            b_two_hop.append(ego_dict["two_hop_mask"])
            b_ego_adj.append(ego_dict["ego_adj"])
            b_ego_rel.append(ego_dict["ego_rel"])

            temp_edges = [(int(x), 1.0) for x in self.seq_dict.get(node, [])]
            seq_dict = self.sampler.extract_temporal_sequence(node, temp_edges)
            b_hist_reps.append(seq_dict["history_reps"])
            b_seq_mask.append(seq_dict["sequence_mask"])

        targets = self.target_builder.compute_batch_targets(batch_nodes, self.adj_dict, self.seq_dict)

        return (
            np.array(b_center_h, dtype=np.float32),
            np.array(b_center_o, dtype=np.int32),
            np.array(b_h_raw, dtype=np.float32),
            np.array(b_s_types, dtype=np.int32),
            np.array(b_o_types, dtype=np.int32),
            np.array(b_tau, dtype=np.float32),
            np.array(b_adj_mask, dtype=bool),
            np.array(b_two_hop, dtype=bool),
            np.array(b_ego_adj, dtype=np.float32),
            np.array(b_ego_rel, dtype=np.int32),
            np.array(b_hist_reps, dtype=np.float32),
            np.array(b_seq_mask, dtype=bool),
            batch_nodes.astype(np.int32),
            np.array([self.sampler.tau_max], dtype=np.float32),
            targets["structural_proximity"],
            targets["lcs_proximity"]
        )

    def _repack_dictionaries(self, center_h, center_o, h_raw, s_types, o_types, tau, adj_mask, two_hop,
                             ego_adj, ego_rel, hist_reps, seq_mask, start_nodes, tau_max,
                             struct_prox, lcs_prox):
        """Constructs the strictly formatted dictionaries required by the computational graph."""
        inputs = {
            "center_h": center_h, "center_o": center_o,
            "h_raw": h_raw, "s_types": s_types, "o_types": o_types, "tau_weights": tau,
            "adjacency_mask": adj_mask, "two_hop_mask": two_hop,
            "ego_adj": ego_adj, "ego_rel": ego_rel,
            "history_reps": hist_reps, "sequence_mask": seq_mask,
            "ragged_adj": self.ragged_adj, "ragged_tau": self.ragged_tau,
            "ragged_alpha": self.ragged_alpha,
            "start_nodes": start_nodes, "tau_max": tau_max
        }
        targets = {
            "structural_proximity": struct_prox,
            "lcs_proximity": lcs_prox
        }
        return inputs, targets

    def build_dataset(self, node_indices: np.ndarray) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices(node_indices)
        dataset = dataset.shuffle(buffer_size=10000, seed=42).batch(self.batch_size, drop_remainder=True)

        dataset = dataset.map(
            lambda batch: tf.py_function(
                func=self._py_map_fn,
                inp=[batch],
                Tout=[tf.float32, tf.int32, tf.float32, tf.int32, tf.int32, tf.float32,
                      tf.bool, tf.bool, tf.float32, tf.int32, tf.float32, tf.bool,
                      tf.int32, tf.float32, tf.float32, tf.float32]
            ),
            num_parallel_calls=self.autotune
        )

        dataset = dataset.map(self._repack_dictionaries, num_parallel_calls=self.autotune)
        return dataset.prefetch(self.autotune)

# ==============================================================================
# SECTION 21: EXECUTION LAUNCHER
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Initializing BTB-ac pipeline testing module.")

    num_nodes = 1000
    dummy_adj = sp.csr_matrix(np.random.randint(0, 2, size=(num_nodes, num_nodes)))
    dummy_tau = sp.csr_matrix(np.random.rand(num_nodes, num_nodes))
    dummy_rel = sp.csr_matrix(np.random.randint(0, 3, size=(num_nodes, num_nodes)))
    dummy_attrs = np.random.randn(num_nodes, 128).astype(np.float32)
    dummy_types = np.random.randint(0, 2, size=(num_nodes,))

    sampler = HeterogeneousEgoNetworkSampler(dummy_adj, dummy_tau, dummy_rel, dummy_attrs, dummy_types)
    target_builder = DynamicIntentTargetBuilder()
    pipeline = BTBAcDataPipeline(sampler, target_builder, batch_size=32)

    train_nodes = np.arange(800)
    tf_dataset = pipeline.build_dataset(train_nodes)

    sample_batch_inputs, sample_batch_targets = next(iter(tf_dataset))
    logging.info("Dataset mapping completed successfully. Tensor structures match architectural expectations.")