# ==============================================================================
# "Building a Tower from Blocks" for Graph Representation Learning
# ==============================================================================

import abc
import copy
import dataclasses
import functools
import logging
import math
from typing import Optional, Tuple, List, Dict, Union, Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, constraints

# ==============================================================================
# SECTION 1: ADVANCED TYPE DEFINITIONS & EXCEPTIONS
# ==============================================================================
class GraphRepresentationLearningError(Exception):
    """Base exception for architecture-specific errors."""
    pass

class TensorSpatialInvarianceError(GraphRepresentationLearningError):
    """Raised when tensor dimensions violate structural constraints."""
    pass

class ConfigurationSemanticsError(GraphRepresentationLearningError):
    """Raised when hyperparameter configurations contradict architectural logic."""
    pass

# ==============================================================================
# SECTION 2: MATHEMATICAL METADATA & CONFIGURATION SYSTEM
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class InteractionSpaceConfig:
    num_node_types: int
    num_base_interaction_types: int
    composite_interaction_types: int
    original_attribute_dims: int
    enable_composite_interactions: bool = True

@dataclasses.dataclass(frozen=True)
class CrossGraphConvolutionConfig:
    d1_latent_space_dim: int
    d2_inter_graph_query_dim: int
    d3_intra_graph_query_dim: int
    num_attention_heads: int
    ffn_expansion_multiplier: int
    dropout_probability: float
    layer_norm_eps: float
    activation_heuristic: str = "gelu"
    kernel_initializer_strat: str = "glorot_uniform"
    apply_bias_in_projections: bool = True

@dataclasses.dataclass(frozen=True)
class GreedyWalkConfig:
    max_walk_length: int
    num_paths_per_node: int
    energy_decay_phi1_init: float
    energy_decay_phi2_init: float
    d4_semantic_query_dim: int
    d5_path_query_dim: int
    d6_inter_path_query_dim: int
    d7_global_query_dim: int

@dataclasses.dataclass(frozen=True)
class ITVMConfig:
    d8_instantaneous_query_dim: int
    d9_long_term_query_dim: int

@dataclasses.dataclass(frozen=True)
class DynamicEvolutionConfig:
    l2_regularization_lambda: float
    gradient_clip_norm: float
    curriculum_gamma_start: float
    curriculum_gamma_end: float
    curriculum_transition_steps: int

@dataclasses.dataclass(frozen=True)
class BTBAcGlobalArchitectureConfig:
    interaction_space: InteractionSpaceConfig
    cg_convolution: CrossGraphConvolutionConfig
    gw_config: GreedyWalkConfig
    itvm_config: ITVMConfig
    evolution_dynamics: DynamicEvolutionConfig
    global_random_seed: int = 42

    def validate_architectural_integrity(self):
        if self.cg_convolution.d1_latent_space_dim % self.cg_convolution.num_attention_heads != 0:
            raise ConfigurationSemanticsError(
                "d1_latent_space_dim must be strictly divisible by num_attention_heads."
            )

# ==============================================================================
# SECTION 3: TENSOR GRAPH OPERATIONS & DECORATORS
# ==============================================================================

def enforce_tensor_manifold(expected_rank: int, signature_name: str = "input") -> Callable:
    """Decorator to enforce spatial invariance across tensor representations."""
    def decorator_enforce(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper_enforce(self, *args, **kwargs):
            target_tensor = None
            if args and isinstance(args[0], (tf.Tensor, tf.RaggedTensor)):
                target_tensor = args[0]
            elif signature_name in kwargs and isinstance(kwargs[signature_name], (tf.Tensor, tf.RaggedTensor)):
                target_tensor = kwargs[signature_name]

            if target_tensor is not None:
                actual_rank = len(target_tensor.shape)
                if actual_rank != expected_rank:
                    raise TensorSpatialInvarianceError(
                        f"Expected rank {expected_rank} for {signature_name}, "
                        f"but received rank {actual_rank}. Shape constraints violated."
                    )
            return func(self, *args, **kwargs)
        return wrapper_enforce
    return decorator_enforce

class AbstractGraphRepresentationLayer(layers.Layer, metaclass=abc.ABCMeta):
    """Base abstraction for BTB-ac encoder blocks."""
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(AbstractGraphRepresentationLayer, self).__init__(**kwargs)
        self.master_config = config
        self.sys_l2_reg = regularizers.L2(config.evolution_dynamics.l2_regularization_lambda)

    def _instantiate_orthogonal_projection_matrix(self,
                                                  shape: Tuple[int, ...],
                                                  tensor_name: str) -> tf.Variable:
        return self.add_weight(
            name=tensor_name,
            shape=shape,
            initializer=self.master_config.cg_convolution.kernel_initializer_strat,
            regularizer=self.sys_l2_reg,
            trainable=True
        )

# ==============================================================================
# SECTION 4: LOW-LEVEL NEURAL UPDATERS (Upd-L operators)
# ==============================================================================

class UniversalUpdateLayer(AbstractGraphRepresentationLayer):
    """
    Implements the core Upd-L operator (Eq. 5).
    Provides residual connections, layer normalization, and FFN interactions.
    """
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(UniversalUpdateLayer, self).__init__(config, **kwargs)
        self.cg_cfg = config.cg_convolution

        self.layer_norm_alpha = layers.LayerNormalization(
            epsilon=self.cg_cfg.layer_norm_eps, name="upd_ln_alpha"
        )
        self.layer_norm_beta = layers.LayerNormalization(
            epsilon=self.cg_cfg.layer_norm_eps, name="upd_ln_beta"
        )

        d1 = self.cg_cfg.d1_latent_space_dim
        expansion = self.cg_cfg.ffn_expansion_multiplier

        self.ffn_transformation_phi = layers.Dense(
            units=d1 * expansion,
            activation=self.cg_cfg.activation_heuristic,
            kernel_regularizer=self.sys_l2_reg,
            name="upd_ffn_phi_expansion"
        )
        self.ffn_transformation_psi = layers.Dense(
            units=d1,
            kernel_regularizer=self.sys_l2_reg,
            name="upd_ffn_psi_projection"
        )
        self.stochastic_dropout_1 = layers.Dropout(self.cg_cfg.dropout_probability)
        self.stochastic_dropout_2 = layers.Dropout(self.cg_cfg.dropout_probability)

    @enforce_tensor_manifold(expected_rank=3, signature_name="identity_manifold")
    def call(self,
             identity_manifold: tf.Tensor,
             aggregated_manifold: tf.Tensor,
             training: bool = False) -> tf.Tensor:

        tf.debugging.assert_shapes({
            identity_manifold: ('B', 'N', 'D'),
            aggregated_manifold: ('B', 'N', 'D')
        })

        pre_norm_residual = identity_manifold + aggregated_manifold
        x_tilde = self.layer_norm_alpha(pre_norm_residual)

        ffn_intermediate = self.ffn_transformation_phi(x_tilde)
        ffn_intermediate = self.stochastic_dropout_1(ffn_intermediate, training=training)

        ffn_projected = self.ffn_transformation_psi(ffn_intermediate)
        ffn_projected = self.stochastic_dropout_2(ffn_projected, training=training)

        final_residual_state = x_tilde + ffn_projected
        return self.layer_norm_beta(final_residual_state)

# ==============================================================================
# SECTION 5: MODULE 1 - CROSS-GRAPH ATTENTION CONVOLUTION COMPONENTS
# ==============================================================================

class SubgraphSpecificLatentProjector(AbstractGraphRepresentationLayer):
    """Implements Eq. 1: Type-specific latent space projection."""

    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(SubgraphSpecificLatentProjector, self).__init__(config, **kwargs)
        self.cfg = config.cg_convolution
        self.num_r = config.interaction_space.composite_interaction_types
        self.num_o = config.interaction_space.num_node_types
        self.d1 = self.cfg.d1_latent_space_dim

    def build(self, input_shape: tf.TensorShape):
        max_original_dim = input_shape[-1]
        self.projection_matrices_M = self._instantiate_orthogonal_projection_matrix(
            shape=(self.num_r, self.num_o, max_original_dim, self.d1),
            tensor_name="M_r_o_latent_projector"
        )
        if self.cfg.apply_bias_in_projections:
            self.projection_biases_b = self.add_weight(
                shape=(self.num_r, self.num_o, self.d1),
                initializer="zeros",
                name="b_r_o_latent_bias",
                trainable=True
            )
        super(SubgraphSpecificLatentProjector, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, center_h: tf.Tensor, center_o: tf.Tensor, raw_neighbors: tf.Tensor,
             s_types: tf.Tensor, o_types: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        batch_size = tf.shape(center_h)[0]
        all_r = tf.range(self.num_r, dtype=tf.int32)
        r_coords = tf.tile(tf.expand_dims(all_r, 0), [batch_size, 1])
        o_coords_center = tf.tile(tf.expand_dims(center_o, 1), [1, self.num_r])
        center_coords = tf.stack([r_coords, o_coords_center], axis=-1)

        M_center = tf.gather_nd(self.projection_matrices_M, center_coords)
        h_center_proj = tf.einsum('bd,brdj->brj', center_h, M_center)

        neighbor_coords = tf.stack([tf.maximum(s_types, 0), tf.maximum(o_types, 0)], axis=-1)
        M_neighbor = tf.gather_nd(self.projection_matrices_M, neighbor_coords)
        h_neighbor_proj = tf.einsum('bnd,bndj->bnj', raw_neighbors, M_neighbor)

        if self.cfg.apply_bias_in_projections:
            b_center = tf.gather_nd(self.projection_biases_b, center_coords)
            h_center_proj += b_center
            b_neighbor = tf.gather_nd(self.projection_biases_b, neighbor_coords)
            h_neighbor_proj += b_neighbor

        return tf.nn.gelu(h_center_proj), tf.nn.gelu(h_neighbor_proj)


class CrossGraphContextualAttention(AbstractGraphRepresentationLayer):
    """Implements Encoder Block 1 (Eq. 2, 3, 4): Cross-graph neighborhood learning."""

    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(CrossGraphContextualAttention, self).__init__(config, **kwargs)
        self.cfg = config.cg_convolution
        self.num_r = config.interaction_space.composite_interaction_types
        self.num_o = config.interaction_space.num_node_types
        self.d1 = self.cfg.d1_latent_space_dim
        self.d2 = self.cfg.d2_inter_graph_query_dim
        self.sqrt_d2 = tf.constant(math.sqrt(float(self.d2)), dtype=tf.float32)
        self.update_layer = UniversalUpdateLayer(config, name="cross_graph_upd_l1")

    def build(self, input_shape: Any):
        self.W_Q1 = self._instantiate_orthogonal_projection_matrix(shape=(self.num_r, self.num_o, self.d1, self.d2), tensor_name="W_Q1_cg")
        self.W_K1 = self._instantiate_orthogonal_projection_matrix(shape=(self.num_r, self.num_o, self.d1, self.d2), tensor_name="W_K1_cg")
        self.W_V1 = self._instantiate_orthogonal_projection_matrix(shape=(self.num_r, self.num_o, self.d1, self.d1), tensor_name="W_V1_cg")
        self.omega_1 = self.add_weight(shape=(1,), initializer="ones", name="omega_1", trainable=True)
        self.alpha_t = self.add_weight(shape=(self.num_r,), initializer="ones", name="alpha_t", trainable=True)
        super(CrossGraphContextualAttention, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, h_neighbors: tf.Tensor, s_types: tf.Tensor, o_types: tf.Tensor,
             tau_weights: tf.Tensor, adj_mask: tf.Tensor, training: bool = False) -> tf.Tensor:

        safe_s = tf.maximum(s_types, 0)
        safe_o = tf.maximum(o_types, 0)
        neighbor_coords = tf.stack([safe_s, safe_o], axis=-1)

        W_Q1_sel = tf.gather_nd(self.W_Q1, neighbor_coords)
        W_K1_sel = tf.gather_nd(self.W_K1, neighbor_coords)
        W_V1_sel = tf.gather_nd(self.W_V1, neighbor_coords)

        Q = tf.einsum('bni,bnij->bnj', h_neighbors, W_Q1_sel)
        K = tf.einsum('bni,bnij->bnj', h_neighbors, W_K1_sel)
        V = tf.einsum('bni,bnij->bnj', h_neighbors, W_V1_sel)

        dot_product = tf.einsum('bmd,bnd->bmn', Q, K) / self.sqrt_d2

        alpha_t_gathered = tf.gather(self.alpha_t, safe_s)
        structural_bias = self.omega_1 * tf.expand_dims(tau_weights * alpha_t_gathered, axis=1)

        logits = dot_product + structural_bias
        inf_mask = tf.cast(tf.logical_not(tf.expand_dims(adj_mask, axis=1)), dtype=logits.dtype) * -1e9

        a_jk = tf.nn.softmax(logits + inf_mask, axis=-1)
        aggregated_message = tf.einsum('bmn,bnd->bmd', a_jk, V)

        h_j_to_i = self.update_layer(identity_manifold=h_neighbors, aggregated_manifold=aggregated_message, training=training)
        return h_j_to_i


class IntraSubgraphContextualAttention(AbstractGraphRepresentationLayer):
    """Implements Encoder Block 2 (Eq. 6): Intra-Subgraph Neighborhood Aggregation."""

    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(IntraSubgraphContextualAttention, self).__init__(config, **kwargs)
        self.cfg = config.cg_convolution
        self.num_r = config.interaction_space.composite_interaction_types
        self.num_o = config.interaction_space.num_node_types
        self.d1 = self.cfg.d1_latent_space_dim
        self.d3 = self.cfg.d3_intra_graph_query_dim
        self.sqrt_d3 = tf.constant(math.sqrt(float(self.d3)), dtype=tf.float32)
        self.update_layer = UniversalUpdateLayer(config, name="intra_subgraph_upd_l2")

    def build(self, input_shape: Any):
        self.W_Q2 = self._instantiate_orthogonal_projection_matrix(shape=(self.num_r, self.num_o, self.d1, self.d3), tensor_name="W_Q2_intra")
        self.W_K2 = self._instantiate_orthogonal_projection_matrix(shape=(self.num_r, self.num_o, self.d1, self.d3), tensor_name="W_K2_intra")
        self.W_V2 = self._instantiate_orthogonal_projection_matrix(shape=(self.num_r, self.num_o, self.d1, self.d1), tensor_name="W_V2_intra")
        self.omega_2 = self.add_weight(shape=(1,), initializer="ones", name="omega_2", trainable=True)
        super(IntraSubgraphContextualAttention, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, h_center_proj: tf.Tensor, center_o: tf.Tensor,
             h_context: tf.Tensor, s_types: tf.Tensor, o_types: tf.Tensor,
             tau_weights: tf.Tensor, adj_mask: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:

        batch_size = tf.shape(h_center_proj)[0]
        r_coords = tf.tile(tf.expand_dims(tf.range(self.num_r, dtype=tf.int32), 0), [batch_size, 1])
        center_coords = tf.stack([r_coords, tf.tile(tf.expand_dims(center_o, 1), [1, self.num_r])], axis=-1)
        neighbor_coords = tf.stack([tf.maximum(s_types, 0), tf.maximum(o_types, 0)], axis=-1)

        W_Q2_sel = tf.gather_nd(self.W_Q2, center_coords)
        W_K2_sel = tf.gather_nd(self.W_K2, neighbor_coords)
        W_V2_sel = tf.gather_nd(self.W_V2, neighbor_coords)

        Q = tf.einsum('brd,brdj->brj', h_center_proj, W_Q2_sel) # [B, R, d3]
        K = tf.einsum('bnd,bndj->bnj', h_context, W_K2_sel)     # [B, N, d3]
        V = tf.einsum('bnd,bndj->bnj', h_context, W_V2_sel)     # [B, N, d1]

        dot = tf.einsum('brd,bnd->brn', Q, K) / self.sqrt_d3

        r_expanded = tf.reshape(tf.range(self.num_r, dtype=tf.int32), (1, self.num_r, 1))
        s_expanded = tf.expand_dims(s_types, axis=1)
        type_match_mask = tf.equal(r_expanded, s_expanded) # [B, R, N]

        final_mask = tf.logical_and(type_match_mask, tf.expand_dims(adj_mask, axis=1))
        valid_r_mask = tf.reduce_any(final_mask, axis=-1)

        bias = self.omega_2 * tf.expand_dims(tau_weights, axis=1)
        logits = dot + bias

        inf_mask = tf.cast(tf.logical_not(final_mask), logits.dtype) * -1e9
        logits += inf_mask

        a_rk = tf.nn.softmax(logits, axis=-1)
        a_rk = tf.where(tf.expand_dims(valid_r_mask, -1), a_rk, tf.zeros_like(a_rk))

        agg_msg = tf.einsum('brn,bnd->brd', a_rk, V)
        l_local = self.update_layer(identity_manifold=h_center_proj, aggregated_manifold=agg_msg, training=training)

        return l_local, valid_r_mask


class SemanticStructureFusionWeighter(AbstractGraphRepresentationLayer):
    """
    Implements Eq. 7: Evaluates the fusion weights based on local representations.
    To prevent exponential computational overhead in neighborhood expansion,
    h_context is utilized as an efficient proxy for 2-hop local representations,
    preserving structural evaluation integrity.
    """

    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(SemanticStructureFusionWeighter, self).__init__(config, **kwargs)
        self.num_r = config.interaction_space.composite_interaction_types
        self.num_o = config.interaction_space.num_node_types

    def build(self, input_shape: Any):
        self.lambda_so = self.add_weight(
            shape=(self.num_r, self.num_o), initializer="ones", name="lambda_adj", trainable=True
        )
        super(SemanticStructureFusionWeighter, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, l_local: tf.Tensor, h_context: tf.Tensor,
             center_o: tf.Tensor, o_types: tf.Tensor,
             adj_mask: tf.Tensor, two_hop_mask: tf.Tensor) -> tf.Tensor:

        same_type_mask = tf.equal(o_types, tf.expand_dims(center_o, axis=1))
        extended_neighborhood_mask = tf.logical_or(adj_mask, two_hop_mask)
        valid_mask = tf.logical_and(extended_neighborhood_mask, same_type_mask)
        float_mask = tf.cast(valid_mask, tf.float32)

        norm_l = tf.nn.l2_normalize(l_local, axis=-1)
        norm_ctx = tf.nn.l2_normalize(h_context, axis=-1)
        cos_sim = tf.einsum('brd,bnd->brn', norm_l, norm_ctx)

        masked_sim = cos_sim * tf.expand_dims(float_mask, axis=1)
        avg_sim = tf.reduce_sum(masked_sim, axis=-1) / tf.maximum(tf.reduce_sum(float_mask, axis=-1, keepdims=True), 1e-9)

        r_indices = tf.tile(tf.expand_dims(tf.range(self.num_r), 0), [tf.shape(center_o)[0], 1])
        coords = tf.stack([r_indices, tf.tile(tf.expand_dims(center_o, 1), [1, self.num_r])], axis=-1)
        lambda_val = tf.gather_nd(self.lambda_so, coords)

        num_valid_neighbors = tf.reduce_sum(float_mask, axis=-1, keepdims=True)
        w_fusion = tf.nn.relu(lambda_val / tf.maximum(num_valid_neighbors, 1.0)) * (1.0 - avg_sim)
        return w_fusion


class FoundationalNodeRepresentationFusion(AbstractGraphRepresentationLayer):
    """Implements Eq. 8: Aggregates local subgraph representations into a foundational representation."""

    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(FoundationalNodeRepresentationFusion, self).__init__(config, **kwargs)
        self.cfg = config.cg_convolution
        self.num_r = config.interaction_space.composite_interaction_types
        self.num_o = config.interaction_space.num_node_types
        self.d1 = self.cfg.d1_latent_space_dim

    def build(self, input_shape: Any):
        self.U_so = self._instantiate_orthogonal_projection_matrix(
            shape=(self.num_r, self.num_o, self.d1, self.d1), tensor_name="U_s_o_fusion"
        )
        super(FoundationalNodeRepresentationFusion, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, l_local: tf.Tensor, w_fusion: tf.Tensor, alpha_t_global: tf.Tensor,
             center_o: tf.Tensor, valid_r_mask: tf.Tensor) -> tf.Tensor:

        logits = w_fusion + tf.expand_dims(alpha_t_global, axis=0)
        inf_mask = tf.cast(tf.logical_not(valid_r_mask), dtype=logits.dtype) * -1e9
        attention_weights = tf.nn.softmax(logits + inf_mask, axis=-1)
        attention_weights = tf.where(valid_r_mask, attention_weights, tf.zeros_like(attention_weights))

        batch_size = tf.shape(center_o)[0]
        r_indices = tf.tile(tf.expand_dims(tf.range(self.num_r), 0), [batch_size, 1])
        coords = tf.stack([r_indices, tf.tile(tf.expand_dims(center_o, 1), [1, self.num_r])], axis=-1)

        U_selected = tf.gather_nd(self.U_so, coords)
        l_projected = tf.einsum('brd,brdj->brj', l_local, U_selected)

        f_o_i = tf.reduce_sum(l_projected * tf.expand_dims(attention_weights, axis=-1), axis=1)
        return tf.nn.gelu(f_o_i)
# ==============================================================================
# SECTION 6: MODULE 2 - FOUNDATIONAL REPRESENTATION QUALITY EVALUATION (Eq 9-14)
# ==============================================================================

class InteractionFrequencyScorer(AbstractGraphRepresentationLayer):
    """
    Computes Eq. 9: Frequency of Interaction.
    Aggregates the total structural interaction weights for a given node.
    """
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(InteractionFrequencyScorer, self).__init__(config, **kwargs)

    @tf.function(experimental_relax_shapes=True)
    def call(self, tau_edge_weights: tf.Tensor, adjacency_mask: tf.Tensor) -> tf.Tensor:
        float_mask = tf.cast(adjacency_mask, dtype=tau_edge_weights.dtype)
        valid_weights = tau_edge_weights * float_mask

        # Shape: (batch_size,)
        d_fre_v_i = tf.reduce_sum(valid_weights, axis=[1, 2])
        return d_fre_v_i


class InteractionDispersionScorer(AbstractGraphRepresentationLayer):
    """
    Computes Eq. 10: Dispersion of Interaction Types.
    Calculates the information entropy of the interaction-type frequency distribution.
    """
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(InteractionDispersionScorer, self).__init__(config, **kwargs)

    @tf.function(experimental_relax_shapes=True)
    def call(self,
             tau_edge_weights: tf.Tensor,
             s_types: tf.Tensor,
             adjacency_mask: tf.Tensor,
             alpha_r_global: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        float_mask = tf.cast(adjacency_mask, dtype=tau_edge_weights.dtype)
        valid_weights = tau_edge_weights * float_mask

        num_r = tf.shape(alpha_r_global)[0]
        s_one_hot = tf.one_hot(tf.maximum(s_types, 0), depth=num_r)
        s_one_hot_masked = s_one_hot * tf.expand_dims(float_mask, axis=-1)

        sum_tau_per_r = tf.reduce_sum(tf.expand_dims(valid_weights, axis=-1) * s_one_hot_masked, axis=1)

        alpha_r_expanded = tf.expand_dims(alpha_r_global, axis=0)
        unnormalized_p_i_r = sum_tau_per_r * alpha_r_expanded

        normalization_factor = tf.reduce_sum(unnormalized_p_i_r, axis=1, keepdims=True)
        p_i_r = unnormalized_p_i_r / tf.maximum(normalization_factor, 1e-12)

        safe_p_i_r = tf.maximum(p_i_r, 1e-12)
        entropy = -tf.reduce_sum(p_i_r * tf.math.log(safe_p_i_r), axis=1)

        return entropy, p_i_r


class InteractionTargetDiversityScorer(AbstractGraphRepresentationLayer):
    """
    Computes Eq. 11, 12, and 13: Diversity of Interaction Targets.
    Fuses relational and characteristic diversity via trainable weights.
    """
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(InteractionTargetDiversityScorer, self).__init__(config, **kwargs)

    def build(self, input_shape: Any):
        self.w_r = self.add_weight(shape=(1,), initializer="ones", name="weight_relational_diversity", trainable=True)
        self.w_c = self.add_weight(shape=(1,), initializer="ones", name="weight_characteristic_diversity", trainable=True)
        super(InteractionTargetDiversityScorer, self).build(input_shape)

    def _calculate_relational_diversity(self, ego_adj: tf.Tensor, ego_rel: tf.Tensor,
                                        alpha_r: tf.Tensor, tau_max: tf.Tensor,
                                        adj_mask: tf.Tensor) -> tf.Tensor:
        """
        Computes Eq. 12: Relational diversity via subgraph density.
        Note: Utilizes global tau_max as an efficient engineering proxy
        for the type-specific tau_max to prevent OOM on sparse slicing.
        """
        valid_ego_mask = tf.logical_and(tf.expand_dims(adj_mask, 2), tf.expand_dims(adj_mask, 1))
        float_mask = tf.cast(valid_ego_mask, tf.float32)

        ego_rel_one_hot = tf.one_hot(tf.maximum(ego_rel, 0), depth=tf.shape(alpha_r)[0])
        alpha_r_expanded = tf.reshape(alpha_r, (1, 1, 1, -1))

        weighted_edges = tf.expand_dims(ego_adj, -1) * ego_rel_one_hot * alpha_r_expanded
        numerator = tf.reduce_sum(weighted_edges * tf.expand_dims(float_mask, -1), axis=[1, 2, 3])

        num_valid_neighbors = tf.reduce_sum(tf.cast(adj_mask, tf.float32), axis=1)
        max_possible_edges = (num_valid_neighbors * (num_valid_neighbors - 1.0)) / 2.0

        denominator = tf.maximum(max_possible_edges, 1.0) * tf.reduce_max(alpha_r) * tf.squeeze(tau_max)
        d_rel = 1.0 - (numerator / tf.maximum(denominator, 1e-12))
        return tf.nn.relu(d_rel)

    def _calculate_characteristic_diversity(self, h_raw: tf.Tensor, s_types: tf.Tensor,
                                            adj_mask: tf.Tensor, p_i_r: tf.Tensor) -> tf.Tensor:
        """Computes Eq. 13: Average semantic cosine similarity within targets."""
        norm_h = tf.nn.l2_normalize(h_raw, axis=-1)
        sim_matrix = tf.einsum('bmd,bnd->bmn', norm_h, norm_h)

        same_r_mask = tf.equal(tf.expand_dims(s_types, 2), tf.expand_dims(s_types, 1))
        valid_node_mask = tf.logical_and(tf.expand_dims(adj_mask, 2), tf.expand_dims(adj_mask, 1))

        diag_zero = 1.0 - tf.eye(tf.shape(sim_matrix)[-1], batch_shape=[tf.shape(sim_matrix)[0]])
        float_mask = tf.cast(tf.logical_and(same_r_mask, valid_node_mask), tf.float32) * diag_zero

        num_r = tf.shape(p_i_r)[1]
        s_one_hot_masked = tf.one_hot(tf.maximum(s_types, 0), depth=num_r) * tf.expand_dims(tf.cast(adj_mask, tf.float32), -1)

        r_pair_mask = tf.expand_dims(s_one_hot_masked, 2) * tf.expand_dims(s_one_hot_masked, 1)
        r_pair_mask = r_pair_mask * tf.expand_dims(float_mask, -1)

        sum_sim_per_r = tf.reduce_sum(tf.expand_dims(sim_matrix, -1) * r_pair_mask, axis=[1, 2])
        count_per_r = tf.reduce_sum(r_pair_mask, axis=[1, 2])
        avg_sim_per_r = sum_sim_per_r / tf.maximum(count_per_r, 1e-9)

        d_char = tf.reduce_sum(p_i_r * (1.0 - avg_sim_per_r), axis=1)
        return tf.nn.relu(d_char)

    @tf.function(experimental_relax_shapes=True)
    def call(self, ego_adj: tf.Tensor, ego_rel: tf.Tensor, alpha_r: tf.Tensor,
             tau_max: tf.Tensor, p_i_r: tf.Tensor, h_raw: tf.Tensor, adj_mask: tf.Tensor) -> tf.Tensor:

        d_rel = self._calculate_relational_diversity(ego_adj, ego_rel, alpha_r, tau_max, adj_mask)
        d_char = self._calculate_characteristic_diversity(h_raw, s_types=ego_rel[:, 0, :], adj_mask=adj_mask, p_i_r=p_i_r)

        return (self.w_r * d_rel) + (self.w_c * d_char)


class ComprehensiveQualityEvaluator(AbstractGraphRepresentationLayer):
    """
    Computes Eq. 14: Comprehensive Foundational Representation Score.
    Aggregates indicators via a node-type specific exponential decay function.
    """
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(ComprehensiveQualityEvaluator, self).__init__(config, **kwargs)
        self.num_o = config.interaction_space.num_node_types

    def build(self, input_shape: Any):
        self.gamma_k_o = self.add_weight(
            shape=(3, self.num_o),
            initializer=initializers.Constant(0.1),
            constraint=constraints.NonNeg(),
            name="gamma_k_o_decay_weights",
            trainable=True
        )
        super(ComprehensiveQualityEvaluator, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, d_fre: tf.Tensor, d_type: tf.Tensor, d_tar: tf.Tensor, node_type_o: tf.Tensor) -> tf.Tensor:
        d_metrics = tf.stack([d_fre, d_type, d_tar], axis=1)

        gamma_selected = tf.gather(tf.transpose(self.gamma_k_o), node_type_o)
        decay_components = tf.exp(-1.0 * gamma_selected * d_metrics)

        d_v_i = tf.reduce_prod(decay_components, axis=1)
        return d_v_i


# ==============================================================================
# SECTION 7: MODULE 2 - ADAPTIVE-TRUNCATION GREEDY WALK (Eq 15-16)
# ==============================================================================

class AdaptiveEnergyGreedyWalker(AbstractGraphRepresentationLayer):
    """Implements Eq. 15 and 16: Adaptive truncation walk based on structural and semantic quality."""

    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(AdaptiveEnergyGreedyWalker, self).__init__(config, **kwargs)
        self.gw_cfg = config.gw_config
        self.max_steps = self.gw_cfg.max_walk_length
        self.num_paths = self.gw_cfg.num_paths_per_node

    def build(self, input_shape: Any):
        self.phi_1 = self.add_weight(shape=(1,), initializer=initializers.Constant(self.gw_cfg.energy_decay_phi1_init), name="phi_1_structural_decay", trainable=True)
        self.phi_2 = self.add_weight(shape=(1,), initializer=initializers.Constant(self.gw_cfg.energy_decay_phi2_init), name="phi_2_semantic_decay", trainable=True)
        super(AdaptiveEnergyGreedyWalker, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, adjacency_tensor: tf.RaggedTensor, tau_tensor: tf.RaggedTensor,
             alpha_tensor: tf.RaggedTensor, global_quality_scores: tf.Tensor, start_nodes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        batch_size = tf.shape(start_nodes)[0]

        paths_ta = tf.TensorArray(dtype=tf.int32, size=self.max_steps, clear_after_read=False)
        energies_ta = tf.TensorArray(dtype=tf.float32, size=self.max_steps, clear_after_read=False)

        initial_paths = tf.tile(tf.expand_dims(start_nodes, 1), multiples=[1, self.num_paths])
        initial_energies = tf.ones((batch_size, self.num_paths), dtype=tf.float32)

        paths_ta = paths_ta.write(0, initial_paths)
        energies_ta = energies_ta.write(0, initial_energies)

        def walk_step(step, current_nodes, current_energies, p_ta, e_ta):
            dense_neighbors = adjacency_tensor.to_tensor(default_value=-1)
            dense_taus = tau_tensor.to_tensor(default_value=0.0)
            dense_alphas = alpha_tensor.to_tensor(default_value=0.0)

            valid_mask = tf.greater_equal(current_nodes, 0)
            safe_current_nodes = tf.maximum(current_nodes, 0)

            neighbors = tf.gather(dense_neighbors, safe_current_nodes)
            edge_taus = tf.gather(dense_taus, safe_current_nodes)
            edge_alphas = tf.gather(dense_alphas, safe_current_nodes)

            valid_neighbor_mask = tf.greater_equal(neighbors, 0)
            row_lengths = tf.reduce_sum(tf.cast(valid_neighbor_mask, tf.int32), axis=-1)
            has_neighbors = tf.logical_and(row_lengths > 0, valid_mask)

            max_indices = tf.maximum(row_lengths - 1, 0)
            random_floats = tf.random.uniform(tf.shape(max_indices), minval=0.0, maxval=1.0)
            sampled_indices = tf.cast(tf.math.floor(random_floats * tf.cast(row_lengths, tf.float32)), tf.int32)

            batch_indices_mesh = tf.range(batch_size, dtype=tf.int32)
            batch_indices = tf.tile(tf.expand_dims(batch_indices_mesh, 1), multiples=[1, self.num_paths])
            path_indices = tf.tile(tf.expand_dims(tf.range(self.num_paths, dtype=tf.int32), 0), multiples=[batch_size, 1])
            gather_indices = tf.stack([batch_indices, path_indices, sampled_indices], axis=-1)

            next_nodes = tf.where(has_neighbors, tf.gather_nd(neighbors, gather_indices), current_nodes)
            next_taus = tf.where(has_neighbors, tf.gather_nd(edge_taus, gather_indices), tf.zeros_like(current_energies))
            next_alphas = tf.where(has_neighbors, tf.gather_nd(edge_alphas, gather_indices), tf.zeros_like(current_energies))

            alpha_tau_product = edge_alphas * edge_taus * tf.cast(valid_neighbor_mask, tf.float32)
            sum_alpha_tau = tf.reduce_sum(alpha_tau_product, axis=-1)
            sum_alpha_tau_gathered = tf.gather_nd(sum_alpha_tau, tf.stack([batch_indices, path_indices], axis=-1))

            norm_alpha_tau = (next_alphas * next_taus) / tf.maximum(sum_alpha_tau_gathered, 1e-9)

            safe_next_nodes = tf.maximum(next_nodes, 0)
            d_v_j = tf.gather(global_quality_scores, safe_next_nodes)

            structural_penalty = self.phi_1 * norm_alpha_tau
            semantic_penalty = self.phi_2 * d_v_j
            energy_drop = 1.0 / (1.0 + tf.exp(structural_penalty + semantic_penalty))

            next_energies = current_energies - energy_drop

            alive_mask = next_energies > 0.0
            next_nodes = tf.where(alive_mask, next_nodes, tf.fill(tf.shape(next_nodes), -1))
            next_energies = tf.where(alive_mask, next_energies, tf.zeros_like(next_energies))

            p_ta = p_ta.write(step, next_nodes)
            e_ta = e_ta.write(step, next_energies)

            return step + 1, next_nodes, next_energies, p_ta, e_ta

        _, _, _, final_p_ta, final_e_ta = tf.while_loop(
            cond=lambda step, *args: step < self.max_steps,
            body=walk_step,
            loop_vars=(tf.constant(1, dtype=tf.int32), initial_paths, initial_energies, paths_ta, energies_ta)
        )

        final_paths_tensor = tf.transpose(final_p_ta.stack(), perm=[1, 2, 0])
        final_energies_tensor = tf.transpose(final_e_ta.stack(), perm=[1, 2, 0])

        return final_paths_tensor, final_energies_tensor


# ==============================================================================
# SECTION 8: MODULE 2 - ENCODER BLOCKS 3 TO 6 (Eq 17-20)
# ==============================================================================

class IntraPathSemanticEncoder(AbstractGraphRepresentationLayer):
    """Implements Encoder Block 3 (Eq. 17): Semantic representation learning within paths."""
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(IntraPathSemanticEncoder, self).__init__(config, **kwargs)
        self.d4 = config.gw_config.d4_semantic_query_dim
        self.sqrt_d4 = tf.constant(math.sqrt(float(self.d4)), dtype=tf.float32)
        self.d1 = config.cg_convolution.d1_latent_space_dim
        self.upd_l3 = UniversalUpdateLayer(config, name="path_upd_l3")

    def build(self, input_shape: Any):
        self.W_Q3 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d4), tensor_name="W_Q3_path")
        self.W_K3 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d4), tensor_name="W_K3_path")
        self.W_V3 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d1), tensor_name="W_V3_path")
        self.omega_3 = self.add_weight(shape=(1,), initializer="ones", name="omega_3", trainable=True)
        super(IntraPathSemanticEncoder, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, f_theta: tf.Tensor, energies: tf.Tensor, path_mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        Q3_proj = tf.einsum('bpnd,dk->bpnk', f_theta, self.W_Q3)
        K3_proj = tf.einsum('bpnd,dk->bpnk', f_theta, self.W_K3)
        V3_proj = tf.einsum('bpnd,dk->bpnk', f_theta, self.W_V3)

        Q_expanded = tf.expand_dims(Q3_proj, axis=3)
        K_expanded = tf.expand_dims(K3_proj, axis=2)
        dot_product = tf.reduce_sum(Q_expanded * K_expanded, axis=-1) / self.sqrt_d4

        E_j = tf.expand_dims(energies, axis=3)
        E_k = tf.expand_dims(energies, axis=2)
        energy_distance = tf.abs(E_k - E_j)

        logits = dot_product + (self.omega_3 * energy_distance)

        valid_2d_mask = tf.logical_and(tf.expand_dims(path_mask, axis=3), tf.expand_dims(path_mask, axis=2))
        inf_mask = tf.cast(tf.logical_not(valid_2d_mask), dtype=logits.dtype) * -1e9

        c_jk = tf.nn.softmax(logits + inf_mask, axis=-1)
        c_jk = tf.where(valid_2d_mask, c_jk, tf.zeros_like(c_jk))

        c_jk_expanded = tf.expand_dims(c_jk, axis=-1)
        V3_expanded = tf.expand_dims(V3_proj, axis=2)
        aggregated_message = tf.reduce_sum(c_jk_expanded * V3_expanded, axis=3)

        batch_paths_shape = [-1, tf.shape(f_theta)[2], self.d1]
        s_theta = self.upd_l3(
            identity_manifold=tf.reshape(f_theta, batch_paths_shape),
            aggregated_manifold=tf.reshape(aggregated_message, batch_paths_shape),
            training=training
        )
        return tf.reshape(s_theta, tf.shape(f_theta))


class PathInformationExtractor(AbstractGraphRepresentationLayer):
    """Implements Encoder Block 4 (Eq. 18): Path extraction relative to central node."""
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(PathInformationExtractor, self).__init__(config, **kwargs)
        self.d5 = config.gw_config.d5_path_query_dim
        self.sqrt_d5 = tf.constant(math.sqrt(float(self.d5)), dtype=tf.float32)
        self.d1 = config.cg_convolution.d1_latent_space_dim
        self.upd_l4 = UniversalUpdateLayer(config, name="path_upd_l4")

    def build(self, input_shape: Any):
        self.W_Q4 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d5), tensor_name="W_Q4_extract")
        self.W_K4 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d5), tensor_name="W_K4_extract")
        self.W_V4 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d1), tensor_name="W_V4_extract")
        self.omega_4 = self.add_weight(shape=(1,), initializer="ones", name="omega_4", trainable=True)
        super(PathInformationExtractor, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, f_o_i: tf.Tensor, s_theta: tf.Tensor, energies: tf.Tensor, path_mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        Q4_proj = tf.einsum('bd,dk->bk', f_o_i, self.W_Q4)
        K4_proj = tf.einsum('bpnd,dk->bpnk', s_theta, self.W_K4)
        V4_proj = tf.einsum('bpnd,dk->bpnk', s_theta, self.W_V4)

        Q_expanded = tf.reshape(Q4_proj, (-1, 1, 1, self.d5))
        dot_product = tf.reduce_sum(Q_expanded * K4_proj, axis=-1) / self.sqrt_d5

        logits = dot_product + (self.omega_4 * energies)

        inf_mask = tf.cast(tf.logical_not(path_mask), dtype=logits.dtype) * -1e9
        d_j_theta = tf.nn.softmax(logits + inf_mask, axis=-1)
        d_j_theta = tf.where(path_mask, d_j_theta, tf.zeros_like(d_j_theta))

        d_j_expanded = tf.expand_dims(d_j_theta, axis=-1)
        X_theta_pre = tf.reduce_sum(d_j_expanded * V4_proj, axis=2)

        zero_identity = tf.zeros_like(X_theta_pre)
        X_theta = self.upd_l4(identity_manifold=zero_identity, aggregated_manifold=X_theta_pre, training=training)
        return X_theta


class InterPathInteractionEncoder(AbstractGraphRepresentationLayer):
    """Implements Encoder Block 5 (Eq. 19): Mutual learning among paths."""
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(InterPathInteractionEncoder, self).__init__(config, **kwargs)
        self.d6 = config.gw_config.d6_inter_path_query_dim
        self.sqrt_d6 = tf.constant(math.sqrt(float(self.d6)), dtype=tf.float32)
        self.d1 = config.cg_convolution.d1_latent_space_dim
        self.upd_l5 = UniversalUpdateLayer(config, name="path_upd_l5")

    def build(self, input_shape: Any):
        self.W_Q5 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d6), tensor_name="W_Q5_inter")
        self.W_K5 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d6), tensor_name="W_K5_inter")
        self.W_V5 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d1), tensor_name="W_V5_inter")
        self.omega_5 = self.add_weight(shape=(1,), initializer="ones", name="omega_5", trainable=True)
        super(InterPathInteractionEncoder, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, X_theta: tf.Tensor, path_lengths: tf.Tensor, training: bool = False) -> tf.Tensor:
        Q5_proj = tf.einsum('bpd,dk->bpk', X_theta, self.W_Q5)
        K5_proj = tf.einsum('bpd,dk->bpk', X_theta, self.W_K5)
        V5_proj = tf.einsum('bpd,dk->bpk', X_theta, self.W_V5)

        Q_expanded = tf.expand_dims(Q5_proj, axis=2)
        K_expanded = tf.expand_dims(K5_proj, axis=1)
        dot_product = tf.reduce_sum(Q_expanded * K_expanded, axis=-1) / self.sqrt_d6

        L_k_cast = tf.cast(tf.expand_dims(path_lengths, axis=1), tf.float32)
        logits = dot_product + (self.omega_5 * L_k_cast)

        valid_paths = tf.cast(path_lengths > 0, tf.bool)
        valid_2d_mask = tf.logical_and(tf.expand_dims(valid_paths, 2), tf.expand_dims(valid_paths, 1))
        inf_mask = tf.cast(tf.logical_not(valid_2d_mask), dtype=logits.dtype) * -1e9

        corr_matrix = tf.nn.softmax(logits + inf_mask, axis=-1)
        corr_matrix = tf.where(valid_2d_mask, corr_matrix, tf.zeros_like(corr_matrix))

        corr_expanded = tf.expand_dims(corr_matrix, axis=-1)
        V_expanded = tf.expand_dims(V5_proj, axis=1)
        aggregated_paths = tf.reduce_sum(corr_expanded * V_expanded, axis=2)

        Y_theta = self.upd_l5(identity_manifold=X_theta, aggregated_manifold=aggregated_paths, training=training)
        return Y_theta


class GlobalRepresentationAggregator(AbstractGraphRepresentationLayer):
    """Implements Encoder Block 6 (Eq. 20): Constructs final global representation."""
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(GlobalRepresentationAggregator, self).__init__(config, **kwargs)
        self.d7 = config.gw_config.d7_global_query_dim
        self.sqrt_d7 = tf.constant(math.sqrt(float(self.d7)), dtype=tf.float32)
        self.d1 = config.cg_convolution.d1_latent_space_dim
        self.upd_l6 = UniversalUpdateLayer(config, name="path_upd_l6")

    def build(self, input_shape: Any):
        self.W_Q6 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d7), tensor_name="W_Q6_global")
        self.W_K6 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d7), tensor_name="W_K6_global")
        self.W_V6 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d1), tensor_name="W_V6_global")
        self.omega_6 = self.add_weight(shape=(1,), initializer="ones", name="omega_6", trainable=True)
        super(GlobalRepresentationAggregator, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, f_o_i: tf.Tensor, Y_theta: tf.Tensor, path_lengths: tf.Tensor, training: bool = False) -> tf.Tensor:
        Q6_proj = tf.einsum('bd,dk->bk', f_o_i, self.W_Q6)
        K6_proj = tf.einsum('bpd,dk->bpk', Y_theta, self.W_K6)
        V6_proj = tf.einsum('bpd,dk->bpk', Y_theta, self.W_V6)

        Q_expanded = tf.expand_dims(Q6_proj, axis=1)
        dot_product = tf.reduce_sum(Q_expanded * K6_proj, axis=-1) / self.sqrt_d7

        L_k_cast = tf.cast(path_lengths, tf.float32)
        logits = dot_product + (self.omega_6 * L_k_cast)

        valid_paths = tf.cast(path_lengths > 0, tf.bool)
        inf_mask = tf.cast(tf.logical_not(valid_paths), dtype=logits.dtype) * -1e9

        corr_weights = tf.nn.softmax(logits + inf_mask, axis=-1)
        corr_weights = tf.where(valid_paths, corr_weights, tf.zeros_like(corr_weights))

        corr_expanded = tf.expand_dims(corr_weights, axis=-1)
        aggregated_global = tf.reduce_sum(corr_expanded * V6_proj, axis=1)

        identity_expanded = tf.expand_dims(f_o_i, axis=1)
        agg_expanded = tf.expand_dims(aggregated_global, axis=1)

        g_o_i = self.upd_l6(identity_manifold=identity_expanded, aggregated_manifold=agg_expanded, training=training)
        return tf.squeeze(g_o_i, axis=1)

# ==============================================================================
# SECTION 9: MODULE 3 - INTEREST TRANSFER VECTOR MODULE (Eq 21-23)
# ==============================================================================

class InstantaneousTransferVectorGenerator(AbstractGraphRepresentationLayer):
    """
    Computes instantaneous interest transfer vectors from sequential interaction histories.
    Generates Delta vectors representing dynamic behavioral shifts.
    """
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(InstantaneousTransferVectorGenerator, self).__init__(config, **kwargs)

    @tf.function(experimental_relax_shapes=True)
    def call(self, interaction_target_representations: tf.Tensor, sequence_mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        subsequent_reps = interaction_target_representations[:, 1:, :]
        precedent_reps = interaction_target_representations[:, :-1, :]
        delta_vectors = subsequent_reps - precedent_reps

        subsequent_mask = sequence_mask[:, 1:]
        precedent_mask = sequence_mask[:, :-1]
        delta_mask = tf.logical_and(subsequent_mask, precedent_mask)

        float_delta_mask = tf.expand_dims(tf.cast(delta_mask, dtype=delta_vectors.dtype), axis=-1)
        clean_delta_vectors = delta_vectors * float_delta_mask

        return clean_delta_vectors, delta_mask


class IntraSequenceMutualLearningEncoder(AbstractGraphRepresentationLayer):
    """Implements Encoder Block 7 (Eq. 21): Mutual learning among instantaneous interest transfer vectors."""
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(IntraSequenceMutualLearningEncoder, self).__init__(config, **kwargs)
        self.itvm_cfg = config.itvm_config
        self.d1 = config.cg_convolution.d1_latent_space_dim
        self.d8 = self.itvm_cfg.d8_instantaneous_query_dim
        self.sqrt_d8 = tf.constant(math.sqrt(float(self.d8)), dtype=tf.float32)
        self.upd_l7 = UniversalUpdateLayer(config, name="itvm_upd_l7")

    def build(self, input_shape: Any):
        self.W_Q7 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d8), tensor_name="W_Q7_itvm")
        self.W_K7 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d8), tensor_name="W_K7_itvm")
        self.W_V7 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d1), tensor_name="W_V7_itvm")
        super(IntraSequenceMutualLearningEncoder, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, delta_vectors: tf.Tensor, delta_mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        Q7_proj = tf.einsum('bsd,dk->bsk', delta_vectors, self.W_Q7)
        K7_proj = tf.einsum('bsd,dk->bsk', delta_vectors, self.W_K7)
        V7_proj = tf.einsum('bsd,dk->bsk', delta_vectors, self.W_V7)

        dot_product = tf.einsum('bmd,bnd->bmn', Q7_proj, K7_proj) / self.sqrt_d8

        mask_expanded = tf.expand_dims(delta_mask, axis=1)
        inf_mask = tf.cast(tf.logical_not(mask_expanded), dtype=dot_product.dtype) * -1e9

        attention_weights = tf.nn.softmax(dot_product + inf_mask, axis=-1)
        aggregated_delta = tf.einsum('bmn,bnd->bmd', attention_weights, V7_proj)

        tilde_delta = self.upd_l7(identity_manifold=delta_vectors, aggregated_manifold=aggregated_delta, training=training)
        return tilde_delta


class LongTermInterestFusionEncoder(AbstractGraphRepresentationLayer):
    """Implements Encoder Block 8 (Eq. 22): Fusion into the long-term interest transfer vector."""
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(LongTermInterestFusionEncoder, self).__init__(config, **kwargs)
        self.itvm_cfg = config.itvm_config
        self.d1 = config.cg_convolution.d1_latent_space_dim
        self.d9 = self.itvm_cfg.d9_long_term_query_dim
        self.sqrt_d9 = tf.constant(math.sqrt(float(self.d9)), dtype=tf.float32)
        self.upd_l8 = UniversalUpdateLayer(config, name="itvm_upd_l8")

    def build(self, input_shape: Any):
        self.W_Q8 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d9), tensor_name="W_Q8_itvm")
        self.W_K8 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d9), tensor_name="W_K8_itvm")
        self.W_V8 = self._instantiate_orthogonal_projection_matrix(shape=(self.d1, self.d1), tensor_name="W_V8_itvm")
        super(LongTermInterestFusionEncoder, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, g_o_i: tf.Tensor, tilde_delta: tf.Tensor, delta_mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        Q8_proj = tf.einsum('bd,dk->bk', g_o_i, self.W_Q8)
        K8_proj = tf.einsum('bnd,dk->bnk', tilde_delta, self.W_K8)
        V8_proj = tf.einsum('bnd,dk->bnk', tilde_delta, self.W_V8)

        dot_product = tf.einsum('bd,bnd->bn', Q8_proj, K8_proj) / self.sqrt_d9

        inf_mask = tf.cast(tf.logical_not(delta_mask), dtype=dot_product.dtype) * -1e9
        attention_weights = tf.nn.softmax(dot_product + inf_mask, axis=-1)

        aggregated_T_o_i = tf.einsum('bn,bnd->bd', attention_weights, V8_proj)

        zero_identity = tf.zeros_like(aggregated_T_o_i)

        # Expand to Rank 3 to satisfy UniversalUpdateLayer's spatial invariance constraints
        T_o_i = self.upd_l8(
            identity_manifold=tf.expand_dims(zero_identity, axis=1),
            aggregated_manifold=tf.expand_dims(aggregated_T_o_i, axis=1),
            training=training
        )
        return tf.squeeze(T_o_i, axis=1)


class DynamicRepresentationGating(AbstractGraphRepresentationLayer):
    """Implements Eq. 23: Dynamic Gating Mechanism."""
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(DynamicRepresentationGating, self).__init__(config, **kwargs)
        self.d1 = config.cg_convolution.d1_latent_space_dim

    def build(self, input_shape: Any):
        self.G_matrix = self._instantiate_orthogonal_projection_matrix(
            shape=(self.d1 * 2, self.d1), tensor_name="G_matrix_gating"
        )
        self.t_bias = self.add_weight(shape=(self.d1,), initializer="zeros", name="t_bias_gating", trainable=True)
        super(DynamicRepresentationGating, self).build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, g_o_i: tf.Tensor, T_o_i: tf.Tensor) -> tf.Tensor:
        # Concatenate along the feature dimension
        concat_features = tf.concat([g_o_i, T_o_i], axis=-1)

        gate_logits = tf.matmul(concat_features, self.G_matrix) + self.t_bias
        gate_activation = tf.nn.sigmoid(gate_logits)

        H_o_i = g_o_i + (gate_activation * T_o_i)
        return H_o_i

# ==============================================================================
# SECTION 10: MODULE 4 - CO-EVOLUTIONARY SELF-SUPERVISION FRAMEWORK (Eq 24-30)
# ==============================================================================

class CoEvolutionarySelfSupervisionEngine(AbstractGraphRepresentationLayer):
    """
    Main objective function calculating module-specific and global synergistic loss
    components to ensure unified evolutionary direction across the architecture.
    """
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(CoEvolutionarySelfSupervisionEngine, self).__init__(config, **kwargs)
        self.dyn_cfg = config.evolution_dynamics

    def _compute_cosine_similarity_matrix(self, matrix_a: tf.Tensor, matrix_b: tf.Tensor) -> tf.Tensor:
        norm_a = tf.nn.l2_normalize(matrix_a, axis=-1)
        norm_b = tf.nn.l2_normalize(matrix_b, axis=-1)
        return tf.matmul(norm_a, norm_b, transpose_b=True)

    @tf.function(experimental_relax_shapes=True)
    def compute_local_structure_fidelity_loss(self, f_foundational: tf.Tensor,
                                              structural_proximity_matrix: tf.Tensor) -> tf.Tensor:
        """Implements Eq. 24: Local Structure Fidelity Loss."""
        cos_sim_matrix = self._compute_cosine_similarity_matrix(f_foundational, f_foundational)
        squared_error = tf.square(cos_sim_matrix - structural_proximity_matrix)

        # Exclude self-alignment (diagonal mask)
        diag_zero_mask = 1.0 - tf.eye(tf.shape(f_foundational)[0], dtype=squared_error.dtype)
        return tf.reduce_sum(squared_error * diag_zero_mask, axis=-1)

    @tf.function(experimental_relax_shapes=True)
    def compute_global_path_discriminability_loss(self, g_global: tf.Tensor, path_reps_P: tf.Tensor,
                                                  path_mask: tf.Tensor) -> tf.Tensor:
        """
        Implements Eq. 25: Global Path Discriminability Loss with diversity penalty.
        Utilizes numerical stabilization (l2_norm and safe denominators) for the contrastive task.
        """
        mask_expanded = tf.expand_dims(tf.cast(path_mask, dtype=path_reps_P.dtype), axis=-1)
        valid_path_reps = path_reps_P * mask_expanded

        # G_aggregated corresponds to the sum of all path representations
        G_aggregated = tf.reduce_sum(valid_path_reps, axis=1)

        norm_g = tf.nn.l2_normalize(g_global, axis=-1)
        norm_G = tf.nn.l2_normalize(G_aggregated, axis=-1)

        pos_sim = tf.reduce_sum(norm_g * norm_G, axis=-1)
        all_sim_matrix = tf.matmul(norm_g, norm_G, transpose_b=True)

        numerator = tf.exp(pos_sim)
        denominator = tf.reduce_sum(tf.exp(all_sim_matrix), axis=-1)
        contrastive_loss = -tf.math.log(numerator / tf.maximum(denominator, 1e-12))

        # Enforce semantic diversity among distinct paths from the same node
        norm_P = tf.nn.l2_normalize(valid_path_reps, axis=-1)
        path_sim_matrix = tf.einsum('bmd,bnd->bmn', norm_P, norm_P)

        num_paths = tf.shape(norm_P)[1]
        diag_zero = 1.0 - tf.eye(num_paths, dtype=path_sim_matrix.dtype)
        valid_sim = tf.abs(path_sim_matrix) * tf.expand_dims(diag_zero, axis=0)

        # Aggregate along the path quantity dimension
        diversity_penalty = tf.reduce_sum(valid_sim, axis=[1, 2])

        return contrastive_loss + diversity_penalty

    @tf.function(experimental_relax_shapes=True)
    def compute_dynamic_intent_alignment_loss(self, H_enhanced: tf.Tensor,
                                              lcs_proximity_matrix: tf.Tensor) -> tf.Tensor:
        """Implements Eq. 26: Dynamic Intent Alignment Loss."""
        cos_sim_matrix = self._compute_cosine_similarity_matrix(H_enhanced, H_enhanced)
        squared_error = tf.square(cos_sim_matrix - lcs_proximity_matrix)

        diag_zero_mask = 1.0 - tf.eye(tf.shape(H_enhanced)[0], dtype=squared_error.dtype)
        return tf.reduce_sum(squared_error * diag_zero_mask, axis=-1)

    @tf.function(experimental_relax_shapes=True)
    def compute_synergistic_regularization(self, f_foundational: tf.Tensor, g_global: tf.Tensor,
                                           H_enhanced: tf.Tensor, D_quality_scores: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Implements Eq. 27 and Eq. 28: Global Synergistic Co-evolution Regularization."""
        dist_g_f = tf.norm(g_global - f_foundational, ord='euclidean', axis=-1)
        dist_H_g = tf.norm(H_enhanced - g_global, ord='euclidean', axis=-1)

        # Eq. 27: Adaptive Responsibility Allocation
        responsibility_allocation = -1.0 * (1.0 - D_quality_scores) * (dist_g_f + dist_H_g)

        # Eq. 28: Orthogonal Information Supplement
        norm_gf = tf.nn.l2_normalize(g_global - f_foundational, axis=-1)
        norm_f = tf.nn.l2_normalize(f_foundational, axis=-1)
        cos_1 = tf.abs(tf.reduce_sum(norm_gf * norm_f, axis=-1))

        norm_Hg = tf.nn.l2_normalize(H_enhanced - g_global, axis=-1)
        norm_g = tf.nn.l2_normalize(g_global, axis=-1)
        cos_2 = tf.abs(tf.reduce_sum(norm_Hg * norm_g, axis=-1))

        orthogonal_supplement = cos_1 + cos_2

        return responsibility_allocation, orthogonal_supplement

    @tf.function(experimental_relax_shapes=True)
    def compute_total_coevolution_loss(self, current_step: tf.Tensor, f_foundational: tf.Tensor,
                                       g_global: tf.Tensor, H_enhanced: tf.Tensor, path_reps_P: tf.Tensor,
                                       path_mask: tf.Tensor, D_quality_scores: tf.Tensor,
                                       structural_prox_matrix: tf.Tensor, lcs_prox_matrix: tf.Tensor) -> tf.Tensor:
        """Orchestrates Eq. 29 and Eq. 30: Total loss dynamically scaled by curriculum scheduler."""
        total_steps = tf.cast(self.dyn_cfg.curriculum_transition_steps, tf.float32)
        step_cast = tf.cast(current_step, tf.float32)

        gamma_t = tf.clip_by_value(step_cast / total_steps,
                                   clip_value_min=self.dyn_cfg.curriculum_gamma_start,
                                   clip_value_max=self.dyn_cfg.curriculum_gamma_end)

        # Eq. 29: Dynamic node-wise weighting transitioning from easy to hard structures
        w_i_t = ((1.0 - gamma_t) * D_quality_scores) + (gamma_t * (1.0 - D_quality_scores))

        L_lo = self.compute_local_structure_fidelity_loss(f_foundational, structural_prox_matrix)
        L_gl = self.compute_global_path_discriminability_loss(g_global, path_reps_P, path_mask)
        L_dy = self.compute_dynamic_intent_alignment_loss(H_enhanced, lcs_prox_matrix)
        R_res, R_or = self.compute_synergistic_regularization(f_foundational, g_global, H_enhanced, D_quality_scores)

        # Eq. 30: Final objective minimization
        node_wise_total_loss = w_i_t * (L_lo + L_gl + L_dy + R_res + R_or)
        batch_total_loss = tf.reduce_mean(node_wise_total_loss)
        return batch_total_loss


# ==============================================================================
# SECTION 11 & 12: METRICS & SCHEDULER CALLBACK (Monitoring Eq 24, 25, 29)
# ==============================================================================

class TopologicalFidelityMetric(tf.keras.metrics.Metric):
    """Tracks the semantic alignment with structural proximity across training epochs."""
    def __init__(self, name="topological_fidelity_score", **kwargs):
        super(TopologicalFidelityMetric, self).__init__(name=name, **kwargs)
        self.cumulative_fidelity = self.add_weight(name="cum_fidelity", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

    @tf.function(experimental_relax_shapes=True)
    def update_state(self, structural_proximity_matrix: tf.Tensor, f_foundational: tf.Tensor, sample_weight=None):
        norm_f = tf.nn.l2_normalize(f_foundational, axis=-1)
        cos_sim_matrix = tf.matmul(norm_f, norm_f, transpose_b=True)
        diag_zero_mask = 1.0 - tf.eye(tf.shape(f_foundational)[0], dtype=cos_sim_matrix.dtype)
        absolute_error = tf.abs(cos_sim_matrix - structural_proximity_matrix) * diag_zero_mask

        mean_error = tf.reduce_sum(absolute_error) / tf.maximum(tf.reduce_sum(diag_zero_mask), 1.0)
        fidelity_score = 1.0 - mean_error

        batch_size = tf.cast(tf.shape(f_foundational)[0], tf.float32)
        self.cumulative_fidelity.assign_add(fidelity_score * batch_size)
        self.total_samples.assign_add(batch_size)

    def result(self):
        return self.cumulative_fidelity / tf.maximum(self.total_samples, 1e-9)

    def reset_state(self):
        self.cumulative_fidelity.assign(0.0)
        self.total_samples.assign(0.0)

class PathDiscriminabilityMetric(tf.keras.metrics.Metric):
    """Tracks the accuracy of the global path contrastive discrimination task."""
    def __init__(self, name="path_discriminability", **kwargs):
        super(PathDiscriminabilityMetric, self).__init__(name=name, **kwargs)
        self.contrastive_acc = self.add_weight(name="contrastive_acc", initializer="zeros")
        self.total_nodes = self.add_weight(name="total_nodes", initializer="zeros")

    @tf.function(experimental_relax_shapes=True)
    def update_state(self, g_global: tf.Tensor, path_reps_P: tf.Tensor, path_mask: tf.Tensor, sample_weight=None):
        mask_expanded = tf.expand_dims(tf.cast(path_mask, dtype=path_reps_P.dtype), axis=-1)
        G_aggregated = tf.reduce_sum(path_reps_P * mask_expanded, axis=1)

        norm_g = tf.nn.l2_normalize(g_global, axis=-1)
        norm_G = tf.nn.l2_normalize(G_aggregated, axis=-1)

        similarity_matrix = tf.matmul(norm_g, norm_G, transpose_b=True)
        predictions = tf.argmax(similarity_matrix, axis=-1)
        labels = tf.range(tf.shape(g_global)[0], dtype=tf.int64)

        correct_predictions = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32))

        self.contrastive_acc.assign_add(correct_predictions)
        self.total_nodes.assign_add(tf.cast(tf.shape(g_global)[0], tf.float32))

    def result(self):
        return self.contrastive_acc / tf.maximum(self.total_nodes, 1e-9)

    def reset_state(self):
        self.contrastive_acc.assign(0.0)
        self.total_nodes.assign(0.0)

class CurriculumEvolutionScheduler(tf.keras.callbacks.Callback):
    """Maintains the global step state governing Eq. 29's phase transitions."""
    def __init__(self, master_config: BTBAcGlobalArchitectureConfig):
        super(CurriculumEvolutionScheduler, self).__init__()
        self.dyn_cfg = master_config.evolution_dynamics
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="curriculum_global_step")

    def on_train_batch_begin(self, batch, logs=None):
        self.global_step.assign_add(1)

# ==============================================================================
# SECTION 13: SERIALIZATION & CONFIGURATION ROUTING
# ==============================================================================

def serialize_dataclass_to_dict(dc_instance: Any) -> Dict:
    """Recursively serializes a dataclass instance into a dictionary."""
    if not dataclasses.is_dataclass(dc_instance):
        return dc_instance
    result = {}
    for field in dataclasses.fields(dc_instance):
        value = getattr(dc_instance, field.name)
        if dataclasses.is_dataclass(value):
            result[field.name] = serialize_dataclass_to_dict(value)
        else:
            result[field.name] = value
    return result

def deserialize_dict_to_dataclass(data: Dict, dc_class: Any) -> Any:
    """Recursively reconstructs a dataclass from a dictionary."""
    field_types = {f.name: f.type for f in dataclasses.fields(dc_class)}
    init_kwargs = {}
    for key, value in data.items():
        if key in field_types:
            target_type = field_types[key]
            if hasattr(target_type, "__dataclass_fields__"):
                init_kwargs[key] = deserialize_dict_to_dataclass(value, target_type)
            else:
                init_kwargs[key] = value
    return dc_class(**init_kwargs)

class BTBSerializableMixin:
    """Ensures Keras model serialization compatibility for custom architectures."""
    def get_config(self):
        config = super().get_config()
        config.update({"master_config": serialize_dataclass_to_dict(self.master_config)})
        return config

    @classmethod
    def from_config(cls, config):
        master_config_dict = config.pop("master_config")
        interaction_cfg = deserialize_dict_to_dataclass(master_config_dict.get("interaction_space", {}), InteractionSpaceConfig)
        cg_cfg = deserialize_dict_to_dataclass(master_config_dict.get("cg_convolution", {}), CrossGraphConvolutionConfig)
        gw_cfg = deserialize_dict_to_dataclass(master_config_dict.get("gw_config", {}), GreedyWalkConfig)
        itvm_cfg = deserialize_dict_to_dataclass(master_config_dict.get("itvm_config", {}), ITVMConfig)
        dyn_cfg = deserialize_dict_to_dataclass(master_config_dict.get("evolution_dynamics", {}), DynamicEvolutionConfig)

        rebuilt_master_config = BTBAcGlobalArchitectureConfig(
            interaction_space=interaction_cfg, cg_convolution=cg_cfg,
            gw_config=gw_cfg, itvm_config=itvm_cfg, evolution_dynamics=dyn_cfg,
            global_random_seed=master_config_dict.get("global_random_seed", 42)
        )
        return cls(config=rebuilt_master_config, **config)

@tf.keras.utils.register_keras_serializable(package="BTBAc")
class ExactGraphTensorRouter(AbstractGraphRepresentationLayer, BTBSerializableMixin):
    """
    Safely routes complex nested dictionaries to specialized computational modules.
    Guarantees structural mapping matching the asynchronous data pipeline.
    """
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        AbstractGraphRepresentationLayer.__init__(self, config, **kwargs)
        BTBSerializableMixin.__init__(self)

    @tf.function(experimental_relax_shapes=True)
    def extract_local_topology(self, inputs: Dict[str, tf.Tensor]) -> Tuple:
        return (inputs['center_h'], inputs['center_o'], inputs['h_raw'],
                inputs['s_types'], inputs['o_types'], inputs['tau_weights'],
                inputs['adjacency_mask'], inputs['two_hop_mask'],
                inputs['ego_adj'], inputs['ego_rel'])

    @tf.function(experimental_relax_shapes=True)
    def extract_dynamic_intent_sequences(self, inputs: Dict[str, tf.Tensor]) -> Tuple:
        return inputs['history_reps'], inputs['sequence_mask']

    @tf.function(experimental_relax_shapes=True)
    def extract_greedy_walk_components(self, inputs: Dict[str, Union[tf.Tensor, tf.RaggedTensor]]) -> Tuple:
        return (inputs['ragged_adj'], inputs['ragged_tau'], inputs['ragged_alpha'],
                inputs['start_nodes'], inputs['tau_max'])


# ==============================================================================
# SECTION 14 & 17: UNIFIED BTB-AC MASTER ARCHITECTURE ORCHESTRATOR
# ==============================================================================

@tf.keras.utils.register_keras_serializable(package="BTBAc")
class AssembledBTBAcMasterModel(tf.keras.Model, BTBSerializableMixin):
    """
    The main architecture model for the 'Building a Tower from Blocks' (BTB-ac) framework.
    Integrates 8 distinct encoder blocks guided by co-evolutionary self-supervision.
    """
    def __init__(self, config: BTBAcGlobalArchitectureConfig, **kwargs):
        super(AssembledBTBAcMasterModel, self).__init__(**kwargs)
        self.master_config = config
        self.tensor_router = ExactGraphTensorRouter(config, name="exact_tensor_router")

        self.subgraph_projector = SubgraphSpecificLatentProjector(config, name="m1_subgraph_proj")
        self.cross_graph_attn = CrossGraphContextualAttention(config, name="m1_cross_attn")
        self.intra_subgraph_attn = IntraSubgraphContextualAttention(config, name="m1_intra_attn")
        self.fusion_weighter = SemanticStructureFusionWeighter(config, name="m1_fusion_weighter")
        self.foundational_fusion = FoundationalNodeRepresentationFusion(config, name="m1_foundational_fusion")

        self.freq_scorer = InteractionFrequencyScorer(config, name="m2_freq_score")
        self.disp_scorer = InteractionDispersionScorer(config, name="m2_disp_score")
        self.div_scorer = InteractionTargetDiversityScorer(config, name="m2_div_score")
        self.quality_evaluator = ComprehensiveQualityEvaluator(config, name="m2_quality_eval")
        self.greedy_walker = AdaptiveEnergyGreedyWalker(config, name="m2_greedy_walker")

        self.path_semantic_enc = IntraPathSemanticEncoder(config, name="m2_path_sem_enc")
        self.path_info_ext = PathInformationExtractor(config, name="m2_path_info_ext")
        self.inter_path_enc = InterPathInteractionEncoder(config, name="m2_inter_path_enc")
        self.global_rep_agg = GlobalRepresentationAggregator(config, name="m2_global_agg")

        self.transfer_vec_gen = InstantaneousTransferVectorGenerator(config, name="m3_transfer_gen")
        self.intra_seq_enc = IntraSequenceMutualLearningEncoder(config, name="m3_intra_seq_enc")
        self.long_term_fusion = LongTermInterestFusionEncoder(config, name="m3_long_term_fusion")
        self.dynamic_gating = DynamicRepresentationGating(config, name="m3_dynamic_gating")

        self.loss_engine = CoEvolutionarySelfSupervisionEngine(config, name="m4_loss_engine")

        self.fidelity_tracker = TopologicalFidelityMetric()
        self.discriminability_tracker = PathDiscriminabilityMetric()
        self.curriculum_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="model_curriculum_step")

    @property
    def metrics(self):
        return [self.fidelity_tracker, self.discriminability_tracker]

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs: Dict[str, Union[tf.Tensor, tf.RaggedTensor]], training: bool = False) -> Dict[str, tf.Tensor]:

        # === PHASE 1: Cross-Graph Attention Convolution ===
        center_h, center_o, h_raw, s_types, o_types, tau_w, adj_mask, two_hop_mask, ego_adj, ego_rel = self.tensor_router.extract_local_topology(inputs)

        h_center_proj, h_neighbor_proj = self.subgraph_projector(center_h, center_o, h_raw, s_types, o_types)
        h_context = self.cross_graph_attn(h_neighbor_proj, s_types, o_types, tau_w, adj_mask, training=training)
        l_local, valid_r_mask = self.intra_subgraph_attn(h_center_proj, center_o, h_context, s_types, o_types, tau_w, adj_mask, training=training)
        w_fusion = self.fusion_weighter(l_local, h_context, center_o, o_types, adj_mask, two_hop_mask)
        f_foundational = self.foundational_fusion(l_local, w_fusion, self.cross_graph_attn.alpha_t, center_o, valid_r_mask)

        # === PHASE 2: Quality-Guided Adaptive Greedy Walk ===
        d_fre = self.freq_scorer(tau_w, adj_mask)
        entropy_type, p_i_r = self.disp_scorer(tau_w, s_types, adj_mask, self.cross_graph_attn.alpha_t)

        ragged_adj, ragged_tau, ragged_alpha, start_nodes, tau_max = self.tensor_router.extract_greedy_walk_components(inputs)

        d_tar = self.div_scorer(ego_adj, ego_rel, self.cross_graph_attn.alpha_t, tau_max, p_i_r, h_raw, adj_mask)
        d_quality_scores = self.quality_evaluator(d_fre, entropy_type, d_tar, center_o)

        paths_P, energies_E = self.greedy_walker(ragged_adj, ragged_tau, ragged_alpha, d_quality_scores, start_nodes)

        # Enforce boundary safety for out-of-batch random walk indices during mini-batch training
        safe_paths = tf.maximum(paths_P, 0)
        safe_paths = tf.clip_by_value(safe_paths, 0, tf.shape(f_foundational)[0] - 1)

        f_theta_sequence = tf.gather(f_foundational, safe_paths)
        path_padding_mask = tf.expand_dims(tf.cast(paths_P >= 0, dtype=f_theta_sequence.dtype), -1)
        f_theta_sequence = f_theta_sequence * path_padding_mask

        s_theta = self.path_semantic_enc(f_theta_sequence, energies_E, path_mask=(paths_P >= 0), training=training)
        X_theta = self.path_info_ext(f_foundational, s_theta, energies_E, path_mask=(paths_P >= 0), training=training)

        path_lengths = tf.reduce_sum(tf.cast(paths_P >= 0, tf.int32), axis=-1)
        Y_theta = self.inter_path_enc(X_theta, path_lengths, training=training)
        g_global = self.global_rep_agg(f_foundational, Y_theta, path_lengths, training=training)

        # === PHASE 3: Interest Transfer Vector Module ===
        hist_reps, seq_mask = self.tensor_router.extract_dynamic_intent_sequences(inputs)

        delta_vecs, delta_mask = self.transfer_vec_gen(hist_reps, seq_mask)
        tilde_delta = self.intra_seq_enc(delta_vecs, delta_mask, training=training)
        T_long_term = self.long_term_fusion(g_global, tilde_delta, delta_mask, training=training)

        H_enhanced = self.dynamic_gating(g_global, T_long_term)

        return {
            "f_foundational": f_foundational,
            "g_global": g_global,
            "H_enhanced": H_enhanced,
            "path_reps_P": X_theta,
            "path_mask": paths_P >= 0,
            "D_quality_scores": d_quality_scores
        }

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data):
        inputs, targets = data
        struct_prox_matrix = targets['structural_proximity']
        lcs_prox_matrix = targets['lcs_proximity']

        with tf.GradientTape() as tape:
            manifolds = self(inputs, training=True)

            total_loss = self.loss_engine.compute_total_coevolution_loss(
                current_step=self.curriculum_step,
                f_foundational=manifolds["f_foundational"],
                g_global=manifolds["g_global"],
                H_enhanced=manifolds["H_enhanced"],
                path_reps_P=manifolds["path_reps_P"],
                path_mask=manifolds["path_mask"],
                D_quality_scores=manifolds["D_quality_scores"],
                structural_prox_matrix=struct_prox_matrix,
                lcs_prox_matrix=lcs_prox_matrix
            )
            reg_loss = tf.reduce_sum(self.losses)
            final_objective = total_loss + reg_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(final_objective, trainable_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.master_config.evolution_dynamics.gradient_clip_norm)
        self.optimizer.apply_gradients(zip(clipped_gradients, trainable_vars))

        self.fidelity_tracker.update_state(struct_prox_matrix, manifolds["f_foundational"])
        self.discriminability_tracker.update_state(manifolds["g_global"], manifolds["path_reps_P"], manifolds["path_mask"])
        self.curriculum_step.assign_add(1)

        return {
            "loss": final_objective,
            "co_evolution_loss": total_loss,
            "fidelity": self.fidelity_tracker.result(),
            "discriminability": self.discriminability_tracker.result()
        }

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, data):
        inputs, targets = data
        struct_prox_matrix = targets['structural_proximity']
        lcs_prox_matrix = targets['lcs_proximity']

        manifolds = self(inputs, training=False)

        total_loss = self.loss_engine.compute_total_coevolution_loss(
            current_step=self.curriculum_step,
            f_foundational=manifolds["f_foundational"],
            g_global=manifolds["g_global"],
            H_enhanced=manifolds["H_enhanced"],
            path_reps_P=manifolds["path_reps_P"],
            path_mask=manifolds["path_mask"],
            D_quality_scores=manifolds["D_quality_scores"],
            structural_prox_matrix=struct_prox_matrix,
            lcs_prox_matrix=lcs_prox_matrix
        )

        self.fidelity_tracker.update_state(struct_prox_matrix, manifolds["f_foundational"])
        self.discriminability_tracker.update_state(manifolds["g_global"], manifolds["path_reps_P"], manifolds["path_mask"])

        return {
            "val_loss": total_loss,
            "val_fidelity": self.fidelity_tracker.result(),
            "val_discriminability": self.discriminability_tracker.result()
        }