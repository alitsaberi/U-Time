import logging

from typing import List

import tensorflow as tf

logger = logging.getLogger(__name__)

def _get_shapes_and_one_hot(y_true, y_pred):
    shape = y_pred.get_shape()
    n_classes = shape[-1]
    # Squeeze dim -1 if it is == 1, otherwise leave it
    dims = tf.cond(tf.equal(y_true.shape[-1] or -1, 1), lambda: tf.shape(y_true)[:-1], lambda: tf.shape(y_true))
    y_true = tf.reshape(y_true, dims)
    y_true = tf.one_hot(tf.cast(y_true, tf.uint8), depth=n_classes)
    return y_true, shape, n_classes


def sparse_dice_loss(y_true, y_pred, smooth=1):
    """
    Approximates the class-wise dice coefficient computed per-batch element
    across spatial image dimensions. Returns the 1 - mean(per_class_dice) for
    each batch element.
    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    intersection = tf.reduce_sum(y_true * y_pred, axis=reduction_dims)
    union = tf.reduce_sum(y_true + y_pred, axis=reduction_dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice, axis=-1, keepdims=True)


def weighted_categorical_crossentropy(y_true, y_pred, class_weights=None):
    """
    Implements weighted categorical crossentropy loss.
    Particularly useful for imbalanced classification tasks.

    Args:
        y_true: Ground truth labels, should be one-hot encoded or sparse
        y_pred: Model predictions (softmax probabilities)
        class_weights: Optional dictionary or list of class weights.
                      If None, all classes are weighted equally.

    Returns:
        Weighted categorical crossentropy loss value
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    
    # Handle class weights
    if class_weights is None:
        class_weights = tf.ones([n_classes])
    elif isinstance(class_weights, dict):
        class_weights = tf.convert_to_tensor([class_weights.get(i, 1.0) for i in range(n_classes)])
    else:
        class_weights = tf.convert_to_tensor(class_weights)
    
    # Ensure weights are float32
    class_weights = tf.cast(class_weights, tf.float32)
    
    # Compute weighted cross entropy
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred)
    weighted_cross_entropy = cross_entropy * class_weights
    
    return tf.reduce_mean(tf.reduce_sum(weighted_cross_entropy, axis=-1))


def temporal_consistency_loss(y_pred, weight_matrix=None):
    """
    Computes temporal consistency loss that penalizes sudden changes in predictions
    between consecutive time steps.
    
    Args:
        y_pred: Model predictions (softmax probabilities) of shape [-1, classes]
               where the first dimension combines batch and time dimensions
        weight_matrix: Optional (n_classes, n_classes) transition penalty matrix.
                      If None, uses identity matrix (penalize all transitions equally)
    
    Returns:
        Temporal consistency loss value
    """
    # Get consecutive time steps
    p_t = y_pred[:, :-1, :]     # predictions at time t
    p_next = y_pred[:, 1:, :]   # predictions at time t+1
    
    # Compute pairwise penalties for each transition
    if weight_matrix is not None:
        W = tf.constant(weight_matrix, dtype=tf.float32)
    else:
        # Create matrix with 1s everywhere and 0s on diagonal
        n_classes = y_pred.shape[-1]
        W = tf.ones([n_classes, n_classes], dtype=tf.float32) - tf.eye(n_classes, dtype=tf.float32)
    
    # Expand dims for broadcasting
    p_t_exp = tf.expand_dims(p_t, -1)       # [batch, time-1, classes, 1]
    p_next_exp = tf.expand_dims(p_next, -2) # [batch, time-1, 1, classes]

    # Compute weighted transition probabilities
    transition_prob = p_t_exp * p_next_exp  # [batch, time-1, classes, classes]
    weighted_penalty = transition_prob * W   # apply penalty matrix

    # Average over all dimensions
    loss = tf.reduce_mean(weighted_penalty)
    return loss


class SparseDiceLoss(tf.keras.losses.Loss):
    """ tf reduction wrapper for sparse_dice_loss """
    def __init__(self,
                 reduction,
                 smooth=1,
                 name='sparse_dice_loss',
                 **kwargs):
        self.smooth = smooth
        super(SparseDiceLoss, self).__init__(
            name=name,
            reduction=reduction
        )

    def get_config(self):
        config = super().get_config()
        config.update({'smooth': self.smooth})
        return config

    def call(self, y_true, y_pred):
        return sparse_dice_loss(y_true, y_pred, smooth=self.smooth)


class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    """tf.keras.losses.Loss wrapper for weighted_categorical_crossentropy"""
    
    def __init__(self,
                 class_weights=None,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='weighted_categorical_crossentropy'):
        super().__init__(name=name, reduction=reduction)
        self.class_weights = class_weights

    def get_config(self):
        config = super().get_config()
        config.update({'class_weights': self.class_weights})
        return config

    def call(self, y_true, y_pred):
        return weighted_categorical_crossentropy(y_true, y_pred, self.class_weights)


class TemporalConsistencyLoss(tf.keras.losses.Loss):
    """tf.keras.losses.Loss wrapper for temporal_consistency_loss"""
    
    def __init__(self,
                 weight_matrix=None,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='temporal_consistency_loss'):
        super().__init__(name=name, reduction=reduction)
        self.weight_matrix = weight_matrix

    def get_config(self):
        config = super().get_config()
        config.update({
            'weight_matrix': self.weight_matrix
        })
        return config
    
    def call(self, y_true, y_pred):
        # Note: y_true is not used in temporal consistency loss
        return temporal_consistency_loss(y_pred, self.weight_matrix)

class CombinedLoss(tf.keras.losses.Loss):
    """Combines multiple loss functions with optional weights."""
    def __init__(self,
                 losses: List[tf.keras.losses.Loss],
                 weights: List[float] = None,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='combined_loss'):
        super().__init__(name=name, reduction=reduction)
        self.losses = losses
        self.weights = weights

    def get_config(self):
        config = super().get_config()
        config.update({'losses': self.losses, 'weights': self.weights})
        return config

    def call(self, y_true, y_pred):
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(y_true, y_pred)
        return total_loss
        