import logging

from typing import List

from keras.losses import SparseCategoricalCrossentropy
from keras.utils.losses_utils import reduce_weighted_loss
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def _sparse_to_one_hot(y_true, y_pred):
    n_classes = y_pred.shape[-1]
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    if len(y_true.shape) == len(y_pred.shape) and y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)

    y_true = tf.one_hot(y_true, n_classes, dtype=y_pred.dtype)
    
    return y_true, y_pred, n_classes


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
    y_true, y_pred, _ = _sparse_to_one_hot(y_true, y_pred)
    # Preserve non-class axes (e.g., time). Reduce only over class axis.
    # Compute per-timestep Dice across classes.
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)  # sum over classes
    union = tf.reduce_sum(y_true + y_pred, axis=-1)         # sum over classes
    dice = (2.0 * intersection + smooth) / (union + smooth)  # shape: [...,]
    # Return per-example, per-timestep loss (no batch/time reduction)
    return 1.0 - dice


def sparse_categorical_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0, label_smoothing=0.0):
    """
    Computes the categorical focal crossentropy loss.
    
    Use this crossentropy loss function when there are two or more label
    classes and if you want to handle class imbalance without using
    class_weights. We expect labels to be provided in a one_hot representation.
    
    According to Lin et al., 2018 (https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a focal factor to down-weight easy examples and focus more on
    hard examples. The general formula for the focal loss (FL) is as follows:
    
    FL(p_t) = (1 - p_t) ** gamma * log(p_t)
    
    where p_t is defined as follows:
    p_t = output if y_true == 1, else 1 - output
    
    (1 - p_t) ** gamma is the modulating_factor, where gamma is a focusing
    parameter. When gamma = 0, there is no focal effect on the cross entropy.
    gamma reduces the importance given to simple examples in a smooth manner.
    
    The authors use alpha-balanced variant of focal loss (FL) in the paper:
    FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t)
    
    where alpha is the weight factor for the classes. If alpha = 1, the
    loss won't be able to handle class imbalance properly as all
    classes will have the same weight. This can be a constant or a list of
    constants. If alpha is a list, it must have the same length as the number
    of classes.
    
    Args:
        y_true: Sparse ground truth labels
        y_pred: Model predictions (logits or softmax probabilities)
        alpha: A weight balancing factor for all classes, default is 0.25 as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using compute_class_weight from sklearn.utils.
        gamma: A focusing parameter, default is 2.0 as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple (easy) examples in a smooth manner.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            0.1, use 0.1 / num_classes for non-target labels and
            0.9 + 0.1 / num_classes for target labels.
    
    Returns:
        Categorical focal crossentropy loss value
    """
    y_true, y_pred, n_classes = _sparse_to_one_hot(y_true, y_pred)
    eps = tf.keras.backend.epsilon()
    
    # Apply label smoothing if specified
    if label_smoothing:
        num_classes = tf.cast(n_classes, y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
    
    # Clip predictions to prevent numerical instability
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    
    # Calculate cross entropy
    cce = -y_true * tf.math.log(y_pred)
    
    # Calculate factors
    modulating_factor = tf.math.pow(1.0 - y_pred, gamma)
    weighting_factor = tf.multiply(modulating_factor, alpha)  
    
    # Apply weighting factor to cross entropy
    focal_cce = tf.multiply(weighting_factor, cce)
    focal_cce = tf.reduce_sum(focal_cce, axis=-1)
    
    return focal_cce


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


def weighted_sparse_kappa_loss(y_true, y_pred, weight_matrix):
    """
    Implements the Weighted Kappa loss function following TensorFlow Addons style.
    
    Weighted Kappa loss was introduced in the
    [Weighted kappa loss function for multi-class classification
    of ordinal data in deep learning]
    (https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666).
    
    Returns the unlogged ratio (numerator / denominator). The ratio typically
    ranges from [0, 2], where:
    - 0 indicates perfect agreement (all predictions correct)
    - 2 indicates random prediction
    - Values < 2 indicate better than random performance
    
    Args:
        y_true: Sparse ground truth labels
        y_pred: Model predictions (softmax probabilities)
        weight_matrix: (n_classes, n_classes) weight matrix.
                      Wij = penalty for confusing class i with class j.
                      Diagonal should typically be 0 (correct predictions).
    
    Returns:
        Unlogged Weighted Kappa ratio (typically in [0, 2], where 0=perfect, 2=random)
    """
    y_true, y_pred, n_classes = _sparse_to_one_hot(y_true, y_pred)
    

    
    epsilon = tf.keras.backend.epsilon()
    
    # Convert weight matrix to tensor
    weight_mat = tf.convert_to_tensor(weight_matrix, dtype=y_pred.dtype)
    weight_mat = tf.reshape(weight_mat, [n_classes, n_classes])
    
    # Flatten extra dimensions (e.g., time) to handle arbitrary input shapes
    batch_and_extra = tf.reduce_prod(tf.shape(y_true)[:-1])
    y_true_flat = tf.reshape(y_true, [batch_and_extra, n_classes])
    y_pred_flat = tf.reshape(y_pred, [batch_and_extra, n_classes])
    
    # Compute weight for each prediction position
    # For weight matrix: weight[i, j] = weight_mat[true_class[i], pred_class[j]]
    # Get true class indices (argmax of one-hot)
    true_class_indices = tf.cast(tf.argmax(y_true_flat, axis=-1), tf.int32)
    # Gather weight_mat rows: [batch_and_extra, n_classes]
    weight = tf.gather(weight_mat, true_class_indices)  # [batch_and_extra, n_classes]
    
    # Numerator: sum of weighted predictions
    numerator = tf.reduce_sum(weight * y_pred_flat)
    
    # Denominator: expected weighted agreement
    label_distribution = tf.reduce_sum(y_true_flat, axis=0, keepdims=True)  # [1, classes]
    pred_distribution = tf.reduce_sum(y_pred_flat, axis=0, keepdims=True)   # [1, classes]
    
    w_pred_distribution = tf.matmul(weight_mat, pred_distribution, transpose_b=True)  # [classes, 1]
    denominator = tf.reduce_sum(tf.matmul(label_distribution, w_pred_distribution))
    denominator /= tf.cast(batch_and_extra, dtype=denominator.dtype)

    return numerator / (denominator + epsilon)


class SparseDiceLoss(tf.keras.losses.Loss):
    """ tf reduction wrapper for sparse_dice_loss """
    def __init__(self,
                 smooth=1,
                 reduction=tf.keras.losses.Reduction.AUTO,
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


class WeightedSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):
    """
    Extension of SparseCategoricalCrossentropy to handle class weights.

    Args:
        class_weights: (Optional) Class weights to be applied to the loss.
        from_logits: (Optional) Whether y_pred is expected to be a logits tensor.
        ignore_class: (Optional) Class to ignore when computing the loss.
        reduction: (Optional) Reduction type for the loss.
        name: (Optional) Name for the loss.
    """

    def __init__(
        self,
        class_weights=None,
        from_logits=False,
        ignore_class=None,
        reduction=tf.keras.losses.Reduction.AUTO,
        name='weighted_sparse_categorical_crossentropy'
    ):
        super().__init__(
            from_logits=from_logits,
            ignore_class=ignore_class,
            reduction=tf.keras.losses.Reduction.NONE,
            name=name
        ) 
        self.class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32) if class_weights is not None else None

    def get_config(self):
        config = super().get_config()
        config.update({'class_weights': self.class_weights})
        return config

    def call(self, y_true, y_pred):
        loss = super().call(y_true, y_pred)
        if self.class_weights is not None:
            weight_mask = tf.squeeze(tf.gather(self.class_weights, tf.cast(y_true, dtype=tf.int32)), axis=-1)
            loss = tf.math.multiply(loss, weight_mask)
            
        return reduce_weighted_loss(loss, self.reduction)


class SparseCategoricalFocalCrossentropy(tf.keras.losses.Loss):
    """tf.keras.losses.Loss wrapper for sparse_categorical_focal_crossentropy
    
    Computes the categorical focal crossentropy loss.
    
    Args:
        alpha: A weight balancing factor for all classes, default is 0.25 as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using compute_class_weight from sklearn.utils.
        gamma: A focusing parameter, default is 2.0 as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple (easy) examples in a smooth manner.
        from_logits: Whether y_pred is expected to be a logits tensor. By
            default, we consider that y_pred encodes a probability distribution.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            0.1, use 0.1 / num_classes for non-target labels and
            0.9 + 0.1 / num_classes for target labels.
        reduction: Type of reduction to apply to the loss. Defaults to AUTO.
        name: Optional name for the loss instance.
    """
    
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        label_smoothing=0.0,
        reduction=tf.keras.losses.Reduction.AUTO,
        name='sparse_categorical_focal_crossentropy'
    ):
        super().__init__(name=name, reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'label_smoothing': self.label_smoothing,
        })
        return config

    def call(self, y_true, y_pred):
        return sparse_categorical_focal_crossentropy(
            y_true, y_pred,
            alpha=self.alpha,
            gamma=self.gamma,
            label_smoothing=self.label_smoothing
        )


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


class WeightedSparseKappaLoss(tf.keras.losses.Loss):
    """tf.keras.losses.Loss wrapper for weighted_sparse_kappa_loss"""
    def __init__(
        self,
        num_classes,
        weightage='quadratic',
        custom_weight_matrix=None,
        use_log=True,
        reduction=tf.keras.losses.Reduction.AUTO,
        name='weighted_sparse_kappa_loss'
    ):
        """
        Args:
            num_classes: Number of unique classes in your dataset.
            weightage: (Optional) Weighting to be considered for calculating
              kappa statistics. A valid value is one of
              ['linear', 'quadratic']. Defaults to 'quadratic' since it's
              mostly used. Ignored if custom_weight_matrix is provided.
            custom_weight_matrix: (Optional) Custom (num_classes, num_classes) weight matrix.
                                 If provided, overrides ordinal distance computation.
                                 Wij = penalty for confusing class i with class j.
                                 Diagonal should typically be 0 (correct predictions).
            use_log: (bool) If True, applies log to the kappa ratio, returning
                    values in [-inf, log 2] range. If False, returns the unlogged
                    ratio in [0, 2] range (0=perfect, 2=random). Default is True for backward compatibility.
            reduction: Reduction type for the loss
            name: (Optional) String name of the loss instance.
        
        Raises:
            ValueError: If the value passed for `weightage` is invalid
              i.e. not any one of ['linear', 'quadratic']
        """
        super().__init__(name=name, reduction=reduction)
        
        if weightage not in ("linear", "quadratic"):
            raise ValueError("Unknown kappa weighting type. Must be 'linear' or 'quadratic'.")
        
        self.num_classes = num_classes
        self.weightage = weightage
        self.custom_weight_matrix = custom_weight_matrix
        self.use_log = use_log
        
        # Create weight matrix based on weightage and custom_weight_matrix
        if custom_weight_matrix is not None:
            # Use custom weight matrix
            self.weight_matrix = custom_weight_matrix
        else:
            # Build ordinal weight matrix based on weightage
            label_vec = np.arange(self.num_classes, dtype=np.float32)
            col_label_vec = label_vec.reshape(self.num_classes, 1)
            row_label_vec = label_vec.reshape(1, self.num_classes)
            col_mat = np.tile(col_label_vec, (1, self.num_classes))
            row_mat = np.tile(row_label_vec, (self.num_classes, 1))
            
            # Compute ordinal distance
            ordinal_distance = np.abs(col_mat - row_mat)
            max_distance = max(num_classes - 1, 1.0)
            
            if weightage == "linear":
                # Normalize: |i - j| / (n_classes - 1)
                self.weight_matrix = ordinal_distance / max_distance
            else:  # quadratic
                # Normalize: |i - j|^2 / (n_classes - 1)^2
                self.weight_matrix = (ordinal_distance ** 2) / (max_distance ** 2)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'weightage': self.weightage,
            'custom_weight_matrix': self.custom_weight_matrix,
            'use_log': self.use_log,
        })
        return config
    
    def call(self, y_true, y_pred):
        loss = weighted_sparse_kappa_loss(
            y_true, 
            y_pred, 
            weight_matrix=self.weight_matrix,
        )
        if self.use_log:
            epsilon = tf.keras.backend.epsilon()
            loss = tf.math.log(loss + epsilon)
        return loss


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
        