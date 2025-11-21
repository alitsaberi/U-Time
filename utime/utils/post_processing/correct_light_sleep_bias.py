import numpy as np

def correct_light_sleep_bias(predictions, tau_d=1.45, tau_r=1.50, tau_w=1.25, 
                             light_idx=1, deep_idx=2, rem_idx=3, wake_idx=0):
    """
    Corrects light sleep bias in probabilistic predictions using the method from:
    "SalientSleepNet: Multimodal salient wave detection network for sleep staging"
    
    The method applies a confirmation step when Light sleep (L) has the highest probability.
    It compares the ratio q(L)/q(i2) to stage-specific thresholds.
    
    Args:
        predictions: Array of shape (n_periods, n_classes) with class probabilities
        tau_d: Threshold for Deep sleep ratio (default: 1.45 from paper)
        tau_r: Threshold for REM sleep ratio (default: 1.50 from paper)
        tau_w: Threshold for Wake ratio (default: 1.25 from paper)
        light_idx: Index of Light sleep class (N2)
        deep_idx: Index of Deep sleep class (N3)
        rem_idx: Index of REM sleep class
        wake_idx: Index of Wake class
    
    Returns:
        corrected_predictions: Array of shape (n_periods,) with corrected class labels
    """
    predictions = np.asarray(predictions)
    n_epochs = predictions.shape[0]
    corrected_predictions = np.zeros(n_epochs, dtype=np.int32)
    
    # Map stage indices to thresholds
    threshold_map = {
        deep_idx: tau_d,
        rem_idx: tau_r,
        wake_idx: tau_w
    }
    
    for i in range(n_epochs):
        probs = predictions[i]
        
        # Sort indices by probability in descending order
        sorted_indices = np.argsort(probs)[::-1]
        i1 = sorted_indices[0]  # Highest probability stage
        i2 = sorted_indices[1]  # Second highest probability stage
        
        # If highest probability is NOT light sleep, assign it directly
        if i1 != light_idx:
            corrected_predictions[i] = i1
        else:
            # Highest probability is Light sleep - apply confirmation step
            q_light = probs[light_idx]
            q_i2 = probs[i2]
            
            # Calculate ratio q(L) / q(i2)
            if q_i2 > 0:
                ratio = q_light / q_i2
            else:
                # If second highest is 0, light sleep is definitely the answer
                ratio = float('inf')
            
            # Get threshold for the second-highest stage
            tau = threshold_map.get(i2, 1.0)  # Default to 1.0 if stage not in map
            
            # Apply threshold comparison
            if ratio > tau:
                # Ratio exceeds threshold, assign light sleep
                corrected_predictions[i] = light_idx
            else:
                # Ratio doesn't exceed threshold, assign second highest
                corrected_predictions[i] = i2
    
    return corrected_predictions
