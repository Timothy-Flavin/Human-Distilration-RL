python
import torch
import torch.nn.functional as F

def temperature_scaled_bc_loss(advantages, expert_actions, epsilon, max_bisection_iters=10):
    """
    Computes a scale-invariant Behavior Cloning loss for Dueling Architectures.
    
    This function forces the advantages to dynamically match the entropy of an 
    epsilon-greedy target policy using a detached, per-sample temperature (alpha). 
    It explicitly cancels the gradient explosion caused by temperature scaling.
    
    Args:
        advantages: Tensor of shape (batch_size, num_actions). The physical advantages A(s,a).
        expert_actions: Tensor of shape (batch_size,), containing the offline dataset actions.
        epsilon: Float, the current exploration rate dictating the target entropy.
        max_bisection_iters: Int, number of iterations for the root-finding algorithm.
        
    Returns:
        loss: Scalar tensor representing the gradient-corrected KL divergence.
    """
    batch_size, num_actions = advantages.shape
    device = advantages.device

    # =========================================================================
    # STEP 1: Define the "Magnet Policy" and Target Entropy
    # =========================================================================
    # The target is not a strict Dirac delta (one-hot), but an epsilon-smoothed
    # distribution. This provides a stable basin of attraction.
    p_max = 1.0 - epsilon + (epsilon / num_actions)
    p_other = epsilon / num_actions
    
    # Calculate the exact target entropy analytically.
    # Entropy H = - sum(p * log(p))
    target_entropy = - (p_max * torch.log(torch.tensor(p_max))) - \
                     ((num_actions - 1) * p_other * torch.log(torch.tensor(p_other)))
    target_entropy = target_entropy.to(device)

    # Construct the target distribution tensor for the KL divergence later.
    target_probs = torch.full_like(advantages, p_other)
    target_probs.scatter_(1, expert_actions.unsqueeze(1), p_max)

    # =========================================================================
    # STEP 2: Vectorized Bisection Search for Per-Sample Alpha
    # =========================================================================
    # We must find a unique alpha for every state such that the entropy of
    # softmax(A(s,a) / alpha) exactly matches the target_entropy.
    # Because entropy is monotonically increasing with alpha, bisection is guaranteed
    # to find the unique root rapidly and safely.
    
    alpha_low = torch.full((batch_size, 1), 1e-4, device=device)
    alpha_high = torch.full((batch_size, 1), 1e3, device=device)
    
    # We don't want gradients flowing through the root-finding process.
    with torch.no_grad():
        for _ in range(max_bisection_iters):
            alpha_mid = (alpha_low + alpha_high) / 2.0
            
            # Compute current logit probabilities
            probs = F.softmax(advantages / alpha_mid, dim=-1)
            
            # Compute current entropy: -sum(p * log(p))
            # Added 1e-8 for numerical stability against log(0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
            
            # If entropy is too high, the temperature (alpha) is too high.
            # If entropy is too low, the temperature (alpha) is too low.
            alpha_high = torch.where(entropy > target_entropy, alpha_mid, alpha_high)
            alpha_low = torch.where(entropy <= target_entropy, alpha_mid, alpha_low)

        # The final alpha is the midpoint of our narrow bounds.
        alpha_final = (alpha_low + alpha_high) / 2.0

    # STEP 3: Detach Alpha
    # This ensures PyTorch treats our computed temperatures as static scalars,
    # preventing the two-timescale optimization instability.
    alpha_detached = alpha_final.detach()

    # =========================================================================
    # STEP 4: Temperature Scaling & Loss Computation
    # =========================================================================
    # We scale the advantages by our found alpha. 
    # Because alpha is small when advantages are small, this shapes the logits 
    # to perfectly match the target epsilon distribution.
    scaled_logits = advantages / alpha_detached
    
    # Calculate log probabilities of our scaled advantages
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    
    # Calculate KL Divergence analytically: sum(target_probs * (log(target_probs) - log_probs))
    # We use manual calculation instead of F.kl_div for total transparency.
    kl_divergence = torch.sum(target_probs * (torch.log(target_probs + 1e-8) - log_probs), dim=-1, keepdim=True)

    # =========================================================================
    # STEP 5: The Gradient Cancellation Scale (The Core Fix)
    # =========================================================================
    # MATHEMATICAL EXPLANATION:
    # If we just returned `kl_divergence`, the backward pass would explode. 
    # The chain rule of KL divergence w.r.t the physical advantages A(s,a) is:
    # d(KL) / dA = d(KL)/d(logits) * d(logits)/dA
    #
    # We know d(logits)/dA = 1 / alpha. 
    # If alpha is 0.001 (because physical advantages are very small), 
    # the gradient flowing backward into the network gets multiplied by 1000.
    # This destroys the value network.
    #
    # By multiplying the final loss by alpha_detached, we inject an external `alpha` 
    # into the chain rule. 
    # 
    # d(Loss) / dA = alpha * [ (softmax_probs - target_probs) * (1 / alpha) ]
    # The `alpha` and `1/alpha` perfectly cancel out.
    # d(Loss) / dA = (softmax_probs - target_probs)
    #
    # The maximum gradient magnitude is now strictly bounded in [-1, 1], completely 
    # independent of the environment's scale or how small alpha gets.
    
    protected_loss = kl_divergence * alpha_detached

    # Return the mean loss across the batch
    return protected_loss.mean()

# =========================================================================
# Example Usage:
# =========================================================================
if __name__ == "__main__":
    batch_size = 32
    num_actions = 5
    
    # Simulate small environment advantages (e.g., natural scale of 0.05)
    mock_advantages = torch.randn(batch_size, num_actions, requires_grad=True) * 0.05
    
    # Simulate offline expert actions
    mock_expert_actions = torch.randint(0, num_actions, (batch_size,))
    
    # Decaying epsilon (e.g., late in training)
    current_epsilon = 0.1
    
    # Calculate loss
    loss = temperature_scaled_bc_loss(mock_advantages, mock_expert_actions, current_epsilon)
    
    # Backward pass
    loss.backward()
    
    # Verify the gradients are safely bounded despite the small advantages
    # Gradient magnitude should strictly be <= 1.0
    print(f"Max gradient magnitude: {mock_advantages.grad.abs().max().item():.4f}")