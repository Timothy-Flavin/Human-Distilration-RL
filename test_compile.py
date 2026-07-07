import torch
import torch.nn.functional as F

@torch.compile
def _find_alpha_bisection(advantages, target_entropy, max_bisection_iters: int):
    batch_size = advantages.shape[0]
    device = advantages.device
    
    alpha_low = torch.full((batch_size, 1), 1e-4, device=device)
    alpha_high = torch.full((batch_size, 1), 1e3, device=device)
    
    for _ in range(max_bisection_iters):
        alpha_mid = (alpha_low + alpha_high) / 2.0
        
        logits = advantages / alpha_mid
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        log_probs = logits - lse
        probs = torch.exp(log_probs)
        
        entropy = lse - torch.sum(probs * logits, dim=-1, keepdim=True)
        
        alpha_high = torch.where(entropy > target_entropy, alpha_mid, alpha_high)
        alpha_low = torch.where(entropy <= target_entropy, alpha_mid, alpha_low)
        
    return (alpha_low + alpha_high) / 2.0

def old_bisection(advantages, target_entropy, max_bisection_iters=10):
    batch_size = advantages.shape[0]
    device = advantages.device
    alpha_low = torch.full((batch_size, 1), 1e-4, device=device)
    alpha_high = torch.full((batch_size, 1), 1e3, device=device)
    
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

    return (alpha_low + alpha_high) / 2.0

adv = torch.randn(32, 5) * 0.05
target_entropy = torch.ones(32, 1) * 0.5

old_alpha = old_bisection(adv, target_entropy)
new_alpha = _find_alpha_bisection(adv, target_entropy, 10)

print("Difference:", torch.abs(old_alpha - new_alpha).max().item())

