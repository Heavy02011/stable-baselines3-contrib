# GRPO Algorithm Implementation Review

**Date**: January 2026  
**Reviewer**: Repository Maintainer  
**Implementation Location**: `sb3_contrib/grpo/`  
**Original Paper**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

---

## Executive Summary

This document reviews the Group Relative Policy Optimization (GRPO) implementation in stable-baselines3-contrib against the original algorithm described in the DeepSeek-Math paper (arXiv:2402.03300). The review identifies several **critical deviations** from the original paper, evaluates implementation quality, and provides recommendations.

### Key Findings

✅ **Correctly Implemented:**
- Group-based advantage normalization within batches
- KL divergence regularization  
- PPO-style clipped surrogate objective
- Optional value function support (hybrid mode)
- Support for discrete and continuous action spaces

⚠️ **Critical Deviations from Original Paper:**
- Uses GAE (Generalized Advantage Estimation) instead of direct reward-based advantages
- Applies group normalization to GAE advantages, not raw rewards
- Designed for standard RL environments, not LLM fine-tuning (the original context)
- Different sampling strategy (rollout-based vs. group sampling per prompt)

**Overall Assessment**: The implementation is a **valid adaptation** of GRPO principles to standard RL environments, but it **deviates significantly** from the original DeepSeek paper's algorithm. It should be considered a "GRPO-inspired" algorithm rather than a direct implementation of the paper.

---

## 1. Original GRPO Algorithm (DeepSeek-Math Paper)

### 1.1 Context and Purpose

GRPO was introduced in the DeepSeek-Math paper for **fine-tuning large language models (LLMs)** using reinforcement learning from human feedback (RLHF). The key innovation was to eliminate the critic network (value function) used in traditional PPO, making training more memory-efficient.

### 1.2 Core Algorithm Components

#### Group Sampling
For each input prompt `q`, the algorithm samples **G outputs** (typically G=4-8) from the current policy:
```
outputs = [sample(policy, q) for _ in range(G)]
```

#### Direct Reward-Based Advantages
For each output in the group, compute advantage using **direct reward normalization**:

```
rewards = [reward_fn(q, o) for o in outputs]
mean_reward = mean(rewards)
std_reward = std(rewards) + ε

A^(i) = (r^(i) - mean_reward) / std_reward
```

**Key Point**: Advantages are computed from **raw rewards**, not from a value function or GAE.

#### Loss Function
```
L_GRPO(θ) = -1/G Σ_i Σ_t [
    π_θ(o_t^(i)|q,o_{<t}^(i)) / π_θ_old(o_t^(i)|q,o_{<t}^(i)) · A_t^(i)
    - β · D_KL(π_θ(·|q) || π_θ_old(·|q))
]
```

With optional clipping:
```
ratio = π_θ / π_θ_old
clipped_ratio = clip(ratio, 1-ε, 1+ε)
L = -min(ratio · A, clipped_ratio · A) - β · KL_penalty
```

### 1.3 Key Characteristics
- **No value function** (critic-free)
- **No GAE** (no temporal bootstrapping)
- **Group-wise sampling**: Multiple outputs per prompt
- **Direct reward normalization**: Advantages from reward statistics, not value estimates
- **Memory efficient**: ~40-60% less memory than PPO

---

## 2. Current Implementation Analysis

### 2.1 Implementation Overview

The current implementation in `sb3_contrib/grpo/grpo.py`:
- Inherits from `OnPolicyAlgorithm` (standard RL base class)
- Uses `RolloutBuffer` for experience collection
- Implements group-based advantage normalization
- Supports optional value function (hybrid mode)

### 2.2 Advantage Computation

#### How It Currently Works

```python
# In OnPolicyAlgorithm.collect_rollouts() (base class):
rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
```

The `RolloutBuffer.compute_returns_and_advantage()` method uses **GAE**:
```python
# Generalized Advantage Estimation (GAE)
for step in reversed(range(buffer_size)):
    delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
    advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
```

Then in `GRPO.train()`:
```python
# Group-based normalization applied to GAE advantages
advantages = rollout_data.advantages
if self.normalize_advantage and len(advantages) > 1:
    advantages = self._compute_group_advantages(advantages)
```

#### The Deviation

**Original GRPO**: 
```
A^(i) = (reward^(i) - mean_group(rewards)) / std_group(rewards)
```

**Current Implementation**: 
```
GAE^(i) = Σ_t [δ_t + γλδ_{t+1} + ...] where δ_t = r_t + γV(s_{t+1}) - V(s_t)
A^(i) = (GAE^(i) - mean_group(GAE)) / std_group(GAE)
```

### 2.3 Detailed Code Review

#### ✅ Correctly Implemented Features

1. **Group-Based Normalization** (`_compute_group_advantages()`)
```python
def _compute_group_advantages(self, advantages: th.Tensor) -> th.Tensor:
    batch_size = advantages.shape[0]
    
    if batch_size <= self.group_size:
        mean = advantages.mean()
        std = advantages.std() + 1e-8
        return (advantages - mean) / std
    
    n_groups = batch_size // self.group_size
    grouped = advantages.view(n_groups, self.group_size)
    group_means = grouped.mean(dim=1, keepdim=True)
    group_stds = grouped.std(dim=1, keepdim=True) + 1e-8
    return ((grouped - group_means) / group_stds).view(-1)
```
✅ This correctly implements group-wise normalization with proper handling of edge cases.

2. **KL Divergence Penalty**
```python
log_ratio = log_prob - rollout_data.old_log_prob
kl_loss = self.kl_coef * th.mean((th.exp(log_ratio) - 1) - log_ratio)
```
✅ Correctly implements KL penalty using the approximation: `KL ≈ (e^(log_ratio) - 1) - log_ratio`

3. **PPO-Style Clipping**
```python
ratio = th.exp(log_prob - rollout_data.old_log_prob)
policy_loss_1 = advantages * ratio
policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
```
✅ Standard clipped surrogate objective, correctly implemented.

4. **Optional Value Function**
```python
value_loss = F.mse_loss(rollout_data.returns, values_pred)
loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + kl_loss
```
✅ Allows `vf_coef=0` for pure GRPO or `vf_coef>0` for hybrid approach.

#### ⚠️ Deviations and Concerns

1. **Use of GAE Instead of Direct Rewards**
   - **Impact**: HIGH - This is the fundamental difference from the paper
   - **Why it matters**: 
     - GAE introduces temporal dependencies and value function estimates
     - Original GRPO's elegance comes from eliminating value function dependency
     - Different mathematical formulation leads to different gradients
   - **Trade-off**: GAE may provide better variance reduction for standard RL tasks

2. **Rollout-Based vs. Group-Sampling Strategy**
   - **Original**: Sample G outputs per prompt, compute rewards, normalize within group
   - **Current**: Collect rollout buffer, then batch and normalize
   - **Impact**: MEDIUM - Changes when and how groups are formed
   - **Note**: Current approach is more suitable for standard RL environments

3. **Action Space Adaptation**
   - **Original**: Designed for autoregressive token generation (LLM outputs)
   - **Current**: Supports Box, Discrete, MultiDiscrete, MultiBinary
   - **Impact**: LOW - This is an appropriate adaptation, not a bug

### 2.4 Parameter Comparison

| Parameter | Paper Default | Implementation Default | Notes |
|-----------|---------------|------------------------|-------|
| `group_size` | 4-8 | 4 | ✅ Matches paper |
| `kl_coef` | ~0.01-0.1 | 0.1 | ✅ Reasonable |
| `clip_range` | ~0.2 | 0.2 | ✅ PPO standard |
| `gae_lambda` | N/A (not used) | 0.95 | ⚠️ Deviation |
| `vf_coef` | 0 (no critic) | 0.0 | ✅ Default matches paper |
| Learning rate | Varied | 3e-4 | ✅ Standard |

---

## 3. Mathematical Correctness

### 3.1 Advantage Computation

**Original GRPO Formula:**
```
A^(i) = (r^(i) - μ_r) / (σ_r + ε)
where μ_r = mean(rewards in group)
      σ_r = std(rewards in group)
```

**Implementation's GAE Formula:**
```
δ_t = r_t + γV(s_{t+1})(1 - done) - V(s_t)
GAE_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
A^(i) = (GAE^(i) - μ_GAE) / (σ_GAE + ε)
```

**Verdict**: ❌ The implementation uses a **different advantage estimation method** than the paper. While mathematically correct for GAE, it's not the GRPO algorithm as described.

### 3.2 KL Divergence

Both use the same approximation:
```
KL(π_old || π_new) ≈ E[(e^(log π_new - log π_old) - 1) - (log π_new - log π_old)]
```

**Verdict**: ✅ Mathematically correct and matches the paper.

### 3.3 Policy Update

Both use clipped surrogate objective:
```
L = -E[min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)]
```

**Verdict**: ✅ Correct implementation.

---

## 4. Adaptation for Standard RL Environments

### 4.1 Why Deviations May Be Justified

The original GRPO paper focused on **LLM fine-tuning**, where:
- Each "episode" is generating one text completion
- Rewards are terminal (only at end of generation)
- Multiple completions per prompt are natural
- No inherent state-transition dynamics

Standard RL environments (CartPole, Pendulum, etc.) have:
- Continuous state transitions
- Step-by-step rewards
- Clear MDP structure
- Different sampling patterns

**The current implementation adapts GRPO principles to this different domain.**

### 4.2 Design Choices

1. **Using GAE**: Provides better credit assignment for multi-step rewards
2. **Rollout buffer**: Standard approach for on-policy RL
3. **Value function option**: Helpful for environments with complex dynamics
4. **Group normalization of GAE**: Retains the "relative comparison" spirit

### 4.3 Is This Still "GRPO"?

**Philosophical Answer**: The implementation captures the **spirit** of GRPO (group-relative comparisons, KL regularization) but uses a different mathematical foundation (GAE vs. direct rewards).

**Naming Recommendation**: Consider one of:
- "GRPO-Adapted" 
- "Group-Normalized PPO"
- "GRPO for Standard RL"
- Keep "GRPO" but clearly document the differences

---

## 5. Testing and Validation

### 5.1 Current Test Coverage

From `tests/test_grpo.py`:
- ✅ Basic training (CartPole, Pendulum)
- ✅ Different group sizes (2, 4, 8)
- ✅ Different KL coefficients (0.0, 0.1, 0.5)
- ✅ Multi-environment support
- ✅ Discrete action spaces
- ✅ Advantage normalization toggle
- ✅ Hybrid mode (with value function)
- ✅ Pretrained model evaluation

**Coverage**: Good for functionality, but **no comparison to paper results** since the paper focused on LLM tasks.

### 5.2 Missing Validations

1. **Performance benchmarks**: No baseline showing GRPO matches or exceeds PPO
2. **Ablation studies**: Effect of group_size, kl_coef not systematically studied
3. **Paper replication**: Not applicable (different domain)
4. **Comparative analysis**: GRPO vs PPO vs A2C on standard benchmarks

---

## 6. Documentation Review

### 6.1 Code Documentation

✅ **Strengths**:
- Comprehensive docstring in `grpo.py`
- Clear parameter descriptions
- Method-level documentation
- Inline comments for key operations

⚠️ **Gaps**:
- Doesn't explicitly state deviation from paper
- Could clarify this is adapted for standard RL, not LLMs
- Missing references to original paper context

### 6.2 External Documentation

❌ **Missing**:
- No `docs/modules/grpo.rst` (documentation page)
- Not listed in algorithm documentation
- No "How GRPO differs from PPO" guide
- No example usage in docs

✅ **Present**:
- Example scripts in `examples/`
- Hyperparameter tuning results
- Performance comparison (SAC vs GRPO)

---

## 7. Recommendations

### 7.1 Critical (Must Address)

1. **Documentation Clarification** 🔴
   - Clearly state this is an **adaptation** of GRPO for standard RL
   - Explain how it differs from the DeepSeek-Math paper
   - Document when to use GRPO vs PPO

   **Suggested addition to docstring**:
   ```python
   """
   Group Relative Policy Optimization (GRPO)
   
   This implementation adapts GRPO from the DeepSeek-Math paper for standard
   reinforcement learning environments. Key differences from the original paper:
   
   1. Uses GAE for advantage estimation instead of direct reward normalization
   2. Designed for continuous state-action RL, not LLM fine-tuning
   3. Applies group normalization to GAE advantages, not raw rewards
   
   The algorithm retains GRPO's core principles of group-relative comparisons
   and KL regularization while adapting to the standard RL setting.
   """
   ```

2. **Create docs/modules/grpo.rst** 🔴
   - Algorithm description
   - Differences from paper
   - When to use it
   - Example usage
   - Hyperparameter guide

3. **Add Paper vs Implementation Table** 🔴
   - Side-by-side comparison
   - Clearly mark deviations
   - Explain rationale

### 7.2 Important (Should Address)

4. **Benchmark Performance** 🟡
   - Compare GRPO vs PPO on standard benchmarks
   - Measure actual performance impact of group normalization
   - Document when GRPO outperforms/underperforms PPO

5. **Ablation Studies** 🟡
   - Effect of group_size (systematic sweep)
   - Impact of kl_coef
   - GAE vs. no GAE (if feasible to implement)

6. **Consider a "Pure GRPO" Implementation** 🟡
   - Implement true paper algorithm for comparison
   - May require custom rollout buffer
   - Useful for understanding trade-offs

### 7.3 Nice to Have (Optional)

7. **Add Architectural Diagram** 🟢
   - Visual comparison: Paper GRPO vs Implementation
   - Helps users understand differences

8. **Performance Profiling** 🟢
   - Memory usage comparison vs PPO
   - Training time comparison
   - Validate efficiency claims

9. **Extended Examples** 🟢
   - More diverse environments
   - Hyperparameter tuning guide
   - Common pitfalls and solutions

---

## 8. Comparison with Other Implementations

### 8.1 LLM-Focused GRPO Implementations

Several implementations exist for the **original** LLM use case:
- HuggingFace TRL library
- Unsloth
- GRPO-Zero (from scratch implementation)
- DeepSeek's reference implementation

**Key Difference**: These use direct reward normalization without GAE, matching the paper exactly.

### 8.2 Positioning This Implementation

This implementation is:
- ✅ **Unique**: First GRPO adaptation for standard RL environments
- ✅ **Practical**: Works with gym/gymnasium environments
- ⚠️ **Different**: Not a direct paper implementation
- ✅ **Complementary**: Fills a gap not addressed by LLM implementations

**Value Proposition**: Brings GRPO's group-relative optimization to the standard RL toolkit.

---

## 9. Code Quality Assessment

### 9.1 Code Style and Standards

✅ **Passes All Checks**:
- `make format`: Clean
- `make check-codestyle`: Pass
- `make lint`: Pass
- Follows stable-baselines3 conventions

### 9.2 Architecture and Design

✅ **Strengths**:
- Clean inheritance from `OnPolicyAlgorithm`
- Minimal code duplication
- Proper use of existing infrastructure
- Good separation of concerns

⚠️ **Potential Improvements**:
- Could extract advantage normalization to a separate module
- May want custom buffer class for true GRPO
- Consider factory pattern for different normalization strategies

### 9.3 Error Handling

✅ **Good**:
- Validates `n_steps * n_envs > 1`
- Warns about batch size mismatch
- Handles edge cases in group normalization (remainders, small batches)

---

## 10. Security and Stability

### 10.1 Numerical Stability

✅ **Well Handled**:
```python
group_stds = grouped.std(dim=1, keepdim=True) + 1e-8  # Prevents division by zero
```

✅ **Gradient Clipping**:
```python
th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
```

### 10.2 Edge Cases

✅ **Batch Size < Group Size**: Handled correctly
✅ **Non-Divisible Batches**: Remainder handled
✅ **Early Stopping**: KL divergence threshold working

---

## 11. Conclusion

### 11.1 Summary of Findings

The GRPO implementation in stable-baselines3-contrib is:

1. **Mathematically Sound**: The code is correct and well-implemented
2. **Not Paper-Accurate**: Uses GAE instead of direct reward normalization
3. **Appropriately Adapted**: Design choices suit standard RL environments
4. **Well-Tested**: Good test coverage for functionality
5. **Under-Documented**: Needs clarity on deviations from paper

### 11.2 Final Verdict

**Implementation Quality**: ⭐⭐⭐⭐½ (4.5/5)
- Excellent code quality
- Solid engineering
- Missing documentation transparency

**Paper Fidelity**: ⭐⭐½ (2.5/5)
- Captures spirit, not letter
- Significant algorithmic differences
- Appropriate for different domain

**Overall Assessment**: This is a **high-quality implementation** of a **GRPO-inspired algorithm** adapted for standard RL. It should be clearly marketed as such, not as a direct implementation of the DeepSeek-Math paper.

### 11.3 Action Items Priority

**High Priority** (Must do before merging/releasing):
1. Update documentation to clarify deviations from paper
2. Create `docs/modules/grpo.rst`
3. Add "Implementation Notes" section to docstring

**Medium Priority** (Should do soon):
4. Benchmark against PPO on standard tasks
5. Document performance characteristics
6. Add ablation study results

**Low Priority** (Nice to have):
7. Consider implementing "pure" GRPO variant
8. Add architectural diagrams
9. Extended examples and tutorials

---

## 12. References

1. **Original Paper**: Shao, Z., et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

2. **GRPO Explanations**:
   - [Group Relative Policy Optimization (GRPO) Illustrated Breakdown](https://epichka.com/blog/2025/grpo/)
   - [The Illustrated GRPO](https://abderrahmanskiredj.github.io/the-illustrated-grpo/)
   - [HuggingFace GRPO Guide](https://huggingface.co/learn/llm-course/chapter12/3b)

3. **Related Work**:
   - Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
   - Schulman, J., et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv:1506.02438.

4. **Implementation References**:
   - [GRPO-Zero: GRPO from Scratch](https://github.com/policy-gradient/GRPO-Zero)
   - [GDPO: Group reward-Decoupled Normalization](https://github.com/NVlabs/GDPO)
   - [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Maintainer**: stable-baselines3-contrib team
