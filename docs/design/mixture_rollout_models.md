git # Mixture of Rollout Models Design

## Overview

This document describes the design for supporting mixture of rollout models in VERL, where we can use both a trainable actor policy and a fixed policy model for generating rollouts during PPO training. The key insight is that we can use different policies as reference distributions for importance sampling in policy gradient methods.

## Mathematical Foundation

For policy gradient methods, we can use any reference policy as long as we compute the correct importance weights:

```
E{r(x, y)} = E{π(x, y)/π_old(x, y) * r(x, y)} = E{π(x, y)/π_fixed(x, y) * r(x, y)}
```

Where:
- **Fixed samples** provide the reference distribution π_fixed(x, y)
- **Actor samples** provide the current policy π(x, y)
- **Importance weights** are computed as π(x, y) / π_fixed(x, y)

## Architecture

### 1. Rollout Allocation Schedule

The system uses a flexible scheduling mechanism to determine which policy to use at each training step.

#### Base Schedule Class

```python
class RolloutAllocationSchedule(ABC):
    @abstractmethod
    def get_policy_choice(self, step: int) -> Literal["actor", "fixed"]:
        """Get which policy to use for current step."""
        pass
```

#### Implemented Schedules

1. **ConstantSchedule**: Random coin flip with probability α
2. **LinearSchedule**: α = α₀ + β * step, capped at 1.0
3. **ExponentialSchedule**: α = 1 - exp(-γ * step)
4. **StepSchedule**: Deterministic switching at specific steps

### 2. Configuration Structure

#### Minimal Configuration Approach

The `rollout_only` configuration only specifies differences from `actor_rollout_ref.rollout`:

```yaml
# ppo_trainer.yaml
rollout_only:
  # Model path for fixed policy (only difference from actor_rollout)
  model_path: "/path/to/your/fixed/policy/model"
  
  # Sampling parameters (only differences from actor_rollout)
  temperature: 0.8
  top_p: 0.95
  top_k: 40
  
  # All other parameters are inherited from actor_rollout_ref.rollout

rollout_allocation_schedule:
  type: "constant"  # or "linear", "exponential", "step"
  alpha: 0.5
  # Additional parameters based on schedule type
```

### 3. Worker Architecture

#### Role Definition

```python
class Role(Enum):
    ActorRollout = "actor_rollout"
    Critic = "critic"
    RewardModel = "reward_model"
    RefPolicy = "ref_policy"
    RolloutOnly = "rollout_only"  # New role for fixed policy
```

#### Worker Initialization

- **Same Worker Class**: Both actor and fixed policy use `ActorRolloutRefWorker`
- **Different Role**: Actor uses `role="actor_rollout"`, fixed uses `role="rollout"`
- **Different Config**: Fixed policy inherits from actor config with overrides
- **Same Resource Pool**: Both workers share the same GPU resources

### 4. Training Loop Integration

#### Policy Choice Decision

At each training step:
1. Query the allocation schedule for policy choice
2. Generate rollouts using the selected policy
3. Mark samples with policy type metadata
4. Continue with standard PPO training

#### Rollout Generation

```python
def _generate_rollouts(self, gen_batch: DataProto) -> DataProto:
    policy_choice = self.rollout_allocation_schedule.get_policy_choice(self.global_steps)
    
    if policy_choice == "actor":
        return self._generate_actor_rollouts(gen_batch)
    else:
        return self._generate_fixed_rollouts(gen_batch)
```

#### Advantage Calculation

The advantage calculation remains unchanged since it works on the mixed samples with proper importance weighting.

#### Actor Updates

Both actor and fixed policy samples contribute to policy gradient updates:

- **Actor samples**: Direct policy gradient updates using standard PPO
- **Fixed samples**: Policy gradient updates using importance sampling weights π_current(x,y) / π_fixed(x,y)

Fixed policy samples can also be used for:
- Value function learning (critic training)
- Reward computation
- Log probability computation

## Implementation Details

### 1. Configuration Inheritance

```python
def _create_rollout_only_config(self):
    """Create rollout-only config by inheriting from actor_rollout_ref.rollout."""
    from omegaconf import OmegaConf
    
    # Start with actor_rollout_ref.rollout config
    rollout_config = OmegaConf.create(self.config.actor_rollout_ref.rollout)
    
    # Override model path
    rollout_config.model.path = self.config.rollout_only.model_path
    
    # Override sampling parameters
    rollout_config.temperature = self.config.rollout_only.get("temperature", rollout_config.temperature)
    rollout_config.top_p = self.config.rollout_only.get("top_p", rollout_config.top_p)
    rollout_config.top_k = self.config.rollout_only.get("top_k", rollout_config.top_k)
    
    # Ensure calculate_log_probs is True for rollout-only
    rollout_config.calculate_log_probs = True
    
    return rollout_config
```

### 2. Importance Sampling Implementation

For fixed policy samples, importance sampling is handled by using the fixed policy to calculate `old_log_probs`:

```python
# In the training loop when computing old_log_probs:
policy_type = batch.meta_info.get("policy_type", "current")

if policy_type == "reference" and self.rollout_only_wg is not None:
    # Use fixed policy to calculate old_log_probs for importance sampling
    old_log_prob = self.rollout_only_wg.compute_log_prob(batch)
else:
    # Use current actor policy (standard PPO)
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

# The policy ratio calculation automatically becomes:
# ratio = exp(log_prob - old_log_prob) = π_current / π_fixed
# No changes needed to loss function or advantage computation
```

This approach is much cleaner because:
- No modifications to the loss function
- No need to compute importance weights separately
- The standard PPO ratio calculation handles importance sampling automatically
- GRPO advantages work unchanged

### 3. Resource Pool Management

Both workers share the same resource pool to maximize GPU utilization:

```python
def init_resource_pool_mgr(self, config):
    # ... existing setup ...
    
    # Map rollout-only worker to global pool
    if config.rollout_only.model_path is not None:
        self.mapping[Role.RolloutOnly] = global_pool_id
```

### 4. Monitoring and Logging

The system tracks allocation decisions, policy usage, and importance sampling statistics:

```python
metrics.update({
    "rollout/policy_choice": 1 if policy_choice == "actor" else 0,
    "rollout/using_actor": policy_choice == "actor",
    "rollout/using_fixed": policy_choice == "fixed",
    "actor/importance_sampling": 1 if using_fixed_policy else 0,
    "actor/importance_weight_mean": importance_weight_mean,
    "actor/importance_weight_std": importance_weight_std,
    "actor/importance_weight_min": importance_weight_min,
    "actor/importance_weight_max": importance_weight_max,
})
```

## Usage Examples

### 1. Constant Schedule (50/50 Random)

```yaml
rollout_allocation_schedule:
  type: "constant"
  alpha: 0.5  # 50% chance of using actor policy
```

### 2. Linear Schedule (Gradual Transition)

```yaml
rollout_allocation_schedule:
  type: "linear"
  alpha_0: 0.1  # Start with 10% actor probability
  beta: 0.01    # Increase by 1% per step
  max_alpha: 1.0
```

### 3. Step Schedule (Deterministic Switching)

```yaml
rollout_allocation_schedule:
  type: "step"
  switch_steps: [100, 500, 1000]  # Switch at these steps
  initial_policy: "fixed"  # Start with fixed policy
```

### 4. Command Line Usage

```bash
python verl/trainer/main_ppo.py \
  --config-path=configs \
  --config-name=ppo_trainer \
  rollout_only.model_path=/path/to/your/fixed/policy/model \
  rollout_only.temperature=0.8 \
  rollout_only.top_p=0.95 \
  rollout_only.top_k=40 \
  rollout_allocation_schedule.type=constant \
  rollout_allocation_schedule.alpha=0.5
```

## Benefits

1. **Sample Efficiency**: Use fixed policy samples to improve value function estimation
2. **Stable Training**: Fixed policy provides a stable reference distribution
3. **Flexible Scheduling**: Control the ratio of fixed vs. current policy samples
4. **Correct Policy Gradient**: Importance weights ensure correct policy gradient computation
5. **Minimal Configuration**: Only specify differences from base configuration
6. **Resource Efficiency**: Both workers share the same GPU resources

## File Structure

```
verl/
├── trainer/
│   ├── ppo/
│   │   ├── rollout_allocation_schedule.py  # New: Schedule implementations
│   │   └── ray_trainer.py                  # Modified: Integration
│   └── main_ppo.py                         # Modified: Worker creation
├── workers/
│   └── fsdp_workers.py                     # Modified: Role support
└── docs/
    └── design/
        └── mixture_rollout_models.md       # This document
```

## Future Extensions

1. **Adaptive Scheduling**: Dynamic adjustment based on training metrics
2. **Multi-Policy Support**: Support for more than two policies
3. **Policy Ensembles**: Combine multiple fixed policies
4. **Curriculum Learning**: Gradually transition from fixed to actor policy
5. **Online Policy Selection**: Choose policy based on sample quality

## Testing Strategy

1. **Unit Tests**: Test individual schedule implementations
2. **Integration Tests**: Test full training loop with mixed policies
3. **Configuration Tests**: Verify configuration inheritance works correctly
4. **Performance Tests**: Ensure no significant overhead from policy switching
5. **Convergence Tests**: Verify training converges with mixed policies

## Migration Guide

### For Existing Users

1. **No Changes Required**: Existing configurations continue to work
2. **Optional Feature**: Only enabled when `rollout_only.model_path` is specified
3. **Backward Compatible**: All existing functionality preserved

### For New Users

1. **Specify Fixed Policy**: Set `rollout_only.model_path`
2. **Choose Schedule**: Select appropriate allocation schedule
3. **Tune Parameters**: Adjust sampling parameters as needed
4. **Monitor Training**: Use provided metrics to track policy usage
