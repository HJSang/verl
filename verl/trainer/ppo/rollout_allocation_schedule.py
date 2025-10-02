# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rollout allocation schedule implementations for mixture of rollout models.

This module provides various scheduling strategies to determine which policy
(actor or fixed) to use for generating rollouts at each training step.
"""

from abc import ABC, abstractmethod
from typing import Literal
import random
import numpy as np


class RolloutAllocationSchedule(ABC):
    """Base class for rollout allocation schedules."""
    
    @abstractmethod
    def get_policy_choice(self, step: int) -> Literal["actor", "fixed"]:
        """Get which policy to use for current step.
        
        Args:
            step: Current training step
            
        Returns:
            "actor" or "fixed" - which policy to use for all rollouts this step
        """
        pass


class ConstantSchedule(RolloutAllocationSchedule):
    """Constant probability schedule: flip coin with prob alpha each step."""
    
    def __init__(self, alpha: float):
        """
        Args:
            alpha: Probability of using actor policy (0.0 to 1.0)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
        self.alpha = alpha
        
    def get_policy_choice(self, step: int) -> Literal["actor", "fixed"]:
        choice = "actor" if random.random() < self.alpha else "fixed"
        print(f"[CONSTANT_SCHEDULE] Step {step}: alpha={self.alpha:.3f}, choice={choice}")
        return choice


class LinearSchedule(RolloutAllocationSchedule):
    """Linear schedule: alpha = alpha_0 + beta * step, capped at 1.0."""
    
    def __init__(self, alpha_0: float, beta: float, max_alpha: float = 1.0):
        """
        Args:
            alpha_0: Initial alpha value
            beta: Linear increase rate per step
            max_alpha: Maximum alpha value (default 1.0)
        """
        if not 0.0 <= alpha_0 <= 1.0:
            raise ValueError(f"alpha_0 must be between 0.0 and 1.0, got {alpha_0}")
        if not 0.0 <= max_alpha <= 1.0:
            raise ValueError(f"max_alpha must be between 0.0 and 1.0, got {max_alpha}")
        if alpha_0 > max_alpha:
            raise ValueError(f"alpha_0 ({alpha_0}) must be <= max_alpha ({max_alpha})")
        
        self.alpha_0 = alpha_0
        self.beta = beta
        self.max_alpha = max_alpha
        
    def get_policy_choice(self, step: int) -> Literal["actor", "fixed"]:
        # Compute current alpha
        alpha = min(self.alpha_0 + self.beta * step, self.max_alpha)
        choice = "actor" if random.random() < alpha else "fixed"
        print(f"[LINEAR_SCHEDULE] Step {step}: alpha={alpha:.3f} (alpha_0={self.alpha_0:.3f}, beta={self.beta:.3f}), choice={choice}")
        return choice


class ExponentialSchedule(RolloutAllocationSchedule):
    """Exponential schedule: alpha = 1 - exp(-gamma * step)."""
    
    def __init__(self, gamma: float):
        """
        Args:
            gamma: Exponential decay rate
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = gamma
        
    def get_policy_choice(self, step: int) -> Literal["actor", "fixed"]:
        alpha = 1 - np.exp(-self.gamma * step)
        choice = "actor" if random.random() < alpha else "fixed"
        print(f"[EXPONENTIAL_SCHEDULE] Step {step}: alpha={alpha:.3f} (gamma={self.gamma:.3f}), choice={choice}")
        return choice


class StepSchedule(RolloutAllocationSchedule):
    """Step schedule: switch at specific step thresholds."""
    
    def __init__(self, switch_steps: list[int], initial_policy: Literal["actor", "fixed"] = "fixed"):
        """
        Args:
            switch_steps: List of steps at which to switch policies
            initial_policy: Policy to use before first switch
        """
        if not isinstance(switch_steps, list) or not all(isinstance(s, int) for s in switch_steps):
            raise ValueError("switch_steps must be a list of integers")
        if initial_policy not in ["actor", "fixed"]:
            raise ValueError(f"initial_policy must be 'actor' or 'fixed', got {initial_policy}")
        
        self.switch_steps = sorted(switch_steps)
        self.initial_policy = initial_policy
        
    def get_policy_choice(self, step: int) -> Literal["actor", "fixed"]:
        # Count how many switches have occurred
        switches = sum(1 for switch_step in self.switch_steps if step >= switch_step)
        
        # Alternate between policies based on number of switches
        if switches % 2 == 0:
            choice = self.initial_policy
        else:
            choice = "fixed" if self.initial_policy == "actor" else "actor"
        
        print(f"[STEP_SCHEDULE] Step {step}: switches={switches}, switch_steps={self.switch_steps}, choice={choice}")
        return choice


def create_rollout_allocation_schedule(config: dict) -> RolloutAllocationSchedule:
    """Create rollout allocation schedule from config.
    
    Args:
        config: Configuration dictionary containing schedule parameters
        
    Returns:
        RolloutAllocationSchedule instance
        
    Raises:
        ValueError: If schedule type is not supported or parameters are invalid
    """
    schedule_type = config.get("type", "constant")
    
    if schedule_type == "constant":
        alpha = config.get("alpha", 0.5)
        return ConstantSchedule(alpha=alpha)
    
    elif schedule_type == "linear":
        alpha_0 = config.get("alpha_0", 0.1)
        beta = config.get("beta", 0.01)
        max_alpha = config.get("max_alpha", 1.0)
        return LinearSchedule(alpha_0=alpha_0, beta=beta, max_alpha=max_alpha)
    
    elif schedule_type == "exponential":
        gamma = config.get("gamma", 0.1)
        return ExponentialSchedule(gamma=gamma)
    
    elif schedule_type == "step":
        switch_steps = config.get("switch_steps", [])
        initial_policy = config.get("initial_policy", "fixed")
        return StepSchedule(switch_steps=switch_steps, initial_policy=initial_policy)
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}. "
                        f"Supported types: constant, linear, exponential, step")
