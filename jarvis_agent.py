import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces

# =====================================================================
# 📚 0. REQUIRED LIBRARIES & ENVIRONMENT SETUP
# =====================================================================
# To run this script locally, ensure you have the following installed:
#   pip install torch gymnasium numpy
# =====================================================================


# =====================================================================
# 🌐 1. THE INTENT PARSER (The Conversational Bridge)
# =====================================================================
class IntentParser:
    """
    Simulates a local conversational interface. It parses natural language prompts
    and dynamically extracts the environment goals and the reward function weights.
    """
    def __init__(self):
        pass

    def parse_user_prompt(self, prompt: str):
        """
        Analyzes a natural language prompt to generate the training target state
        and a specific reward schema dynamically.
        """
        # Defensive check: ensure prompt is a valid string
        if prompt is None:
            prompt = ""
        else:
            prompt = str(prompt)
            
        # Normalize prompt
        prompt_lower = prompt.strip().lower()
        
        # Default target configuration
        initial_state = [5, 0, 0, 0]  # [unsorted_files, sorted_files, critical_deleted, error_count]
        goal_state = [0, 5, 0, 0]
        
        # Dynamic reward weights mapping
        reward_rules = {
            "correct_sort": 10.0,
            "critical_deletion_penalty": -20.0,
            "error_penalty": -5.0,
            "step_penalty": -0.5,
            "goal_bonus": 50.0
        }
        
        # Custom Parsing Logic representing natural language mapping
        if "downloads" in prompt_lower or "sort" in prompt_lower:
            # Sort task settings
            initial_state = [5, 0, 0, 0] # 5 files to sort
            goal_state = [0, 5, 0, 0]
            reward_rules["correct_sort"] = 12.0  # Increased reward value
            reward_rules["critical_deletion_penalty"] = -25.0 # Higher safety penalty
            print("🤖 Intent Parser: Detected file cleanup & sorting task.")
            
        elif "backup" in prompt_lower or "copy" in prompt_lower:
            # Backup task settings
            initial_state = [3, 0, 0, 0] # 3 files to backup
            goal_state = [0, 3, 0, 0]
            reward_rules["correct_sort"] = 15.0  # Used as task success reward
            print("🤖 Intent Parser: Detected backup/copy task.")
            
        elif any(keyword in prompt_lower for keyword in ["cache", "temp", "clean", "erase"]):
            # Clear temporary files / cache
            initial_state = [8, 0, 0, 0] # 8 cache files to clear
            goal_state = [0, 8, 0, 0]
            reward_rules["correct_sort"] = 10.0 # Reward per cache file cleared
            reward_rules["critical_deletion_penalty"] = -30.0 # High penalty for deleting critical folders
            print("🤖 Intent Parser: Detected cache clearance / temp file deletion task.")
            
        else:
            print("🤖 Intent Parser: Generic task detected. Using default configuration.")

        return {
            "initial_state": initial_state,
            "goal_state": goal_state,
            "reward_rules": reward_rules
        }


# =====================================================================
# 🛠️ 2. THE DYNAMIC GYMNASIUM ENVIRONMENT
# =====================================================================
class DynamicAgentEnv(gym.Env):
    """
    A custom, dynamic Gymnasium environment initialized using the output 
    of the Intent Parser. State variables change dynamically based on the 
    user's target state and reward guidelines.
    """
    def __init__(self, initial_state, goal_state, reward_rules):
        super(DynamicAgentEnv, self).__init__()
        
        self.initial_state = np.array(initial_state, dtype=np.float32)
        self.goal_state = np.array(goal_state, dtype=np.float32)
        self.reward_rules = reward_rules
        
        # State space representation:
        # [0]: Unsorted/pending items
        # [1]: Correctly processed items
        # [2]: Critical items deleted/ruined (Safety violation)
        # [3]: Invalid action count
        self.state_dim = len(initial_state)
        
        # Action space: 
        #   0: READ_DIR (Safety check/inspect directory - low penalty)
        #   1: MOVE_FILE (Performs sorting/relocation)
        #   2: DELETE_CRITICAL_FILE (Destructive operation - safety penalty)
        #   3: NO_OP (idle step)
        self.action_space = spaces.Discrete(4)
        
        # Observation space accepts the state vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        self.state = None
        self.steps_taken = 0
        self.max_steps = 15
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_state.copy()
        self.steps_taken = 0
        return self.state, {}
        
    def step(self, action):
        self.steps_taken += 1
        reward = 0.0
        terminated = False
        truncated = False
        
        # Keep track of previous state
        prev_state = self.state.copy()
        
        # Step reward penalty to encourage the policy to find the shortest path
        step_penalty = self.reward_rules.get("step_penalty", -0.5)
        reward += step_penalty
        
        # Action State Transitions
        if action == 0:  # READ_DIR
            # Inspect files: harmless operation
            pass
            
        elif action == 1:  # MOVE_FILE
            if self.state[0] > 0:
                self.state[0] -= 1  # Reduce pending files
                self.state[1] += 1  # Increase sorted files
                reward += self.reward_rules.get("correct_sort", 10.0)
            else:
                # No files are left to move: invalid action error
                self.state[3] += 1
                reward += self.reward_rules.get("error_penalty", -5.0)
                
        elif action == 2:  # DELETE_CRITICAL_FILE
            # Highly dangerous action
            self.state[2] += 1
            reward += self.reward_rules.get("critical_deletion_penalty", -20.0)
            
        elif action == 3:  # NO_OP
            # Idling
            pass
            
        # Defensive check to prevent state dimensions from underflowing
        self.state = np.maximum(self.state, 0.0)
        
        # Goal evaluation (only award bonus on transition)
        if self.state[0] == 0 and prev_state[0] > 0:
            terminated = True
            reward += self.reward_rules.get("goal_bonus", 50.0)
            
        # Step limit cut-off
        if self.steps_taken >= self.max_steps:
            truncated = True
            
        return self.state, reward, terminated, truncated, {}


# =====================================================================
# 🧠 3. THE REINFORCEMENT LEARNING AGENT
# =====================================================================
class PolicyNetwork(nn.Module):
    """
    Lightweight policy network (brain) that receives the state vector
    as input and returns action probabilities.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, state):
        logits = self.network(state)
        probs = torch.softmax(logits, dim=-1)
        # Clamp probabilities to avoid NaN/negative-infinity log-probabilities during backprop
        return torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)


class REINFORCEAgent:
    """
    Implements a policy gradient update step using log-probabilities 
    reinforced by rewards collected during rollout episodes.
    """
    def __init__(self, state_dim, action_dim, lr=0.003, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.baseline = 0.0
        
    def get_action(self, state):
        state_t = torch.FloatTensor(state)
        probs = self.policy(state_t)
        
        # Sample an action from the action probability distribution
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)
        
    def train_step(self, rewards, log_probs):
        """
        Applies mathematical updates based on discounted cumulative rewards.
        
        🧠 RLAIF Mechanics:
        Higher rewards increase the log-probability of successful actions taken in 
        that state (making them more likely to repeat), while negative rewards/penalties 
        suppress action likelihood, directly shifting the policy away from mistakes.
        """
        discounted_rewards = []
        G = 0
        
        # Calculate return backwards: G_t = sum(gamma^k * R_t+k)
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_rewards.insert(0, G)
            
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # Update and subtract moving average baseline to reduce gradient variance
        mean_return = discounted_rewards.mean().item()
        self.baseline = 0.9 * self.baseline + 0.1 * mean_return
        discounted_rewards = discounted_rewards - self.baseline
            
        policy_loss = []
        for log_prob, g in zip(log_probs, discounted_rewards):
            # Gradient ascent objective: maximize E[log pi(a|s) * (G - b)] -> minimize -log pi(a|s) * (G - b)
            policy_loss.append(-log_prob * g)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()


# =====================================================================
# 🔄 4. THE EXECUTION & EVALUATION LOOP
# =====================================================================
def run_autonomous_agent(user_command: str, train_episodes: int = 300):
    print(f"\n💬 User Prompt: '{user_command}'")
    
    # Step 1: Parse intent and construct dynamic goals/rewards
    parser = IntentParser()
    task_config = parser.parse_user_prompt(user_command)
    
    # Step 2: Initialize environment
    env = DynamicAgentEnv(
        initial_state=task_config["initial_state"],
        goal_state=task_config["goal_state"],
        reward_rules=task_config["reward_rules"]
    )
    
    # Step 3: Initialize RL Brain
    agent = REINFORCEAgent(state_dim=env.state_dim, action_dim=env.action_space.n)
    
    print("\n🚀 Training policy on simulator...")
    
    # Track historical rewards for evaluation
    recent_rewards = []
    
    for episode in range(train_episodes):
        state, info = env.reset()
        rewards = []
        log_probs = []
        
        for step in range(env.max_steps):
            action, log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
            
            if terminated or truncated:
                break
                
        # Perform Policy Gradient Optimization
        agent.train_step(rewards, log_probs)
        recent_rewards.append(sum(rewards))
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(recent_rewards[-50:])
            print(f"  Episode {episode+1:3d}/{train_episodes} | Average Reward (Last 50): {avg_reward:7.2f}")
            
    # Step 4: Execute final learned policy
    print("\n🎬 Executing Learned Sequence:")
    state, info = env.reset()
    actions_taken = []
    
    for step in range(env.max_steps):
        # Sample action greedily (without exploration)
        state_t = torch.FloatTensor(state)
        with torch.no_grad():
            probs = agent.policy(state_t)
        action = torch.argmax(probs).item()
        
        state, reward, terminated, truncated, _ = env.step(action)
        
        # Decode action to human-readable string
        action_map = {
            0: "READ_DIR",
            1: "MOVE_FILE",
            2: "DELETE_CRITICAL_FILE",
            3: "NO_OP"
        }
        actions_taken.append(action_map[action])
        
        if terminated or truncated:
            break
            
    print(f"  Final Action Sequence: {' -> '.join(actions_taken)}")
    print(f"  Final State Vector: {state} (Goal state: {env.goal_state})")
    
    if state[0] == 0 and state[2] == 0:
        print("  🎉 Execution Status: SUCCESS (Goal reached safely!)")
    else:
        print("  ⚠️ Execution Status: FAILED (Goal not reached or safety violation occurred)")


if __name__ == "__main__":
    # Test Run 1: Sorting and Clearing downloads task
    run_autonomous_agent(
        user_command="Jarvis, clear out my downloads folder and sort the files by extension",
        train_episodes=300
    )
    
    # Test Run 2: Copying / Backup task
    run_autonomous_agent(
        user_command="Jarvis, copy my project files to the backup drive",
        train_episodes=300
    )
    
    # Test Run 3: Cache clearing task
    run_autonomous_agent(
        user_command="Jarvis, please erase the temporary cache files",
        train_episodes=400
    )
