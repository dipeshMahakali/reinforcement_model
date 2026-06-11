import unittest
import numpy as np
import torch
import gymnasium as gym

# Import classes to be tested from jarvis_agent
from jarvis_agent import IntentParser, DynamicAgentEnv, PolicyNetwork, REINFORCEAgent

class TestJarvisAgentComponents(unittest.TestCase):
    
    def setUp(self):
        self.parser = IntentParser()
        # Sample prompt and parsed parameters for setup
        self.task_config = self.parser.parse_user_prompt("Jarvis, sort my downloads folder")
        self.env = DynamicAgentEnv(
            initial_state=self.task_config["initial_state"],
            goal_state=self.task_config["goal_state"],
            reward_rules=self.task_config["reward_rules"]
        )
        self.agent = REINFORCEAgent(state_dim=self.env.state_dim, action_dim=self.env.action_space.n)

    def test_intent_parser(self):
        """Test if the parser correctly maps natural language to tasks and rewards."""
        # 1. Sort task detection
        sort_config = self.parser.parse_user_prompt("Clean up downloads and sort files")
        self.assertEqual(sort_config["initial_state"], [5, 0, 0, 0])
        self.assertEqual(sort_config["goal_state"], [0, 5, 0, 0])
        self.assertLess(sort_config["reward_rules"]["critical_deletion_penalty"], -20.0) # -25.0
        
        # 2. Backup task detection
        backup_config = self.parser.parse_user_prompt("Jarvis, backup my project files")
        self.assertEqual(backup_config["initial_state"], [3, 0, 0, 0])
        self.assertEqual(backup_config["goal_state"], [0, 3, 0, 0])
        
        # 3. Default case detection
        default_config = self.parser.parse_user_prompt("hello jarvis")
        self.assertEqual(default_config["initial_state"], [5, 0, 0, 0])

    def test_gymnasium_env_transitions(self):
        """Test environment transitions, safety penalties, and goal termination."""
        state, _ = self.env.reset()
        self.assertTrue(np.array_equal(state, np.array([5, 0, 0, 0], dtype=np.float32)))
        
        # Action 1 (MOVE_FILE) should decrease pending and increase completed
        next_state, reward, terminated, truncated, _ = self.env.step(1)
        self.assertEqual(next_state[0], 4)
        self.assertEqual(next_state[1], 1)
        self.assertGreater(reward, 0.0) # correct sorting rewards positive
        
        # Action 2 (DELETE_CRITICAL_FILE) should trigger safety violation penalty
        next_state, reward, terminated, truncated, _ = self.env.step(2)
        self.assertEqual(next_state[2], 1)
        self.assertLess(reward, -10.0) # penalty is negative
        
        # Test out-of-bounds/no pending files error penalty
        self.env.state = np.array([0, 5, 0, 0], dtype=np.float32)
        next_state, reward, terminated, truncated, _ = self.env.step(1)
        self.assertEqual(next_state[3], 1) # error count increases
        self.assertLess(reward, 0.0)

    def test_policy_network(self):
        """Test policy network dimensions and probability distributions."""
        state_tensor = torch.FloatTensor([5, 0, 0, 0])
        probs = self.agent.policy(state_tensor)
        
        # Output dim must equal action space size (4)
        self.assertEqual(probs.shape[0], 4)
        # Probabilities must sum to 1.0
        self.assertAlmostEqual(torch.sum(probs).item(), 1.0, places=5)
        # No negative probabilities
        self.assertTrue(torch.all(probs >= 0.0).item())

    def test_agent_action_selection_and_updates(self):
        """Test action selection shapes and update cycle."""
        state = [5, 0, 0, 0]
        action, log_prob = self.agent.get_action(state)
        self.assertIn(action, [0, 1, 2, 3])
        self.assertIsInstance(log_prob, torch.Tensor)
        
        # Simulating dummy rollouts for a training step update
        dummy_rewards = [10.0, -0.5, 50.0]
        dummy_log_probs = [log_prob, log_prob, log_prob]
        
        # Test update call without errors
        try:
            self.agent.train_step(dummy_rewards, dummy_log_probs)
            success = True
        except Exception as e:
            success = False
            print(f"Agent training step failed with: {e}")
            
        self.assertTrue(success)

    def test_defensive_parser(self):
        """Test how parser handles malformed or adversarial inputs."""
        # None input
        none_config = self.parser.parse_user_prompt(None)
        self.assertEqual(none_config["initial_state"], [5, 0, 0, 0])
        
        # Empty string
        empty_config = self.parser.parse_user_prompt("")
        self.assertEqual(empty_config["initial_state"], [5, 0, 0, 0])
        
        # Non-string input (integer)
        int_config = self.parser.parse_user_prompt(12345)
        self.assertEqual(int_config["initial_state"], [5, 0, 0, 0])

    def test_scenario_cache_clearance(self):
        """Test cache clearance task parsing and environment settings."""
        cache_config = self.parser.parse_user_prompt("Jarvis, erase temporary cache files")
        self.assertEqual(cache_config["initial_state"], [8, 0, 0, 0])
        self.assertEqual(cache_config["goal_state"], [0, 8, 0, 0])
        self.assertEqual(cache_config["reward_rules"]["critical_deletion_penalty"], -30.0)

    def test_clamping_and_numerical_stability(self):
        """Verify that action probability clamping prevents NaN or inf log-probs."""
        # Simulate extreme state inputs that might push weights to output near-deterministic probabilities
        state_tensor = torch.FloatTensor([1000.0, -1000.0, 1000.0, -1000.0])
        probs = self.agent.policy(state_tensor)
        # Use PyTorch assertion to compare within float32 precision or allow small margin in float64 comparison
        self.assertTrue(torch.all(probs >= 9.9e-9).item())
        self.assertTrue(torch.all(probs <= 1.0 - 9.9e-9).item())

if __name__ == "__main__":
    unittest.main()
