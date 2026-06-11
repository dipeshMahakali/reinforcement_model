# 🌐 Zero-Dependency Hosting Plan: Jarvis RL Web Console

To run, inspect, and evaluate the Jarvis Agent without installing PyTorch, Gymnasium, or Python on your system, we can package it inside a **Docker Container** paired with a **Streamlit Web UI**. 

This plan details how to set this up locally or deploy it to the cloud.

---

## 🛠️ 1. The Containerized Web UI Architecture
We wrap the agent code in a simple Streamlit interface (`app.py`) that lets you input commands, click a button to train the policy network, and visualize the learning curve and decision sequence in real time.

```
+-------------------------------------------------------------+
|                     User Browser (Web UI)                   |
|  [ Input Prompt ] -> [ Click Train ] -> [ Dynamic Charts ]  |
+------------------------------+------------------------------+
                               | HTTP / WebSockets
                               v
+------------------------------+------------------------------+
|                      Docker Container                       |
|   +-----------------------------------------------------+   |
|   |         Streamlit App Server (Port 8501)            |   |
|   |  +--------------------+    +---------------------+  |   |
|   |  |   Intent Parser    |    |  Gymnasium Sandbox  |  |   |
|   |  +--------------------+    +---------------------+  |   |
|   |  +--------------------+    +---------------------+  |   |
|   |  | PyTorch Policy Net |    |   REINFORCE Loop    |  |   |
|   |  +--------------------+    +---------------------+  |   |
|   +-----------------------------------------------------+   |
+-------------------------------------------------------------+
```

---

## 📂 2. Configuration Files

### File A: `app.py` (Streamlit Web Interface)
Create this file in your project folder to build the interactive UI:
```python
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from jarvis_agent import IntentParser, DynamicAgentEnv, REINFORCEAgent

st.set_page_config(page_title="Jarvis RL Console", layout="wide")
st.title("🧠 Jarvis RL Agentic Loop Web Console")
st.markdown("Fine-tune a local Reinforcement Learning policy network based on your natural language inputs.")

# Sidebar controls
st.sidebar.header("Hyperparameters")
lr = st.sidebar.slider("Learning Rate", 0.001, 0.05, 0.01, step=0.001)
episodes = st.sidebar.slider("Training Episodes", 50, 1000, 300, step=50)

# Main Prompt Input
prompt = st.text_input("Enter conversational command for Jarvis:", "Jarvis, sort my downloads folder")

if st.button("🚀 Train & Run Agent"):
    # 1. Parse intent
    parser = IntentParser()
    task_config = parser.parse_user_prompt(prompt)
    
    st.write(f"**Detected Task Rules:** `{task_config['reward_rules']}`")
    
    # 2. Setup environment & Agent
    env = DynamicAgentEnv(
        initial_state=task_config["initial_state"],
        goal_state=task_config["goal_state"],
        reward_rules=task_config["reward_rules"]
    )
    agent = REINFORCEAgent(state_dim=env.state_dim, action_dim=env.action_space.n, lr=lr)
    
    # 3. Training Loop
    rewards_history = []
    progress_bar = st.progress(0.0)
    
    for ep in range(episodes):
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
        
        agent.train_step(rewards, log_probs)
        rewards_history.append(sum(rewards))
        progress_bar.progress((ep + 1) / episodes)
        
    # 4. Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Training Performance")
        fig, ax = plt.subplots()
        # Smooth rewards using rolling window
        rolling_r = np.convolve(rewards_history, np.ones(10)/10, mode='valid')
        ax.plot(rolling_r, label="Rolling Avg Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.grid(True)
        st.pyplot(fig)
        
    with col2:
        st.subheader("🎬 Final Policy Execution")
        state, info = env.reset()
        actions_taken = []
        action_map = {0: "READ_DIR", 1: "MOVE_FILE", 2: "DELETE_CRITICAL_FILE", 3: "NO_OP"}
        
        for step in range(env.max_steps):
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                probs = agent.policy(state_t)
            action = torch.argmax(probs).item()
            state, reward, terminated, truncated, _ = env.step(action)
            actions_taken.append(action_map[action])
            if terminated or truncated:
                break
                
        st.write("👉 **Action Path:**")
        st.success(" -> ".join(actions_taken))
        st.write(f"🎯 **Final State:** `{state}` (Target: `{env.goal_state}`)")
```

### File B: `Dockerfile`
Create a `Dockerfile` to containerize the environment:
```dockerfile
# Use a lightweight official PyTorch image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system utilities needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    gymnasium \
    numpy \
    matplotlib \
    streamlit

# Copy app files into the container
COPY jarvis_agent.py .
COPY app.py .

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🚀 3. How to Run Locally with Docker
Once you save the above files in your project directory:

1. **Build the Docker Image**:
   ```bash
   docker build -t jarvis-rl-console .
   ```
2. **Run the Container**:
   ```bash
   docker run -p 8501:8501 jarvis-rl-console
   ```
3. **Inspect the Console**:
   Open [http://localhost:8501](http://localhost:8501) in your browser. You can fully test prompts and watch the training process live with zero Python dependencies installed on your host system.

---

## ☁️ 4. Free Cloud Hosting Options (No Setup Required)

If you want to host it online, the following methods require no server maintenance:

### Option A: Hugging Face Spaces (Highly Recommended)
Hugging Face offers free, instant hosting for Streamlit/Gradio apps:
1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Click **New Space** and select **Streamlit** as the SDK.
3. Push your files (`app.py`, `jarvis_agent.py`, and a simple `requirements.txt` containing `gymnasium`, `torch`, `matplotlib`) to the Space's Git repository.
4. Hugging Face automatically builds the container and serves it under a public HTTPS URL.

### Option B: Render / Google Cloud Run
- **Render**: Connect your GitHub repository containing the `Dockerfile` and select **Web Service** deployment. Render will build and host the Docker container on a free tier.
- **Cloud Run**: Deploy the Docker container directly to Google Cloud. It scales to zero when not in use, making it extremely cost-effective.
