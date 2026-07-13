# Delivery App Use Case README

## Overview
This project demonstrates a small RLHF-inspired support agent for a delivery-app use case. The agent handles support tickets such as delayed orders, minor app glitches, payment issues, and backend/API failures.

The workflow combines:
- a simple rule-based decision policy,
- a Q-table for reinforcement learning style updates,
- a small LangGraph workflow,
- and a human feedback loop for support-agent evaluation.

## Main Files
- Delivery_app_usecase.py: the main agent implementation, including the state graph, tools, and RLHF loop.
- delivery_q_table.json: stores learned Q-values between runs.
- rlhf-venv: the Python virtual environment used for this project.

## What the Agent Does
The agent can choose from four actions:
- ask_more_questions
- check_status_tool
- offer_resolution_tool
- escalate_to_support

It classifies the issue severity and then decides whether to ask follow-up questions, provide a resolution, or escalate to a human agent.

## How the Workflow Works
1. A ticket enters with basic issue signals such as issue type, delay, and user type.
2. The agent builds a state key and chooses an action.
3. The selected action executes a tool or decision step.
4. A human support lead provides feedback (1, 0, or -1).
5. The Q-table is updated based on that reward.

## Setup
1. Activate the virtual environment:
   ```bash
   source rlhf-venv/bin/activate
   ```
2. Create a .env file in the project root with your Google API key:
   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```

## Run the Example
```bash
python Delivery_app_usecase.py
```

During execution, the script will prompt for support feedback. Enter:
- 1 for a good action,
- 0 for neutral,
- -1 for a bad or unhelpful action.

## Customization Ideas
You can experiment by changing the initial ticket values in the main block of Delivery_app_usecase.py, such as:
- issue_type
- delay_minutes
- user_type

This helps test how the agent behaves for different support scenarios.
