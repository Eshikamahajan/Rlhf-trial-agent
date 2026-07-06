# RLHF Healthcare Agent

## 1. Use Case

- The agent receives patient symptoms.
- Decides what action to take next.
- Receives feedback after each action.
- Learns from this feedback using a Q-table.
- The goal is to improve future triage decisions.


## 2. Actions And State

### Actions

- `ask_more_questions`: Ask for more symptoms.
- `call_urgency_tool`: Use Gemini to classify urgency.
- `call_care_tool`: Suggest care based on urgency.
- `refer_doctor`: Refer the patient to a doctor.

### State

- `symptoms`: Current list of patient symptoms.
- `urgency`: Current urgency level.
- `state_key`: Simple state label used by the Q-table.
- `action`: Last action selected by the agent.
- `output`: Text result from the action.
- `turn`: Current turn number.
- `q_table`: Stored action values.
- `reward`: Human feedback score.

### Graph Flow

```text
classify_state
  -> choose_action
  -> act
  -> human_feedback
  -> update_q
  -> continue or end
```

## 3. Important Definitions

- `Q-table`: A table that stores how useful each action is for each state.
- `reward`: Feedback from the user.
- `ALPHA`: Learning rate. It controls how strongly new feedback changes the Q-table.
- `GAMMA`: Discount factor. It controls how much future reward matters.
- `EPSILON`: Exploration rate. It controls how often the agent tries a random action.
- `MAX_TURNS`: Maximum number of turns in one episode.
- `state_key`: A compact name for the current situation, such as `symptoms=2_urgency=unknown`.
- `epsilon-greedy`: The strategy used to choose actions. The agent usually picks the best known action, but sometimes explores randomly.

## 5. Sample Execution

Example interaction:

```text
GOOGLE_API_KEY loaded successfully.

[Turn 0] State: symptoms=2_urgency=unknown -> Agent picks action: call_urgency_tool
  -> Tool 1 result -> urgency classified as: medium
  Rate this action: (1 = good, 0 = neutral, -1 = bad/unsafe)
  Clinician feedback: 1

[Turn 1] State: symptoms=2_urgency=medium -> Agent picks action: call_care_tool
  -> Tool 2 result -> Recommend a same-week doctor visit.
  Rate this action: (1 = good, 0 = neutral, -1 = bad/unsafe)
  Clinician feedback: 1

--- Episode finished ---
Final Q-table (saved to q_table.json):
```

The Q-table is updated after each feedback score.
The saved values are reused in the next run.
