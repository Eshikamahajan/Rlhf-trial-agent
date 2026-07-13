### Learnings

- As soon as a state enters, update_q_table function pre-enters the state and it is intialized to [0,0,0,...] in the q table and post every execution it gets modified based on the user feedback. 

- cold-start RL in production is a bad idea
- Better idea 
    - RL + instruction based in a sandbox, let user send feedback, pre-fix a q table from that learning
    - pass that learned and mature Q table with instructions to the production. 
- instruction based may or may not involve llm calls. here simple If else based on condition severity

- Whenever a q table has same weights across states, it picks up a random state by exploration (20%) through epsilon greedy method. 

- Here if the severity was already known i.e. moderate it shouldnt have picked up check_status_tool, instead ask_more_questions

- issue=order_delay_severity=moderate_user=regular - A state
    - Here, it could also pick ask_more_question , based on future runs, epsilon greedy method, it will pick that tool too and we can give our scoring to update the q-table. that's the benefit of epsilon greedy or random exploration. 