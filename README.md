# Emergent_Language
Cognitive and Reasoning (2023 Fall) course project by Core123.
## Group Members
Weinan Qian, Feiyang Xie, Ningxin Pan, Haochen Zhao
## Model pipeline
train0(Env info --agent0 encode--> message0 --agent1 decoder--> state rep --table lookup)train1(--agent1 encoder--> message1 --agent0 decoder--> agent0 action --> Env reward)  
(split into train0 and train1 because lookup table truncates gradient)
## Current setting:
1. state: multidiscrete([3,3,3]), message is out of env setting
2. single round communication
3. message length 1, vocab size > #state, expecting to learn (state,char) pairs