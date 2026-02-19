# PW-II

steps to run

# Step 1 – train the base classifier (saves model_weights.pt)
python train_classifier.py --epochs 15

# Step 2 – train the RL defense agent (needs model_weights.pt)
python train_rl_defense.py --classifier_weights model_weights.pt --epochs 30

# Step 3 – evaluate (needs both model_weights.pt and rl_agent_final.pth)
python evaluate.py --classifier_weights model_weights.pt --rl_agent_path rl_defense_out/rl_agent_final.pth