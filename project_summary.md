This project focuses on developing and evaluating a Reinforcement Learning (RL) based dynamic defense mechanism against adversarial attacks on image classifiers. The core idea is to train an RL agent to select an appropriate defensive image transformation on-the-fly, based on the characteristics of an input image, to maintain the classifier's accuracy against adversarial perturbations.

**I. Core Classifier (`models/base_classifier.py`)**

*   **`GhostModule`**: A convolutional block to generate more feature maps with fewer parameters.
*   **`ChannelAttention` & `SpatialAttention`**: Components of a CBAM (Convolutional Block Attention Module) to apply attention mechanisms in both channel and spatial dimensions.
*   **`CBAM`**: Combines channel and spatial attention modules.
*   **`EfficientNetB0_GCBAM`**: The main image classification model. It uses an EfficientNetB0 backbone, enhanced with a `GhostModule` and a `CBAM` before its final classification layers. Can be initialized with or without ImageNet pre-trained weights.
*   **`HFFetalPlanesDataset`**: A custom `torch.utils.data.Dataset` for the `qingyuyang/Fetal_Planes_DB` dataset from HuggingFace. It handles image loading, grayscale conversion, resizing, and normalization to `[0, 1]`.
*   **`load_fetal_planes_hf`**: Function to load and split the Fetal Planes dataset into training, validation, and test sets.
*   **`train_model`**: A utility function that encapsulates the training loop for the classifier, including Adam optimizer, Cross-Entropy loss, learning rate scheduling (ReduceLROnPlateau), and checkpointing/resumption capabilities.

**II. Adversarial Attacks (`attacks/fgsm.py`, `attacks/bim.py`, `attacks/pgd.py`)**

These three scripts are highly similar, serving as templates for adversarially training the `EfficientNetB0_GCBAM` classifier using different `torchattacks` methods. They share the model architecture, dataset handling, and training infrastructure, differing primarily in the specific attack algorithm and its parameters:

*   **`attacks/fgsm.py`**: Implements **FGSM (Fast Gradient Sign Method)** adversarial training. It's a single-step attack, characterized by the `eps` parameter. Training on clean examples for Epoch 0, then on FGSM-perturbed examples.
*   **`attacks/bim.py`**: Implements **BIM (Basic Iterative Method)** adversarial training. It's an iterative attack, similar to PGD but without a random start. Uses `eps`, `alpha`, and `steps`.
*   **`attacks/pgd.py`**: Implements **PGD (Projected Gradient Descent)** adversarial training. This is an iterative attack with a crucial `random_start` feature, making it a stronger attack. Uses `eps`, `alpha`, `steps`, and `random_start`.
*   **Common Logic**:
    *   Load dataset, initialize `EfficientNetB0_GCBAM`.
    *   Use `torchattacks` for generating adversarial examples.
    *   Training loop involves generating adversarial examples (after Epoch 0 for clean training) and training the model on these perturbed images.
    *   Includes mixed-precision training (AMP), checkpointing, and TensorBoard logging.
    *   Evaluates overall and per-class accuracy on a test set at the end.

**III. Defense Mechanisms (`defense/entropy.py`, `defense/rl_defense.py`)**

*   **`defense/entropy.py`**:
    *   **Purpose**: Extracts `STATE_DIM=12` features from an image batch to describe its characteristics, particularly in the context of adversarial perturbations. These features form the state observed by the RL agent.
    *   **Features (e.g.)**: Pixel entropy, classifier confidence, prediction entropy, gradient magnitude, Laplacian noise, Sobel edge energy, local spatial variance, top-2 softmax margin, top-3 cumulative probability, global pixel standard deviation, L∞ of Laplacian, DCT high/low frequency energy ratio.
    *   **`compute_state_features_batch`**: Efficiently computes these features for a batch, leveraging GPU operations and a single backward pass for gradient computation.
*   **`defense/rl_defense.py`**:
    *   **`DEFENSE_NAMES`**: A list of 10 discrete defense actions (e.g., "none", "gaussian_blur", "jpeg_75", "median_blur", "bit_depth_4", "randomized_smoothing", "adaptive_jpeg").
    *   **Primitive Defense Functions**: Implementations for each defense, largely GPU-native where possible (blur, median, bit-depth), with CPU-bound operations (JPEG) parallelized using `ThreadPoolExecutor`.
    *   **`apply_defense_batch`**: The main function for the RL agent, taking a batch of images and a list of actions (one per image). It efficiently applies the chosen defenses by grouping GPU-native actions and parallelizing CPU-bound ones.
    *   **Named Batch Functions**: Backward-compatible wrappers for static defenses, used by `evaluate.py`.

**V. Reinforcement Learning Components (`rl/dqn_agent.py`, `rl/replay_buffer.py`, `rl/train_rl.py`)**

*   **`rl/replay_buffer.py`**:
    *   **`ReplayBuffer`**: A basic, fixed-size circular experience replay buffer storing (state, action, reward, next_state, done) transitions for random mini-batch sampling. (Note: The `DQNAgent` primarily uses `PrioritizedReplayBuffer` from its own definition).
*   **`rl/dqn_agent.py`**:
    *   **`PrioritizedReplayBuffer`**: An advanced replay buffer that samples experiences based on their temporal difference (TD) error magnitude, using importance-sampling weights.
    *   **`DuelingDQN`**: The neural network architecture for the Q-function, separating value (`V(s)`) and advantage (`A(s,a)`) streams for more stable learning.
    *   **`DQNAgent`**: The core DQN agent.
        *   Uses two `DuelingDQN` networks: `online_net` (for action selection and learning) and `target_net` (for stable targets).
        *   Implements Double DQN for reduced overestimation bias.
        *   Employs a cosine-annealed epsilon decay schedule for exploration-exploitation balance.
        *   Uses Adam optimizer with gradient clipping and a cosine annealing learning rate scheduler.
        *   **`update()`**: Performs one learning step: samples from PER, computes Double DQN targets, calculates weighted Huber loss, backpropagates, updates network, and updates PER priorities.
        *   Provides `select_action` and `select_action_batch` for epsilon-greedy action selection.
        *   Includes `save` and `load` for persistence.
*   **`rl/train_rl.py`**:
    *   **`RLDefenseTrainer`**: Orchestrates the training of the `DQNAgent`.
    *   **Key Training Improvements**:
        *   **Multi-attack Batching**: Splits input batches and applies different attacks (FGSM, BIM, PGD) to segments, providing diverse adversarial examples in each step.
        *   **Warm-up Phase**: Starts with only FGSM attacks for a set number of batches to allow the agent to learn basic strategies before facing stronger attacks.
        *   **Continuous Softmax-based Reward**: Calculates reward as `2 * P(correct_class) - 1`, giving a dense signal based on the classifier's confidence in the ground-truth class on the defended image.
    *   **Training Workflow**:
        *   Loads the *frozen* classifier and initializes attacks.
        *   Initializes the `DQNAgent`.
        *   The `_train_one_epoch` method:
            *   Generates adversarial examples (using multi-attack batching and warm-up).
            *   Computes state features (`defense/entropy.py`).
            *   Agent selects actions (`rl/dqn_agent.py`).
            *   Applies defenses (`defense/rl_defense.py`).
            *   Classifier predicts, and rewards are calculated.
            *   Transitions are stored, and the agent is updated.
            *   Logs metrics to TensorBoard.
        *   The `validate` method performs greedy evaluation on the validation set across all attack types, logging accuracy and defense frequencies.
        *   Saves `DQNAgent` checkpoints.

**VI. Evaluation (`evaluate.py`)**

*   **`FullEvaluator`**: The central class for comprehensive evaluation.
    *   Loads the trained `EfficientNetB0_GCBAM` classifier and the trained `DQNAgent`.
    *   Defines `torchattacks` for FGSM, BIM, and PGD.
    *   **Evaluation Loop**: Iterates through different attack conditions ("clean", "fgsm", "bim", "pgd") and all defense mechanisms (static: "none", "gaussian", "jpeg", etc., and dynamic: "rl").
    *   **RL Defense Application**: For the "rl" defense, it computes state features, uses the `DQNAgent` to select an action, and applies the defense.
    *   **Metrics**: Calculates overall accuracy, per-class accuracy, confusion matrices, RL defense selection frequency, McNemar statistical significance tests, and perturbation analysis (L2/Linf norms).
    *   **Plotting**: Generates a wide array of `matplotlib` plots (heatmaps, bar charts, confusion matrices, attention maps, training curves from TensorBoard logs).
    *   Saves a `summary_results.json` with all key metrics.

---

**Overall Project Flow:**

1.  **`train_classifier.py`**: Trains the base `EfficientNetB0_GCBAM` classifier on clean images.
2.  **(Optional, but recommended for robustness) `attacks/pgd.py` (or `fgsm.py`, `bim.py`)**: Adversarially trains the classifier using PGD (or FGSM/BIM) to make it more robust. The resulting weights are typically used as the `classifier_weights` for RL training.
3.  **`train_rl_defense.py`**: Trains the `DQNAgent`. This agent learns to select optimal defense transformations by interacting with an environment where it's presented with adversarially attacked images (generated using multi-attack batching and a warm-up phase). It receives continuous rewards based on the classifier's performance on the defended images.
4.  **`evaluate.py`**: Compares the performance of the trained RL defense against various static defenses and against different adversarial attacks. It generates detailed metrics and visualizations to assess the effectiveness of the dynamic defense strategy.

The project demonstrates a comprehensive approach to adversarial robustness by combining a strong base classifier, various adversarial attacks for training, a rich set of defense mechanisms, and an advanced reinforcement learning agent to intelligently apply these defenses.