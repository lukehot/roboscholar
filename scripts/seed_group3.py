#!/usr/bin/env python3
"""Seed content for papers group 3 (World Models + Cross-Embodiment)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_admin_client

PAPERS = [
    # ------------------------------------------------------------------
    # #32  DayDreamer
    # ------------------------------------------------------------------
    {
        "number": 32,
        "summary": (
            "DayDreamer applies the Dreamer world model framework to real physical robots "
            "including a quadruped (A1) and a robotic arm, demonstrating that learned latent "
            "world models are sample-efficient enough for real-hardware learning. By training "
            "a latent dynamics model from limited real-world interaction and then optimizing "
            "policies via imagined rollouts, DayDreamer achieves locomotion and manipulation "
            "behaviors within one hour of real-world data collection."
        ),
        "key_learnings": (
            "- World models trained on real robot data can generate sufficiently accurate imagined rollouts to train effective policies, closing the sim-to-real gap from the opposite direction\n"
            "- The Dreamer framework's sample efficiency (learning from ~1 hour of interaction) makes world model learning practical for physical hardware where data is expensive\n"
            "- Latent dynamics models avoid the computational cost of pixel-level prediction while retaining enough information for control\n"
            "- The approach works across morphologies (legged locomotion and arm manipulation) without fundamental architectural changes, suggesting generality of the world model paradigm\n"
            "- Online fine-tuning of the world model during deployment allows the policy to adapt to distribution shifts caused by the robot's own improving behavior"
        ),
        "reading_guide": (
            "Start with Section 1 (Introduction) for motivation on why world models matter for real robots. "
            "Read Section 3 (Method) carefully, focusing on how Dreamer's RSSM latent model is adapted for "
            "real-world deployment. Section 4 (Experiments) is essential -- pay special attention to the "
            "learning curves showing sample efficiency versus model-free baselines. Section 4.3 (ablations) "
            "reveals what components of the world model matter most. Skim Section 2 (Background) if you "
            "already know Dreamer/RSSM."
        ),
        "questions": [
            {
                "question": "DayDreamer learns locomotion in roughly one hour of real-world data. What is the key architectural property of the RSSM world model that enables this sample efficiency?",
                "choices": [
                    "The RSSM uses a recurrent state that combines deterministic and stochastic components, enabling accurate multi-step predictions from limited data",
                    "The RSSM pre-trains on simulation data and fine-tunes on real data via domain randomization",
                    "The RSSM uses a very large Transformer backbone that memorizes the dynamics from few examples",
                    "The RSSM operates directly in pixel space, giving it maximum information for dynamics prediction",
                ],
                "correct_index": 0,
                "explanation": "The Recurrent State-Space Model (RSSM) splits its latent state into a deterministic path (GRU) for long-horizon memory and a stochastic path for capturing uncertainty. This hybrid representation allows accurate multi-step imagined rollouts from relatively little data, because the deterministic component provides stable predictions while the stochastic component prevents overconfident extrapolation.",
                "why_it_matters": "Understanding the RSSM architecture is fundamental to evaluating any Dreamer-family paper. The deterministic-stochastic split is not just a design detail -- it directly determines how far ahead the model can reliably imagine, which in turn determines how much real data you need. Competitors like TDMPC2 made different choices (pure latent learning with TD targets) precisely because of RSSM's limitations at longer horizons.",
            },
            {
                "question": "DayDreamer trains policies entirely from imagined rollouts inside the world model. What is the primary failure mode that limits this 'dreaming' approach as tasks become more complex?",
                "choices": [
                    "The policy overfits to artifacts in the world model's predictions, exploiting inaccuracies rather than learning real dynamics",
                    "Imagined rollouts are too slow computationally to scale beyond simple tasks",
                    "The world model cannot represent contact-rich interactions at all",
                    "The approach requires reward shaping that becomes intractable for complex tasks",
                ],
                "correct_index": 0,
                "explanation": "When policies are optimized purely inside a learned model, they can exploit model inaccuracies -- finding 'adversarial' action sequences that achieve high reward in the model but fail on the real robot. This model exploitation problem worsens as tasks require longer horizons or more precise dynamics, since small prediction errors compound over time.",
                "why_it_matters": "Model exploitation is the central challenge of all world-model-based policy learning. It explains why DayDreamer works well on relatively short-horizon tasks but why more complex manipulation requires either shorter imagination horizons, model-predictive control (like TDMPC2), or periodic re-collection of real data. Recognizing this failure mode helps you evaluate claims from any world model paper.",
            },
            {
                "question": "DayDreamer updates the world model online as new real data arrives during deployment. Why is this online update critical rather than training a fixed world model once?",
                "choices": [
                    "Online updates are needed because the robot hardware degrades over time",
                    "The policy's improving behavior changes the data distribution, and the world model must track these non-stationary dynamics to remain accurate",
                    "Online updates are only needed to handle changes in lighting conditions",
                    "A fixed world model would run out of memory for storing imagined trajectories",
                ],
                "correct_index": 1,
                "explanation": "As the policy improves, it visits different states than the initial random exploration -- a form of distribution shift. A world model trained only on early exploratory data becomes inaccurate in the regions the improved policy actually visits. Online updates ensure the model remains accurate in policy-relevant state regions, preventing the model exploitation problem from worsening.",
                "why_it_matters": "This non-stationarity problem is unique to world-model-based RL and does not arise in imitation learning. It connects to broader concepts in off-policy learning and distribution shift. Understanding it explains why DayDreamer alternates between real data collection and imagination, and why purely offline world model training (as in some video prediction approaches) faces different challenges.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #33  TDMPC2
    # ------------------------------------------------------------------
    {
        "number": 33,
        "summary": (
            "TD-MPC2 presents a scalable framework for continuous control that jointly learns a latent "
            "world model, a value function, and a policy, then uses model-predictive control (MPC) at "
            "test time to plan through the learned model. It scales to a single generalist agent across "
            "80 diverse tasks spanning multiple domains without task-specific tuning, demonstrating that "
            "world model planning can match or exceed model-free methods at scale."
        ),
        "key_learnings": (
            "- Combining temporal-difference learning for value estimation with model-predictive control for planning yields agents that are both sample-efficient (from the model) and performant (from planning)\n"
            "- Learning entirely in latent space avoids pixel-level reconstruction losses, enabling the model to focus on dynamics-relevant features rather than visual details\n"
            "- A single set of hyperparameters and architecture can work across 80 tasks spanning locomotion, manipulation, and more, challenging the assumption that world model methods need per-task tuning\n"
            "- The MPC planning at test time provides a form of implicit ensembling and error correction that mitigates model exploitation\n"
            "- Scaling model capacity (up to 317M parameters) consistently improves multi-task performance, suggesting world model approaches have favorable scaling properties for control"
        ),
        "reading_guide": (
            "Begin with Section 1 to understand the motivation of combining TD learning with MPC. "
            "Section 3 (Method) is the core -- focus on the five loss functions (consistency, reward, "
            "value, priority, policy) and how they interact. Section 3.3 on the MPPI planning procedure "
            "is critical for understanding test-time behavior. Section 4 scaling experiments are the main "
            "empirical contribution. Appendix A has important architecture details often missed in the "
            "main text."
        ),
        "questions": [
            {
                "question": "TD-MPC2 learns its world model entirely in latent space without a reconstruction loss. What is the key advantage of this over approaches like Dreamer that include an observation reconstruction objective?",
                "choices": [
                    "It avoids the need for a visual encoder entirely",
                    "The latent space can learn to discard visually complex but dynamically irrelevant information, focusing model capacity on control-relevant features",
                    "Reconstruction losses are computationally infeasible for continuous control",
                    "It allows the model to handle only proprioceptive inputs, not images",
                ],
                "correct_index": 1,
                "explanation": "Reconstruction losses force the world model to allocate capacity to predicting every visual detail -- shadows, textures, backgrounds -- even though most pixels are irrelevant for control. By dropping reconstruction and using only dynamics-relevant losses (consistency, reward, value), TD-MPC2's latent space can focus entirely on information that matters for decision-making.",
                "why_it_matters": "The 'what should the world model learn to predict?' question is one of the deepest design choices in model-based RL. Dreamer-style reconstruction, reward-only prediction, contrastive objectives, and TD-MPC2's consistency losses each encode different assumptions about what information matters. Your choice here determines whether the model wastes capacity on irrelevant visual complexity or ignores task-relevant structure.",
            },
            {
                "question": "TD-MPC2 uses MPPI (Model Predictive Path Integral) planning at test time rather than simply executing the learned policy directly. What fundamental benefit does this planning provide?",
                "choices": [
                    "MPPI is faster to execute than a neural network forward pass",
                    "MPPI allows the agent to consider multiple action sequences and select among them using the value function, providing error correction beyond what the policy alone achieves",
                    "MPPI enables the agent to handle discrete action spaces that the policy cannot represent",
                    "MPPI eliminates the need for a learned value function entirely",
                ],
                "correct_index": 1,
                "explanation": "MPPI samples multiple candidate action sequences, rolls them out through the world model, evaluates them with the learned value function, and selects the best. This look-ahead planning corrects errors in the policy network by explicitly evaluating consequences -- like having both fast intuition (the policy) and deliberate reasoning (the planning).",
                "why_it_matters": "The interplay between amortized policy execution and test-time planning is a deep theme in decision-making. Pure policy execution (model-free) is fast but myopic. Pure planning (classic MPC) is deliberate but slow. TD-MPC2's hybrid -- using the policy to warm-start MPPI -- represents a principled middle ground. Understanding this trade-off is essential for evaluating when world model planning is worth its computational overhead.",
            },
            {
                "question": "TD-MPC2 scales a single agent to 80 tasks without per-task hyperparameter tuning. What is the most significant architectural change that enables this multi-task generalization?",
                "choices": [
                    "Using a task-conditioned normalization scheme (SimNorm) that replaces standard layer normalization across the model",
                    "Training separate world models for each task domain and selecting at inference time",
                    "Using a mixture-of-experts architecture with task-specific routing",
                    "Pre-training on internet-scale video data before fine-tuning on control tasks",
                ],
                "correct_index": 0,
                "explanation": "TD-MPC2 introduces SimNorm (simplicial normalization using softmax over feature groups) to replace standard normalization layers. This stabilizes training across tasks with vastly different observation/action scales and dynamics. Without it, a single set of hyperparameters fails because different tasks produce features at incompatible scales.",
                "why_it_matters": "Normalization might seem like a minor architectural detail, but in multi-task settings it becomes critical. Different tasks produce features, rewards, and gradients at wildly different scales. SimNorm's ability to handle this without per-task tuning is what makes the 'one agent, 80 tasks' result possible. This connects to broader challenges in multi-task optimization and loss balancing that arise in any generalist model.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #34  UniSim
    # ------------------------------------------------------------------
    {
        "number": 34,
        "summary": (
            "UniSim proposes using large-scale video generation models as universal simulators for "
            "training robot policies. By fine-tuning a pre-trained video diffusion model on action-conditioned "
            "robot data, UniSim can generate realistic future observations conditioned on robot actions, "
            "effectively serving as a learned physics simulator. Policies trained entirely within UniSim's "
            "generated simulations transfer to the real world."
        ),
        "key_learnings": (
            "- Video generation models pre-trained on internet-scale data capture rich physical priors (object permanence, gravity, contact) that can be fine-tuned for robot simulation\n"
            "- Action conditioning transforms a passive video generator into an interactive simulator: given the current frame and a robot action, the model predicts the next observation\n"
            "- The approach sidesteps the sim-to-real gap by generating photorealistic observations rather than requiring hand-engineered simulation assets\n"
            "- UniSim can simulate diverse scenarios including novel objects and environments not seen in the robot training data, leveraging the pre-trained model's visual knowledge\n"
            "- The fidelity-diversity trade-off in video generation directly impacts policy quality: too little fidelity causes policy failure, while too little diversity causes overfitting"
        ),
        "reading_guide": (
            "Read Section 1 for the core insight about video models as simulators. Section 3 (Method) "
            "is essential -- focus on how action conditioning is injected into the video diffusion model "
            "and the two-stage training (internet pre-training then robot fine-tuning). Section 4.2 on "
            "policy learning within UniSim shows the complete pipeline. Section 5 (limitations) is "
            "unusually important -- understand where the generated simulations break down."
        ),
        "questions": [
            {
                "question": "UniSim fine-tunes a pre-trained video generation model with action conditioning for robot simulation. What is the fundamental advantage over training an action-conditioned video model from scratch on robot data alone?",
                "choices": [
                    "Pre-trained video models generate frames faster than models trained from scratch",
                    "Pre-training captures visual priors (lighting, physics, object appearance) from internet-scale data that are impossible to learn from limited robot datasets alone",
                    "Pre-trained video models natively understand robot action spaces without fine-tuning",
                    "Training from scratch requires simulation data, creating a circular dependency",
                ],
                "correct_index": 1,
                "explanation": "Robot datasets are tiny compared to internet video. A model trained from scratch on robot data alone would produce blurry, unrealistic frames and fail to generalize to new objects or environments. Pre-training on internet video teaches the model how the visual world works -- realistic rendering, physics intuitions, object diversity -- which then transfers when action conditioning is added.",
                "why_it_matters": "This is the core hypothesis behind video-model-as-simulator approaches (UniSim, IRASim, Cosmos). If internet video pre-training does not transfer meaningful physics priors to robotics, the entire paradigm fails. Understanding what transfers (visual appearance, rough physics) and what does not (precise contact dynamics, force feedback) determines where these approaches succeed and where traditional simulators remain necessary.",
            },
            {
                "question": "Policies trained inside UniSim's generated simulations must transfer to the real world. What is the primary source of transfer failure that distinguishes this from the traditional sim-to-real gap?",
                "choices": [
                    "UniSim's visual domain is too different from real cameras",
                    "Compounding generation errors in long-horizon rollouts produce physically implausible states that the real robot never encounters",
                    "The action space in UniSim does not match real robot actuators",
                    "UniSim cannot generate observations at real-time frame rates",
                ],
                "correct_index": 1,
                "explanation": "Unlike traditional simulators that maintain exact physical state, UniSim generates observations autoregressively -- each frame is conditioned on previously generated frames. Small generation errors compound over long horizons, producing observations that drift from physical plausibility. The policy then trains on states it will never encounter in reality.",
                "why_it_matters": "Compounding errors in autoregressive generation is the fundamental limitation of all video-model-as-simulator approaches. Traditional simulators do not have this problem because physics engines maintain consistent state. This error accumulation explains why these approaches work better for short-horizon tasks and why methods like DIAMOND use diffusion to reduce per-step error. Understanding this limitation is essential for evaluating the entire 'generative world model' paradigm.",
            },
            {
                "question": "UniSim generates visual observations but not proprioceptive feedback (joint torques, forces). Why does this limitation fundamentally constrain the types of tasks these video-model simulators can handle?",
                "choices": [
                    "Most robot tasks do not require proprioceptive feedback at all",
                    "Contact-rich manipulation tasks require force/torque sensing for stable grasping and insertion, which cannot be inferred from visual observation alone",
                    "Proprioceptive feedback is only needed for locomotion, not manipulation",
                    "Adding proprioceptive channels would simply require a larger video model",
                ],
                "correct_index": 1,
                "explanation": "Tasks like peg insertion, gear meshing, or deformable object manipulation depend on force feedback that is invisible in images. A video model can show whether a peg entered a hole but cannot provide the force signals needed to learn compliant insertion policies. This restricts video-model simulators to visually-observable tasks where success can be determined from pixels.",
                "why_it_matters": "This reveals a fundamental modality gap in the video-as-simulator paradigm. Real robots sense forces, torques, and contacts that have no visual manifestation. PhysDreamer and Genesis attempt to address this by incorporating physics priors, but the question of whether video generation can ever replace full physical simulation remains open. This limitation shapes which applications are suitable for which simulation approach.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #36  DIAMOND
    # ------------------------------------------------------------------
    {
        "number": 36,
        "summary": (
            "DIAMOND (Diffusion for World Modeling) replaces traditional deterministic or VAE-based "
            "dynamics models with a diffusion model that predicts future observation frames. By leveraging "
            "the high-fidelity generation capabilities of diffusion models, DIAMOND achieves state-of-the-art "
            "world model quality on Atari benchmarks, producing visually coherent long-horizon rollouts "
            "that enable training competitive RL agents entirely in imagination."
        ),
        "key_learnings": (
            "- Diffusion models' iterative denoising naturally captures multi-modal transition distributions, handling stochastic environments better than single-step prediction models\n"
            "- Higher visual fidelity in world model predictions directly translates to better policy performance, as the policy receives less corrupted training signal\n"
            "- DIAMOND operates in observation space rather than latent space, trading computational cost for interpretability and avoiding representation learning pitfalls\n"
            "- The denoising schedule and number of diffusion steps create a direct trade-off between generation quality and simulation speed\n"
            "- DIAMOND achieves human-level performance on several Atari games when training agents entirely within the diffusion world model"
        ),
        "reading_guide": (
            "Section 1 motivates why diffusion models are a natural fit for world modeling. Section 3 "
            "is the method core -- focus on how actions condition the denoising process and how frames "
            "are generated autoregressively. Section 4 compares against prior world models (IRIS, TWM) "
            "and shows the visual quality advantage. Read Section 4.3 on the relationship between "
            "generation quality and agent performance carefully -- this is the key empirical insight."
        ),
        "questions": [
            {
                "question": "DIAMOND uses diffusion models for dynamics prediction instead of discrete tokenization (like IRIS) or VAE-based latent models (like Dreamer). What is the core advantage of diffusion for world modeling specifically?",
                "choices": [
                    "Diffusion models are faster to train than VAEs or discrete tokenizers",
                    "The iterative denoising process naturally represents multi-modal transition distributions and avoids the blurriness of single-step regression models",
                    "Diffusion models do not require action conditioning",
                    "Diffusion models can generate observations at arbitrary resolutions without retraining",
                ],
                "correct_index": 1,
                "explanation": "In stochastic environments, the next observation given an action is not deterministic -- multiple futures are possible. Single-step regression models (MSE loss) average over these modes, producing blurry predictions. Diffusion models naturally represent the full distribution by learning to denoise from pure noise, capturing each mode with high fidelity through the iterative sampling process.",
                "why_it_matters": "The choice of generative model for world modeling is a fundamental design decision. VAEs tend to produce blurry outputs, discrete tokens lose fine detail, and deterministic models cannot handle stochasticity. Diffusion offers high fidelity but at significant computational cost. Understanding these trade-offs explains why different papers make different choices and helps you evaluate whether a given approach is appropriate for a given task.",
            },
            {
                "question": "DIAMOND operates in observation space (predicting pixels) rather than latent space. What does this architectural choice sacrifice compared to latent-space world models like Dreamer or TD-MPC2?",
                "choices": [
                    "It cannot handle stochastic environments",
                    "Computational cost scales with observation dimensionality, making high-resolution or long-horizon simulation expensive",
                    "It cannot be conditioned on actions",
                    "The predicted observations cannot be used for policy gradient updates",
                ],
                "correct_index": 1,
                "explanation": "Predicting full observations (e.g., 64x64 or 84x84 frames) at each step is far more expensive than predicting compact latent vectors. Each diffusion step requires multiple denoising iterations over the full observation. For long-horizon rollouts or high-resolution inputs, this cost multiplies, making observation-space models orders of magnitude slower than latent-space alternatives.",
                "why_it_matters": "This is the fundamental speed-quality trade-off in world modeling. Latent-space models (Dreamer, TD-MPC2) are fast but may discard task-relevant visual information. Observation-space models (DIAMOND) preserve all visual detail but are slow. Understanding this trade-off explains why Dreamer dominates in settings requiring many imagination steps while DIAMOND excels where visual fidelity matters most.",
            },
            {
                "question": "DIAMOND shows that higher visual quality in world model predictions correlates with better downstream agent performance. Why is this relationship NOT obvious a priori?",
                "choices": [
                    "Higher visual quality always guarantees better policies in every setting",
                    "A world model could produce visually perfect predictions but have incorrect dynamics, or produce blurry predictions that still capture the correct state-action-reward structure",
                    "Visual quality has no connection to RL performance in any theoretical framework",
                    "Lower quality images actually train more robust policies through implicit data augmentation",
                ],
                "correct_index": 1,
                "explanation": "Visual quality (measured by FID, SSIM) and dynamics accuracy are not the same thing. A model could render beautiful frames with wrong physics, or produce blurry frames that correctly predict rewards and state transitions. DIAMOND's empirical finding that visual quality correlates with agent performance suggests that in practice, the same model capacity that produces sharp images also captures better dynamics -- but this is an empirical finding, not a guaranteed relationship.",
                "why_it_matters": "Conflating visual quality with dynamics accuracy is a common mistake when evaluating world models. Many papers report impressive FID scores or visual samples without demonstrating that these translate to better control. DIAMOND's contribution is showing this correlation empirically in Atari, but it may not hold in all domains -- especially where dynamics are simple but visuals are complex.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #37  Cosmos
    # ------------------------------------------------------------------
    {
        "number": 37,
        "summary": (
            "Cosmos is NVIDIA's World Foundation Model platform that trains large-scale video generation "
            "models designed to serve as world simulators for robotics and autonomous driving. Built on "
            "both autoregressive and diffusion-based Transformer architectures at scales up to 14B parameters, "
            "Cosmos is pre-trained on massive video datasets and provides a general-purpose world model "
            "backbone that downstream applications can fine-tune for specific simulation needs."
        ),
        "key_learnings": (
            "- Foundation model paradigm applied to world simulation: pre-train a massive video model once, fine-tune for specific robots, environments, or tasks\n"
            "- Cosmos offers both autoregressive (causal Transformer) and diffusion-based architectures, providing different speed-quality trade-offs for different applications\n"
            "- Video tokenization is critical: Cosmos develops specialized video tokenizers that compress spatiotemporal data into discrete or continuous token sequences\n"
            "- Scale matters for world models just as for language models: larger Cosmos models generate more physically consistent and longer-horizon predictions\n"
            "- The platform approach (pre-trained models + fine-tuning APIs) aims to democratize world model access, similar to how foundation LLMs democratized NLP"
        ),
        "reading_guide": (
            "Start with Section 1 for the vision of World Foundation Models. Section 3 on the video "
            "tokenizer is crucial -- it determines the representation that all downstream models operate on. "
            "Section 4 covers the two architectures (autoregressive and diffusion); compare their trade-offs. "
            "Section 5 on pre-training data curation is often overlooked but essential for understanding "
            "model quality. Section 7 (applications) shows downstream use in robotics and driving."
        ),
        "questions": [
            {
                "question": "Cosmos offers both autoregressive and diffusion-based world model architectures. In the context of robotics simulation, what is the key trade-off between these two paradigms?",
                "choices": [
                    "Autoregressive models only work for language, diffusion models only for images",
                    "Autoregressive models enable faster token-by-token generation with KV-caching but enforce a fixed causal ordering, while diffusion models capture richer multi-modal distributions but require iterative denoising",
                    "Diffusion models are always more accurate but cannot be conditioned on actions",
                    "Autoregressive models require more training data than diffusion models",
                ],
                "correct_index": 1,
                "explanation": "Autoregressive models generate tokens sequentially with KV-cache efficiency, but impose a fixed left-to-right ordering that may not match the spatial structure of video. Diffusion models generate all tokens simultaneously through iterative denoising, naturally handling multi-modal distributions, but each 'step' requires a full model forward pass. For robotics, this means autoregressive models are faster for real-time planning while diffusion models produce higher-quality predictions.",
                "why_it_matters": "This autoregressive vs. diffusion choice permeates the entire generative AI landscape. In robotics specifically, the trade-off maps to a fundamental question: do you need fast, sequential predictions for real-time MPC planning, or high-quality multi-modal predictions for offline policy training? Cosmos offering both architectures acknowledges that different downstream applications have different requirements.",
            },
            {
                "question": "Cosmos's video tokenizer compresses raw video into token sequences before world model training. Why is tokenizer quality arguably MORE important for world models than for language models?",
                "choices": [
                    "Video tokenizers are simply more complex to implement than text tokenizers",
                    "Information loss during video tokenization is irreversible -- physical details discarded by the tokenizer (small objects, subtle motions) can never be recovered by the world model, directly limiting simulation fidelity",
                    "Language tokenizers never lose information, so quality does not matter there",
                    "Video tokenizers determine the model's parameter count while text tokenizers do not",
                ],
                "correct_index": 1,
                "explanation": "Text tokenization is nearly lossless (BPE encoding preserves all characters). Video tokenization is fundamentally lossy -- compressing a 256x256x3 frame to a few hundred tokens necessarily discards information. If the tokenizer drops fine object details or subtle contact dynamics, no amount of world model capacity can recover them. This makes the tokenizer a hard information bottleneck.",
                "why_it_matters": "The video tokenizer sets an upper bound on world model quality that the generative model cannot exceed. This is why Cosmos devotes significant effort to tokenizer design. The same principle applies across all video-based world models (DIAMOND, UniSim, GR-2) -- if you cannot evaluate the tokenizer's reconstruction quality, you cannot evaluate the world model's potential. Always check tokenizer fidelity before model fidelity.",
            },
            {
                "question": "Cosmos positions itself as a 'foundation model for world simulation.' What is the strongest argument AGAINST this foundation model approach compared to task-specific world models?",
                "choices": [
                    "Foundation models are always worse than task-specific models",
                    "A general-purpose visual world model may allocate capacity to photorealistic rendering rather than dynamically-accurate physics prediction, since these objectives can conflict",
                    "Task-specific world models do not require any training data",
                    "Foundation world models cannot be fine-tuned for specific tasks",
                ],
                "correct_index": 1,
                "explanation": "A model trained to generate photorealistic video optimizes for visual quality -- plausible-looking frames. But physically accurate simulation requires correct dynamics -- objects should follow Newton's laws, not just look realistic. These objectives can conflict: a visually beautiful but physically inaccurate rollout may fool a human but will train a bad policy. Task-specific models can focus entirely on dynamics accuracy.",
                "why_it_matters": "This tension between visual realism and physical accuracy is the central open question for video-model-as-simulator approaches. It is why Genesis takes the opposite approach (physics engine first, rendering second) and why traditional simulators like MuJoCo remain dominant for tasks requiring precise dynamics. Understanding this trade-off helps you evaluate whether a given world model is actually suitable for policy training versus just good at generating pretty videos.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #38  Cosmos Policy
    # ------------------------------------------------------------------
    {
        "number": 38,
        "summary": (
            "Cosmos Policy builds on NVIDIA's Cosmos world foundation model to learn robot policies "
            "via imagination. It fine-tunes the pre-trained Cosmos world model with action conditioning, "
            "then trains manipulation and navigation policies by generating imagined rollouts. The approach "
            "demonstrates that large-scale pre-trained world models can reduce the real-world data needed "
            "for policy learning by providing a rich generative prior for simulation."
        ),
        "key_learnings": (
            "- Pre-trained world foundation models can be adapted for policy learning by adding action-conditioning mechanisms during fine-tuning\n"
            "- Imagined rollouts from fine-tuned Cosmos models provide sufficient signal for training manipulation policies that transfer to real robots\n"
            "- The quality of the underlying world model (Cosmos) directly bounds the quality of policies trained within it\n"
            "- Action conditioning can be injected via cross-attention or concatenation with visual tokens, each with different trade-offs for generation quality and action controllability\n"
            "- The approach naturally inherits Cosmos's visual diversity, enabling policy training across environments not present in the robot fine-tuning data"
        ),
        "reading_guide": (
            "Read the introduction for the vision of policy learning through world model imagination. "
            "Focus on the action conditioning mechanism -- how robot actions are injected into the "
            "Cosmos generation process. The experimental evaluation comparing policies trained in Cosmos "
            "imagination vs. real data vs. traditional simulation is the key result. Pay attention to "
            "failure case analysis showing where the world model's limitations propagate to policy failures."
        ),
        "questions": [
            {
                "question": "Cosmos Policy fine-tunes a pre-trained video generation model with action conditioning for policy learning. What is the key challenge in adding action conditioning to a model pre-trained without actions?",
                "choices": [
                    "The pre-trained model's weights must be entirely retrained from scratch",
                    "Action conditioning must be integrated without disrupting the model's learned visual priors, requiring careful architectural injection points and fine-tuning strategies",
                    "Actions cannot be represented as tokens compatible with video generation models",
                    "The pre-trained model already implicitly conditions on actions, so explicit conditioning creates conflicts",
                ],
                "correct_index": 1,
                "explanation": "The pre-trained Cosmos model has learned rich visual priors from internet video. Naively adding action conditioning (e.g., concatenating action tokens) can disrupt these learned representations, causing the model to lose its generation quality. The challenge is injecting action influence in a way that steers generation without destroying the visual knowledge -- typically through lightweight adapters, cross-attention, or low-rank fine-tuning.",
                "why_it_matters": "This is the same challenge faced by all approaches that adapt pre-trained generative models for control (UniSim, IRASim, GR-2). The solution determines whether you retain the pre-training benefits or waste them. It connects to broader adapter/fine-tuning literature (LoRA, prompt tuning) and the fundamental tension between preserving pre-trained knowledge and learning new capabilities.",
            },
            {
                "question": "Cosmos Policy trains policies by imagining rollouts inside a learned world model rather than collecting real-world data. Compared to collecting equivalent amounts of real-world demonstrations, what is the hidden cost of this approach?",
                "choices": [
                    "Imagined rollouts are always cheaper than real-world data collection",
                    "The policy's performance ceiling is bounded by the world model's fidelity, and verifying that the world model is sufficiently accurate for a given task requires real-world evaluation anyway",
                    "Imagined rollouts cannot represent visual observations, only proprioceptive states",
                    "World model imagination only works for navigation, not manipulation",
                ],
                "correct_index": 1,
                "explanation": "While imagination is computationally cheap, you cannot know whether the imagined physics are accurate enough until you test on a real robot. A policy that achieves 95% success in imagination might achieve 10% in reality if the world model gets contact dynamics wrong. This means real-world evaluation is still required, and if the world model is inadequate for a task, you must collect real data anyway -- potentially after wasting compute on imagination training.",
                "why_it_matters": "This is the 'evaluation paradox' of world-model-based policy learning. The promise is reduced real-world data, but the verification still requires real-world experiments. Understanding this hidden cost prevents overestimating the practical benefits and helps design validation pipelines that catch world model inadequacies early.",
            },
            {
                "question": "Cosmos Policy inherits the visual diversity from Cosmos's internet video pre-training. Why might this inherited diversity be misleading for robotic policy learning?",
                "choices": [
                    "More visual diversity always helps policy learning",
                    "The visual diversity from internet video represents passive observation distributions, not the action-conditional distributions that robots encounter when actively manipulating objects",
                    "Visual diversity only matters for navigation tasks, not manipulation",
                    "Internet videos have lower resolution than robot camera feeds",
                ],
                "correct_index": 1,
                "explanation": "Internet videos show how scenes look, but robots interact with scenes -- pushing, grasping, deforming objects. The distribution of visual states a robot encounters through active manipulation (e.g., a half-inserted peg, a partially folded cloth) may be poorly represented in passive internet video. The model may generate visually diverse but dynamically implausible action-conditioned predictions for these interaction-heavy states.",
                "why_it_matters": "This passive-vs-active distribution mismatch is a deep limitation of all approaches that transfer from internet video to robotics. Internet video captures the world as observed, not as manipulated. Understanding this distinction prevents overconfidence in claims about 'zero-shot transfer' from video models and helps identify which tasks (visually-driven, low contact) benefit most from video pre-training.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #39  RoboDreamer
    # ------------------------------------------------------------------
    {
        "number": 39,
        "summary": (
            "RoboDreamer introduces compositional world models for robot imagination by decomposing "
            "scene understanding into object-centric representations. Instead of learning a monolithic "
            "dynamics model over the full observation, RoboDreamer models individual objects and their "
            "interactions, enabling compositional generalization to novel object configurations and "
            "tasks not seen during training."
        ),
        "key_learnings": (
            "- Compositional world models decompose scenes into objects and relations, enabling combinatorial generalization to new configurations\n"
            "- Object-centric representations allow the world model to re-compose known object dynamics into novel multi-object scenarios\n"
            "- Language grounding connects natural language instructions to specific object slots in the world model, enabling instruction-conditioned imagination\n"
            "- The compositional structure provides interpretability: you can inspect which objects the model attends to and how they interact\n"
            "- The approach trades monolithic model capacity for structured inductive bias, improving generalization at the cost of more complex training"
        ),
        "reading_guide": (
            "Focus on Section 3 for the compositional world model architecture -- how objects are "
            "discovered, represented, and how their interactions are modeled. The language grounding "
            "component (Section 3.2) connecting instructions to object slots is novel. Section 4 "
            "experiments should focus on compositional generalization results: does the model handle "
            "new object combinations not in training? Compare against monolithic baselines carefully."
        ),
        "questions": [
            {
                "question": "RoboDreamer uses object-centric decomposition for world modeling. Compared to a monolithic world model of equivalent capacity, what is the key advantage for robotic manipulation?",
                "choices": [
                    "Object-centric models require less training data overall",
                    "Compositional structure enables combinatorial generalization: dynamics learned for objects A and B individually can be composed to predict interactions with novel object C",
                    "Object-centric models always produce more visually realistic predictions",
                    "Monolithic models cannot handle multiple objects in a scene",
                ],
                "correct_index": 1,
                "explanation": "A monolithic model that has seen cups and plates separately must see them together to predict their interaction. An object-centric model learns per-object dynamics and interaction rules that can compose: it can predict a novel cup-plate interaction by combining the learned dynamics of each object type with a learned interaction model. This compositional generalization is combinatorially more data-efficient.",
                "why_it_matters": "Compositional generalization is one of the core unsolved challenges in robot learning. Robots encounter novel object combinations constantly -- every kitchen has a different set of items. Understanding whether a world model generalizes compositionally or merely interpolates within its training distribution is essential for predicting real-world deployment success.",
            },
            {
                "question": "RoboDreamer must discover and segment objects from raw observations without explicit object labels. What is the fundamental challenge this creates for world model accuracy?",
                "choices": [
                    "Object segmentation is computationally expensive but always accurate with modern vision models",
                    "Errors in object discovery propagate through the world model: if an object is missed or incorrectly segmented, its dynamics cannot be modeled, causing cascading prediction failures",
                    "Object discovery only works for rigid objects, not deformable ones",
                    "Object segmentation is only needed during training, not during inference",
                ],
                "correct_index": 1,
                "explanation": "Object-centric world models are only as good as their object discovery. If the segmentation misses a small but dynamically important object (e.g., a bolt being inserted), the world model has no slot for it and cannot predict its dynamics. These errors cascade: incorrect object representations lead to incorrect interaction predictions, compounding over time. This makes robust object discovery a prerequisite, not an afterthought.",
                "why_it_matters": "This reveals the hidden dependency in object-centric approaches: they add an object discovery pipeline that becomes a potential failure point. Monolithic models avoid this by operating on raw observations. The trade-off is between compositional generalization (object-centric) and robustness to discovery errors (monolithic). Understanding this trade-off is important for evaluating any structured world model approach.",
            },
            {
                "question": "RoboDreamer connects language instructions to object slots for instruction-conditioned imagination. Why is this language-object grounding architecturally simpler than language conditioning in monolithic world models?",
                "choices": [
                    "Language instructions do not need to be processed at all in object-centric models",
                    "In an object-centric model, language naturally maps to specific object slots (e.g., 'pick up the red cup' maps to the red-cup slot), while monolithic models must learn to ground language across the entire unstructured latent space",
                    "Language conditioning in monolithic models requires a separate language model, while object-centric models do not",
                    "Object-centric models only support simple one-word commands, not full sentences",
                ],
                "correct_index": 1,
                "explanation": "Language instructions typically refer to specific objects and their desired states ('move the block to the left of the plate'). In an object-centric model, this maps directly to operations on the relevant object slots -- the structure provides a natural alignment between language referents and model components. Monolithic models must learn this grounding implicitly across an unstructured latent space, which is harder and less interpretable.",
                "why_it_matters": "Language grounding in world models is essential for instruction-following robots. The object-centric approach offers a compelling advantage: the model's internal structure mirrors the compositional structure of language itself. This connects to broader research on compositional representations and neurosymbolic AI, where structured representations facilitate language-conditioned reasoning.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #40  IRASim
    # ------------------------------------------------------------------
    {
        "number": 40,
        "summary": (
            "IRASim (Interaction-Aware Simulation) generates future video frames conditioned on robot "
            "actions, focusing specifically on modeling the interactions between robot end-effectors and "
            "objects. Unlike generic video prediction models, IRASim explicitly models how robot actions "
            "cause physical changes in the scene, enabling more accurate action-conditioned simulation "
            "for manipulation policy training."
        ),
        "key_learnings": (
            "- Explicitly modeling robot-object interactions in the video generation process produces more accurate action-conditioned predictions than generic video generation\n"
            "- Interaction-aware conditioning injects robot action information at the spatial locations where the robot contacts objects, rather than globally\n"
            "- The approach bridges the gap between passive video prediction (what will happen next?) and interactive simulation (what will happen if I do this action?)\n"
            "- Training on robot manipulation datasets allows the model to learn contact dynamics, pushing, grasping, and other interaction patterns\n"
            "- Generated videos can serve as training data augmentation for downstream policy learning, reducing real-world data requirements"
        ),
        "reading_guide": (
            "Section 1 motivates why interaction-awareness matters for action-conditioned video generation. "
            "Section 3 on the interaction-aware conditioning mechanism is the key technical contribution -- "
            "understand how actions are spatially localized rather than globally conditioned. Section 4 "
            "compares against action-agnostic and globally-conditioned baselines. Focus on the qualitative "
            "results showing how IRASim correctly predicts object motion in response to robot actions."
        ),
        "questions": [
            {
                "question": "IRASim spatially localizes action conditioning at the robot-object interaction point rather than conditioning globally. What problem does this spatial localization solve?",
                "choices": [
                    "Global conditioning is computationally infeasible for video generation models",
                    "Global action conditioning allows the model to 'hallucinate' object motion far from the robot contact point, while spatial localization constrains the action's causal effect to the physically plausible interaction region",
                    "Spatial localization enables the model to handle multiple robot arms simultaneously",
                    "Global conditioning cannot represent continuous robot actions, only discrete ones",
                ],
                "correct_index": 1,
                "explanation": "When actions are conditioned globally (e.g., concatenated to every spatial position), the video model may generate implausible effects: objects moving without being touched, or effects propagating instantly across the scene. Spatial localization says 'this action affects this region' -- constraining the model to generate physically plausible causal effects that propagate from the contact point.",
                "why_it_matters": "The causality of action effects is a fundamental physics principle that generic video models do not respect. IRASim's spatial localization is an inductive bias that encodes this physical prior. This design choice illustrates a broader principle: the more physical structure you encode in the architecture, the less data you need to learn correct dynamics. But it also limits flexibility for tasks where actions have non-local effects (e.g., pulling a tablecloth).",
            },
            {
                "question": "IRASim is trained on robot manipulation datasets to learn interaction dynamics. What is the fundamental data limitation compared to training on internet-scale video?",
                "choices": [
                    "Robot datasets have lower image resolution than internet video",
                    "Robot manipulation datasets have limited object diversity and scene variety, so the model struggles with novel objects and environments not represented in the training data",
                    "Robot datasets do not include action labels",
                    "Internet video provides better physical dynamics than robot data",
                ],
                "correct_index": 1,
                "explanation": "Robot manipulation datasets contain thousands to hundreds of thousands of demonstrations with perhaps dozens of object types. Internet video contains millions of videos with enormous object and scene diversity. IRASim's interaction modeling is highly accurate for trained objects but may produce incorrect interaction dynamics for novel objects whose physical properties (weight, friction, deformability) differ from the training set.",
                "why_it_matters": "This is the fundamental tension between interaction accuracy (from robot-specific training) and visual/physical diversity (from internet-scale pre-training). UniSim and Cosmos take the opposite approach -- starting with internet-scale pre-training and adding action conditioning. Understanding this trade-off helps evaluate which approach is better for which deployment scenario: lab settings with known objects vs. open-world deployment with novel objects.",
            },
            {
                "question": "IRASim can be used to augment training data for downstream policy learning. What is the key risk of training policies on generated video rollouts mixed with real data?",
                "choices": [
                    "Generated and real data require different policy architectures",
                    "The policy may learn to exploit systematic biases in the generated data -- such as overly smooth contact transitions or missing failure modes -- that do not exist in reality",
                    "Mixing data sources always degrades policy performance",
                    "Generated videos cannot include reward signals",
                ],
                "correct_index": 1,
                "explanation": "Generated videos have systematic biases: contacts may look smoother than reality, certain failure modes (slipping, collision rebounds) may be underrepresented, and subtle physics (friction, deformation) may be incorrect. A policy trained on this mixed data may learn shortcuts that exploit these biases, performing well on generated data but failing in reality. Careful validation and mixing ratios are essential.",
                "why_it_matters": "Data augmentation via generation is increasingly common in robotics. Understanding the failure modes of synthetic data -- not just its benefits -- is critical for responsible deployment. This issue connects to broader concerns about training on model-generated data in language models (model collapse) and the importance of distribution match between training and deployment data.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #41  PhysDreamer
    # ------------------------------------------------------------------
    {
        "number": 41,
        "summary": (
            "PhysDreamer integrates physical priors into world model learning, going beyond purely "
            "data-driven video prediction by incorporating knowledge of physics (rigid body dynamics, "
            "soft body deformation, fluid behavior). By combining neural rendering with physics-based "
            "simulation priors, PhysDreamer produces imagined rollouts that are not just visually "
            "plausible but physically consistent, enabling more reliable policy training."
        ),
        "key_learnings": (
            "- Purely data-driven video prediction models can produce visually plausible but physically impossible predictions; physics priors constrain the output space to physically valid trajectories\n"
            "- PhysDreamer estimates physical properties (material type, stiffness, density) from visual observations and uses them to guide dynamics prediction\n"
            "- Combining differentiable physics simulation with neural rendering allows end-to-end training while maintaining physical consistency\n"
            "- Physics priors are most impactful for contact-rich interactions where pure learning struggles: collisions, deformation, fluid-object interactions\n"
            "- The approach trades generality for physical accuracy: it handles fewer visual scenarios than unconstrained video models but produces more reliable dynamics"
        ),
        "reading_guide": (
            "Begin with Section 1 for the motivation of physics-informed world models. Section 3 on "
            "the architecture is critical -- understand how physics simulation is integrated with neural "
            "rendering. Section 3.2 on material property estimation shows how physical parameters are "
            "inferred from visual input. Section 4 experiments should focus on comparisons against "
            "physics-free baselines on contact-rich tasks."
        ),
        "questions": [
            {
                "question": "PhysDreamer integrates physics simulation priors into learned world models. What class of prediction errors does this specifically address that pure data-driven models struggle with?",
                "choices": [
                    "Errors in background rendering and lighting estimation",
                    "Physically impossible object behaviors under contact -- objects interpenetrating, violating conservation of momentum, or deforming incorrectly -- which data-driven models produce when extrapolating beyond training examples",
                    "Errors in camera pose estimation and view synthesis",
                    "Slow inference speed of data-driven models",
                ],
                "correct_index": 1,
                "explanation": "Data-driven video models learn statistical regularities but do not have hard constraints for physical laws. When predicting scenarios slightly outside their training distribution, they may generate objects passing through each other, unrealistic bouncing, or incorrect deformations. Physics priors provide hard constraints (non-penetration, momentum conservation, constitutive models) that prevent these physically impossible predictions.",
                "why_it_matters": "For robotic manipulation, physical correctness matters more than visual quality. A policy trained on a world model that allows object interpenetration will learn unrealistic strategies. PhysDreamer's approach represents a middle ground between pure simulation (correct but limited visually) and pure learning (flexible but physically unconstrained). Understanding when physics priors are necessary versus when data alone suffices is essential for choosing the right world model approach.",
            },
            {
                "question": "PhysDreamer estimates physical properties (stiffness, density) from visual observations. Why is this property estimation step a critical bottleneck for the entire approach?",
                "choices": [
                    "Physical property estimation is computationally expensive but always accurate",
                    "Incorrect physical property estimates propagate through the physics simulation, producing dynamics errors that are internally consistent but wrong -- the model confidently predicts incorrect behavior",
                    "Physical properties cannot be estimated from visual observations at all",
                    "Property estimation is only needed for rigid objects, not deformable ones",
                ],
                "correct_index": 1,
                "explanation": "If the model estimates a foam object as rigid (wrong stiffness), the physics simulation will confidently predict rigid-body dynamics -- no deformation on contact. This is worse than a pure data-driven model's blurry uncertainty because the prediction looks physically plausible but is systematically wrong. The physics engine amplifies estimation errors rather than smoothing them out.",
                "why_it_matters": "This reveals a subtle failure mode of physics-informed approaches: they can be confidently wrong. Pure data-driven models tend to produce blurry, uncertain predictions in unfamiliar situations -- which at least signals low confidence. Physics-informed models can produce sharp, confident, but incorrect predictions when physical properties are misestimated. This distinction is important for safety-critical applications.",
            },
            {
                "question": "PhysDreamer combines differentiable physics with neural rendering. What computational challenge does this hybrid approach introduce compared to purely neural world models?",
                "choices": [
                    "Differentiable physics is not compatible with gradient-based training",
                    "The physics simulation introduces stiff differential equations and contact discontinuities that create challenging gradient landscapes, requiring specialized solvers and training procedures",
                    "Neural rendering cannot handle the output of physics simulation",
                    "The hybrid approach requires double the GPU memory of either approach alone",
                ],
                "correct_index": 1,
                "explanation": "Physics simulations involve contact events (discontinuous), stiff differential equations (requiring small time steps), and constraint solving (iterative). Backpropagating through these operations produces noisy or exploding gradients. This requires specialized techniques: differentiable contact models, gradient clipping, curriculum training on contact complexity, or alternating optimization between the physics and rendering components.",
                "why_it_matters": "Differentiable physics is a beautiful idea in theory but challenging in practice. The gradient issues explain why most world model papers avoid explicit physics and why Genesis takes a different approach (generative engine rather than differentiable physics). Understanding these computational challenges helps you evaluate claims about 'end-to-end differentiable physics simulation' with appropriate skepticism.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #42  Genesis
    # ------------------------------------------------------------------
    {
        "number": 42,
        "summary": (
            "Genesis is a generative physics engine that can programmatically create diverse physics "
            "simulations across multiple physics domains (rigid bodies, soft bodies, fluids, articulated "
            "bodies). Rather than learning physics from data, Genesis provides a fast, differentiable "
            "physics simulation platform that uses generative AI to automate the creation of simulation "
            "scenarios, environments, and tasks at scale."
        ),
        "key_learnings": (
            "- Genesis inverts the video-model-as-simulator paradigm: instead of learning physics from video, it uses a physics engine as the ground truth and applies generative AI to automate scenario creation\n"
            "- The platform supports multiple physics solvers (rigid, soft, fluid, cloth) unified under a single differentiable framework\n"
            "- Generative scene creation uses language or procedural specifications to automatically generate diverse training environments, addressing the environment diversity bottleneck\n"
            "- Differentiable simulation enables gradient-based policy optimization directly through the physics, bypassing the need for reward engineering or RL exploration\n"
            "- Speed is a key contribution: Genesis runs significantly faster than existing simulators (MuJoCo, Isaac Sim), enabling large-scale parallel data generation"
        ),
        "reading_guide": (
            "Section 1 frames the key insight: generative AI for scenario creation, physics engines for "
            "dynamics. Section 3 covers the physics engine architecture -- focus on how multiple physics "
            "solvers are unified. Section 4 on generative scenario creation shows the AI-assisted pipeline "
            "for creating diverse environments. Section 5 benchmarks speed against existing simulators. "
            "Section 6 applications in policy learning demonstrate the complete pipeline."
        ),
        "questions": [
            {
                "question": "Genesis uses a physics engine for dynamics rather than learning physics from video like UniSim or Cosmos. What is the fundamental advantage of this approach for contact-rich manipulation?",
                "choices": [
                    "Physics engines are easier to implement than video generation models",
                    "Physics engines maintain exact physical state and enforce conservation laws, providing ground-truth dynamics for contact, friction, and deformation that learned models can only approximate",
                    "Physics engines produce more visually realistic renderings than video models",
                    "Physics engines do not require any data or calibration",
                ],
                "correct_index": 1,
                "explanation": "For contact-rich tasks (assembly, insertion, deformable manipulation), precise force, friction, and collision dynamics are essential. Physics engines solve the exact equations of motion with hard constraints (non-penetration, friction cones, conservation laws). Learned video models can only approximate these dynamics from data, and their errors compound at contact events where small force differences produce large state divergence.",
                "why_it_matters": "This highlights the fundamental divide in the world model literature: learned models (flexible, photorealistic, broad coverage) vs. physics engines (precise, physically correct, narrow coverage). Neither is universally better. Understanding when each is appropriate -- video models for visually-driven tasks with simple contacts, physics engines for precise manipulation -- is essential for system design.",
            },
            {
                "question": "Genesis uses generative AI to automate the creation of simulation scenarios rather than relying on manual scene design. What bottleneck in traditional robot simulation does this address?",
                "choices": [
                    "Traditional simulators are too slow to run in real-time",
                    "The primary bottleneck in simulation-based robot learning is not simulator speed but the manual effort required to create diverse, realistic environments and tasks -- Genesis automates this creative process",
                    "Traditional simulators cannot model rigid body physics",
                    "Generative scenario creation is needed because physics engines cannot handle more than one object",
                ],
                "correct_index": 1,
                "explanation": "Building a diverse simulation environment (placing objects, setting materials, creating task variations) takes extensive human engineering. This manual bottleneck means most sim-to-real papers train in only a handful of environments. Genesis uses generative AI (language-to-scene, procedural generation) to create thousands of diverse scenarios automatically, enabling the data diversity that generalization requires.",
                "why_it_matters": "Data diversity -- not simulator fidelity -- is often the real bottleneck in sim-to-real transfer. Domain randomization and procedural generation partially address this, but they require manual specification of randomization ranges. Genesis's generative approach goes further by using AI to create qualitatively different scenarios. This connects to the broader insight that data diversity matters more than model capacity for generalization.",
            },
            {
                "question": "Genesis provides differentiable simulation, enabling gradient-based policy optimization through the physics engine. What is the key limitation of this approach compared to RL-based policy learning?",
                "choices": [
                    "Differentiable simulation cannot handle continuous action spaces",
                    "Gradient-based optimization through physics finds local optima and struggles with tasks requiring discrete decisions, mode switching, or exploration of qualitatively different strategies",
                    "Differentiable simulation is always slower than reinforcement learning",
                    "Gradient-based optimization cannot handle visual observations",
                ],
                "correct_index": 1,
                "explanation": "Differentiable physics provides analytical gradients but only for the current strategy. It cannot explore qualitatively different approaches (e.g., grasping from the left vs. right). The optimization landscape of physical tasks often has many local optima separated by contact discontinuities. RL's stochastic exploration can discover globally better strategies that gradient descent misses, especially for tasks with discrete mode switches (pick-and-place vs. push).",
                "why_it_matters": "Differentiable simulation vs. RL is a fundamental trade-off in robot learning. Differentiable approaches are fast and precise for smooth optimization landscapes but fail on complex, multi-modal tasks. RL is robust to landscape complexity but sample-inefficient. Most practical systems combine both: differentiable physics for trajectory refinement, RL for high-level strategy discovery. Understanding this trade-off helps you choose the right optimization approach for a given task.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #43  VideoVLA
    # ------------------------------------------------------------------
    {
        "number": 43,
        "summary": (
            "VideoVLA uses video prediction models as generalizable robot manipulators by first generating "
            "a video of the desired future manipulation, then extracting robot actions from the predicted "
            "video. This two-stage approach -- predict what the task should look like, then figure out what "
            "actions achieve it -- decouples visual planning from motor control, leveraging pre-trained "
            "video generation models for their visual planning capabilities."
        ),
        "key_learnings": (
            "- Decomposing manipulation into visual planning (video prediction) and action extraction (inverse model) leverages the strengths of each component separately\n"
            "- Pre-trained video generation models serve as visual planners that generalize to novel objects and scenes through their internet-scale pre-training\n"
            "- The inverse model (video-to-action) can be much simpler than an end-to-end policy because it only needs to map between predicted visual states rather than reason about the full task\n"
            "- This decoupling enables transferring the video model's generalization to robot control without requiring robot action labels during video model pre-training\n"
            "- The approach inherits both the strengths (visual diversity, physical intuition) and weaknesses (temporal consistency, physical accuracy) of the underlying video model"
        ),
        "reading_guide": (
            "Section 1 for the key insight of video prediction as visual planning. Section 3 covers "
            "the two-stage architecture: video generation for planning and inverse dynamics for action "
            "extraction. Focus on Section 3.2 (inverse model design) -- this is where the approach "
            "connects visual predictions to actual robot commands. Section 4 evaluations should focus "
            "on generalization to unseen objects and tasks."
        ),
        "questions": [
            {
                "question": "VideoVLA separates visual planning (video prediction) from action extraction (inverse model). What is the key advantage of this decomposition over end-to-end VLA models?",
                "choices": [
                    "End-to-end models cannot generate visual predictions at all",
                    "The video planning model and inverse model can be trained on different data distributions: the video model on internet-scale video (no actions) and the inverse model on robot data (with actions), maximizing data utilization",
                    "The decomposition eliminates the need for real robot demonstrations entirely",
                    "End-to-end models are always slower at inference than decomposed architectures",
                ],
                "correct_index": 1,
                "explanation": "End-to-end VLAs require paired (observation, language, action) data, which is scarce. VideoVLA's decomposition lets the video model learn from the vast supply of internet video (no actions needed) while only the inverse model requires the scarce robot action data. This dramatically expands the effective training data for the planning component.",
                "why_it_matters": "Data availability drives architectural choices in robotics. The decomposition strategy is a form of divide-and-conquer that matches each component to its richest data source. This same principle appears across the field: SuSIE decomposes into subgoal prediction + action execution, and RT-H separates language reasoning from motor control. Understanding when decomposition helps (data-limited regimes) vs. hurts (when end-to-end training finds better joint representations) is essential.",
            },
            {
                "question": "The inverse model in VideoVLA maps predicted video frames to robot actions. What is the primary failure mode when the video prediction contains physically implausible transitions?",
                "choices": [
                    "The inverse model will output zero actions",
                    "The inverse model may produce actions that attempt to achieve the impossible predicted state, leading to unsafe or erratic robot behavior when no valid action sequence exists",
                    "Physically implausible video predictions are automatically detected and filtered",
                    "The inverse model ignores video predictions and uses a hardcoded fallback policy",
                ],
                "correct_index": 1,
                "explanation": "If the video model predicts an object teleporting or interpenetrating (physically impossible), the inverse model will still try to find actions that achieve the predicted visual state. Since no valid actions exist for impossible transitions, the inverse model outputs nonsensical commands. This creates a safety concern: the robot attempts physically impossible goals, potentially leading to collisions or damage.",
                "why_it_matters": "This is a critical safety issue for any decomposed planning architecture. The inverse model has no mechanism to reject implausible plans -- it always tries to follow the video prediction. This highlights the importance of physical feasibility checking in planning pipelines and connects to broader safety concerns in AI-driven robotics: the system confidently executes invalid plans because no component is responsible for feasibility validation.",
            },
            {
                "question": "VideoVLA relies on temporal consistency in generated video for coherent action extraction. Why is temporal consistency harder for video generation models than spatial consistency within a single frame?",
                "choices": [
                    "Temporal consistency only requires higher resolution, which is straightforward to add",
                    "Video models generate frames sequentially or in parallel, and small per-frame errors in object position, shape, or appearance compound across frames, producing drift that has no analog in single-frame generation",
                    "Spatial consistency is equally challenging for modern video models",
                    "Temporal consistency is only important for videos longer than 10 seconds",
                ],
                "correct_index": 1,
                "explanation": "In single-frame generation, the model produces one globally consistent image. In video, small errors compound: an object might shift by 2 pixels per frame, resulting in large drift over 30 frames. The model must maintain identity, position, and physical plausibility across all frames simultaneously. This temporal coherence challenge is fundamentally harder because small errors accumulate rather than averaging out.",
                "why_it_matters": "Temporal consistency is the Achilles' heel of all video-prediction-based approaches to robotics. It limits effective prediction horizons and explains why shorter planning horizons tend to work better. Understanding this compounding error problem is essential for evaluating any paper that uses video prediction for robot planning -- the impressive single-frame quality in demos often masks significant temporal drift over longer horizons.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #45  HPT
    # ------------------------------------------------------------------
    {
        "number": 45,
        "summary": (
            "Heterogeneous Pre-trained Transformers (HPT) address the challenge of pre-training a single "
            "policy across robots with different observation spaces (cameras, proprioception, tactile) and "
            "action spaces (joint angles, end-effector poses, different DOFs). HPT uses modality-specific "
            "tokenizers to project heterogeneous inputs into a shared token space, then processes them "
            "with a single Transformer trunk, enabling cross-robot pre-training and transfer."
        ),
        "key_learnings": (
            "- Heterogeneous tokenization is the key architectural innovation: modality-specific encoders convert diverse sensor inputs and action outputs into a unified token format\n"
            "- A shared Transformer trunk can learn cross-robot representations when the tokenization layer handles the heterogeneity\n"
            "- Pre-training across diverse robot datasets (different embodiments, sensors, tasks) transfers to new robots via fine-tuning only the tokenizers while keeping the trunk frozen\n"
            "- The approach reveals that manipulation policies share more cross-robot structure than previously assumed -- a single trunk captures common manipulation strategies\n"
            "- The tokenizer design must balance information preservation (lossless conversion of each modality) with token efficiency (manageable sequence lengths)"
        ),
        "reading_guide": (
            "Section 1 motivates the heterogeneity problem clearly. Section 3 (Architecture) is the core "
            "-- focus on how modality-specific stem tokenizers (Section 3.1) and action-specific heads "
            "(Section 3.2) interface with the shared trunk. Section 4's cross-robot transfer experiments "
            "are the main empirical contribution. Figure 2 is essential for understanding the architecture. "
            "Section 4.3 ablations on trunk size and data diversity are informative."
        ),
        "questions": [
            {
                "question": "HPT uses modality-specific tokenizers to project heterogeneous robot inputs into a shared token space. What is the fundamental design tension in these tokenizers?",
                "choices": [
                    "Tokenizers must choose between supporting images and supporting proprioception, but not both",
                    "Tokenizers must preserve enough modality-specific information for control while producing representations generic enough for the shared trunk to process -- too much specificity prevents cross-robot transfer, too little loses critical control information",
                    "Tokenizers must be pre-trained separately for each robot, eliminating any cross-robot benefit",
                    "The only challenge is computational: tokenizers are the inference bottleneck",
                ],
                "correct_index": 1,
                "explanation": "If tokenizers preserve every detail of each modality, the resulting tokens are so robot-specific that the shared trunk cannot find common patterns across robots. If tokenizers are too aggressive in standardization, they lose critical information (e.g., tactile forces, precise joint angles) needed for fine-grained control. The optimal tokenizer finds the abstraction level where cross-robot commonalities live.",
                "why_it_matters": "This tokenizer design tension is the central challenge of any cross-embodiment architecture (HPT, CrossFormer, GR00T). The abstraction level of the shared representation determines what transfers and what is lost. Too abstract: the model cannot do precise control. Too specific: the model cannot transfer. Understanding this trade-off is essential for evaluating cross-embodiment claims and designing practical transfer pipelines.",
            },
            {
                "question": "HPT pre-trains a shared Transformer trunk across multiple robot datasets, then fine-tunes only the tokenizers for new robots. Why is this 'freeze trunk, adapt tokenizers' transfer strategy more effective than fine-tuning the entire model?",
                "choices": [
                    "The trunk has fewer parameters than the tokenizers, so freezing it is faster",
                    "Fine-tuning the entire model on a new robot's small dataset causes catastrophic forgetting of cross-robot manipulation knowledge captured in the trunk, while adapting only the tokenizers preserves this shared knowledge",
                    "The trunk cannot be fine-tuned due to architectural constraints",
                    "Full fine-tuning requires more GPU memory than is available on consumer hardware",
                ],
                "correct_index": 1,
                "explanation": "The trunk captures shared manipulation strategies (approach, grasp, lift patterns) across all training robots. Fine-tuning it on one new robot's small dataset overwrites this shared knowledge (catastrophic forgetting). By freezing the trunk and only adapting the input/output tokenizers, the model retains its cross-robot competence while learning to map a new robot's specific sensors and actuators into the shared representation.",
                "why_it_matters": "This transfer strategy parallels the broader 'foundation model + adapter' paradigm in NLP (LoRA, prefix tuning). In robotics, it is even more important because robot datasets are tiny compared to language data. Understanding when to freeze vs. fine-tune which components is a critical practical skill. The same principle applies to VLAs: freezing the vision-language backbone and adapting the action head is common practice.",
            },
            {
                "question": "HPT reveals that diverse robot datasets share common manipulation structure in the shared trunk. What type of cross-robot transfer does this evidence suggest is feasible vs. infeasible?",
                "choices": [
                    "All types of transfer are equally feasible across all robot morphologies",
                    "Manipulation strategy transfer (approach-grasp-lift patterns) appears feasible, but fine-grained motor control transfer (precise force modulation, contact-rich assembly) likely requires embodiment-specific adaptation",
                    "Only visual feature transfer works; no motor strategy transfer occurs",
                    "Transfer only works between robots with identical kinematic structures",
                ],
                "correct_index": 1,
                "explanation": "HPT's shared trunk captures high-level manipulation strategies that generalize across robots: recognizing graspable configurations, planning approach trajectories, sequencing pick-and-place. But fine-grained motor control -- exactly how hard to grip, precise insertion forces, contact-rich manipulation -- depends heavily on specific robot dynamics (mass, friction, compliance). These require the embodiment-specific tokenizers and potentially embodiment-specific training.",
                "why_it_matters": "Understanding the granularity of cross-robot transfer is critical for setting realistic expectations. The field often conflates 'cross-embodiment transfer' with 'universal robot control.' HPT's results suggest the truth is nuanced: high-level strategies transfer, but low-level control does not. This has practical implications: cross-embodiment pre-training helps bootstrap learning on new robots but does not eliminate the need for robot-specific fine-tuning.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #46  CrossFormer
    # ------------------------------------------------------------------
    {
        "number": 46,
        "summary": (
            "CrossFormer enables policy transfer across different robot embodiments using a Transformer "
            "architecture with embodiment-specific tokenization and a shared cross-embodiment attention "
            "mechanism. It demonstrates that a single policy model can control robots with different "
            "morphologies (arms with different DOFs, grippers vs. dexterous hands) by learning shared "
            "manipulation representations that abstract away embodiment details."
        ),
        "key_learnings": (
            "- Cross-embodiment transfer is possible when the policy operates on an abstraction level above specific joint configurations -- task-space representations and visual features transfer better than joint-space policies\n"
            "- Embodiment-specific tokenizers handle the morphology gap, converting each robot's unique observation/action format into a common representation\n"
            "- Cross-attention between visual tokens and embodiment tokens allows the model to learn how each robot's capabilities relate to the visual task\n"
            "- Data from diverse embodiments provides a form of augmentation: the model learns what aspects of manipulation are embodiment-invariant\n"
            "- Fine-tuning on a new embodiment with minimal data is more effective after cross-embodiment pre-training than training from scratch"
        ),
        "reading_guide": (
            "Read Section 1 for the motivation and scope of cross-embodiment transfer. Section 3 details "
            "the architecture: embodiment tokenizers, the shared Transformer trunk, and the cross-attention "
            "mechanism. Section 4 experiments test transfer to held-out embodiments -- focus on few-shot "
            "transfer results. Compare with HPT's approach to understand different solutions to the same "
            "problem. Section 4.3 ablations on which components to share vs. specialize are informative."
        ),
        "questions": [
            {
                "question": "CrossFormer uses cross-attention between visual and embodiment tokens rather than simply concatenating them. What does this architectural choice enable?",
                "choices": [
                    "Cross-attention is faster than concatenation at inference time",
                    "Cross-attention allows the model to dynamically compute which visual features are relevant given the current robot's capabilities, rather than treating all visual information equally regardless of embodiment",
                    "Concatenation is not possible when observation spaces have different dimensionalities",
                    "Cross-attention eliminates the need for embodiment-specific tokenizers",
                ],
                "correct_index": 1,
                "explanation": "Different robots can reach different workspace regions, grasp different object sizes, and apply different forces. Cross-attention lets the model attend to visual features relevant to the current robot's capabilities: a small gripper attends more to small graspable features, while a dexterous hand attends to contact surfaces for in-hand manipulation. Concatenation treats all visual features equally regardless of the robot's ability to act on them.",
                "why_it_matters": "This architectural insight -- that what visual information matters depends on the robot -- has implications beyond CrossFormer. It suggests that vision processing in robot policies should not be embodiment-agnostic. The same scene looks 'different' to a parallel-jaw gripper vs. a dexterous hand because different affordances are available. This connects to affordance-based reasoning and embodied perception research.",
            },
            {
                "question": "CrossFormer demonstrates transfer across different robot DOF configurations (e.g., 6-DOF arm to 7-DOF arm). What is the most challenging type of morphology transfer that remains difficult for this approach?",
                "choices": [
                    "Transfer between arms with different joint limits",
                    "Transfer between fundamentally different morphologies -- such as from a fixed-base arm to a mobile manipulator or a humanoid -- where the task-execution strategy itself must change, not just the joint mapping",
                    "Transfer between robots with different camera placements",
                    "Transfer between robots from different manufacturers with the same DOF count",
                ],
                "correct_index": 1,
                "explanation": "CrossFormer handles similar morphologies well (6-DOF to 7-DOF arm) because the manipulation strategy is the same -- only the joint mapping changes. But transferring from a fixed arm to a mobile manipulator requires fundamentally different strategies: the mobile robot must plan base movement, coordinate base-arm motion, and handle a different workspace. The shared trunk's learned strategies may not apply when the execution paradigm itself changes.",
                "why_it_matters": "This exposes the boundary of current cross-embodiment transfer. Similar morphologies share manipulation strategies; dissimilar morphologies may not. This distinction is crucial for evaluating claims in papers like GR00T and FLOWER that target 'universal' robot policies. True cross-embodiment generalization requires strategy-level transfer, not just kinematic mapping -- and this remains an open problem.",
            },
            {
                "question": "CrossFormer trains on data from multiple embodiments simultaneously. What is the risk of data imbalance when some robot platforms contribute significantly more demonstrations than others?",
                "choices": [
                    "Data imbalance has no effect on the shared trunk's representations",
                    "The shared trunk's representations become biased toward the dominant embodiment's manipulation strategies, causing the model to learn strategies that transfer poorly to underrepresented morphologies",
                    "Data imbalance only affects training speed, not model quality",
                    "More data from one embodiment always helps all other embodiments equally",
                ],
                "correct_index": 1,
                "explanation": "If 80% of training data comes from Franka Panda arms, the shared trunk's 'common manipulation knowledge' is disproportionately shaped by Franka-specific behaviors: its workspace constraints, speed limits, and typical task setups. When transferring to an underrepresented robot, these biased representations may not capture the target robot's relevant manipulation patterns. Careful data balancing or importance weighting is needed.",
                "why_it_matters": "Data imbalance is a pervasive problem in cross-embodiment learning because some robot platforms are far more common in research (Franka, UR5) than others. This mirrors data imbalance issues in multilingual NLP (English dominates). Understanding how imbalance affects the shared representation helps design better training procedures and set realistic expectations for transfer to underrepresented platforms.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #47  GR00T N1
    # ------------------------------------------------------------------
    {
        "number": 47,
        "summary": (
            "GR00T N1 is NVIDIA's generalist robot foundation model designed for humanoid and "
            "manipulation tasks across different embodiments. It combines vision-language understanding "
            "with motor control in a single model, using a dual-system architecture that separates "
            "high-level reasoning (language, visual understanding) from low-level motor control "
            "(action generation) while sharing representations through a learned interface."
        ),
        "key_learnings": (
            "- The dual-system architecture mirrors cognitive science's System 1/System 2 framework: a slow deliberative system for task understanding and a fast reactive system for motor control\n"
            "- Cross-embodiment design handles humanoids and manipulation arms through embodiment-specific action heads attached to a shared perception-reasoning backbone\n"
            "- Large-scale pre-training on paired vision-language-action data creates a foundation model that can be fine-tuned for specific robots with limited demonstrations\n"
            "- The model uses diffusion-based action generation for the motor control component, capturing multi-modal action distributions important for manipulation\n"
            "- Integration with NVIDIA's simulation stack (Isaac Sim, Cosmos) enables large-scale synthetic data generation for pre-training"
        ),
        "reading_guide": (
            "Section 1 frames the vision of generalist robot models. Focus on the architecture overview "
            "(Section 3) -- the separation between the vision-language backbone and the action generation "
            "module is the key design choice. Section 3.2 on the diffusion-based action head explains how "
            "multi-modal action distributions are captured. Section 4 on pre-training data sources "
            "and Section 5 on fine-tuning results across embodiments are the main empirical contributions."
        ),
        "questions": [
            {
                "question": "GR00T N1 uses a dual-system architecture separating high-level reasoning from low-level motor control. What is the key advantage of this separation compared to end-to-end VLA models that produce actions directly from a single model?",
                "choices": [
                    "The separation makes the model smaller and faster to train",
                    "Each system can operate at different temporal frequencies -- slow deliberation for task understanding (1-2 Hz) and fast reactive control (50+ Hz) -- which a single model running at one frequency cannot efficiently handle",
                    "The separation eliminates the need for any pre-training",
                    "End-to-end models cannot process language instructions",
                ],
                "correct_index": 1,
                "explanation": "Task understanding (parsing instructions, recognizing objects, planning sequences) operates at human timescales (seconds). Motor control (trajectory tracking, force regulation, reactive grasping) operates at millisecond timescales. A single model must either run its full computation at the fast rate (wasteful and slow) or run at the slow rate (too sluggish for reactive control). The dual-system design lets each component run at its natural frequency.",
                "why_it_matters": "The multi-frequency control problem is fundamental to robotics and often ignored in VLA papers. Running a 7B parameter VLM at 50 Hz is computationally infeasible. But running control at 2 Hz produces jerky, unresponsive behavior. GR00T's dual-system design addresses this with an architecture that scales. This same insight drives pi0's action chunking and RT-H's hierarchical decomposition.",
            },
            {
                "question": "GR00T N1 uses diffusion-based action generation rather than autoregressive token prediction for its motor control system. Why is this choice particularly important for humanoid control?",
                "choices": [
                    "Diffusion models are faster to sample from than autoregressive models for high-dimensional outputs",
                    "Humanoid robots have high-dimensional, continuous action spaces with inherently multi-modal distributions (many valid joint configurations for the same task), which diffusion naturally captures while tokenization struggles",
                    "Autoregressive models cannot generate actions for more than 7 DOF",
                    "Diffusion-based actions are more interpretable to human operators",
                ],
                "correct_index": 1,
                "explanation": "A humanoid has 30+ DOF with highly redundant kinematics -- many joint configurations achieve the same end-effector pose. This creates inherently multi-modal action distributions. Autoregressive token prediction imposes an artificial ordering on joints and discretizes continuous values, losing the correlation structure between joints. Diffusion generates the full joint configuration simultaneously, naturally representing the multi-modal, correlated distribution over high-dimensional actions.",
                "why_it_matters": "As robots become more complex (humanoids, dexterous hands), action space dimensionality and multi-modality increase dramatically. The choice between diffusion, flow matching, and tokenization for action generation becomes increasingly consequential. Understanding why diffusion is favored for high-DOF systems (and where tokenization still works well, like simple grippers) is essential for evaluating generalist robot architectures.",
            },
            {
                "question": "GR00T N1 is designed to work across different embodiments via shared pre-training. What is the most significant open question about whether this cross-embodiment pre-training actually helps?",
                "choices": [
                    "Whether the model can handle different camera resolutions across robots",
                    "Whether the shared representations capture embodiment-invariant manipulation knowledge, or whether the model simply memorizes separate policies for each embodiment seen during training without meaningful transfer",
                    "Whether the model supports both simulation and real-world deployment",
                    "Whether NVIDIA's hardware is required for inference",
                ],
                "correct_index": 1,
                "explanation": "A model trained on data from 10 robots could be learning 10 separate policies under one set of weights (memorization) or learning shared manipulation principles that compose with embodiment-specific adaptations (transfer). Only zero-shot or few-shot performance on a truly novel embodiment distinguishes these hypotheses. Most cross-embodiment papers, including GR00T, have limited evidence for genuine transfer to unseen morphologies.",
                "why_it_matters": "This is perhaps the most important open question in cross-embodiment robot learning. If cross-embodiment pre-training merely memorizes per-robot policies, it provides no advantage over training separate models. True cross-embodiment transfer would be transformative -- enabling rapid deployment on new robots. But evaluating transfer requires testing on embodiments NOT in the training set, which few papers convincingly demonstrate.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #49  GR-2
    # ------------------------------------------------------------------
    {
        "number": 49,
        "summary": (
            "GR-2 is ByteDance's generalist robot model that uses video generation as its world model "
            "backbone for policy learning. It pre-trains a large video generation model on internet-scale "
            "video, fine-tunes it with robot action conditioning, then uses the resulting world model to "
            "train manipulation and navigation policies. GR-2 demonstrates that video generation quality "
            "directly translates to downstream policy performance."
        ),
        "key_learnings": (
            "- Video generation pre-training on internet data provides strong visual and physical priors that transfer to robotics when fine-tuned with action conditioning\n"
            "- GR-2 scales to 1.5B+ parameters, showing that larger video generation models produce better world models which in turn train better policies\n"
            "- The approach unifies visual imagination and policy learning in a single pipeline: generate what should happen, then learn to make it happen\n"
            "- Fine-tuning with robot data is essential: the pre-trained video model alone does not understand how actions map to visual changes\n"
            "- GR-2 demonstrates generalization to novel objects and scenes by leveraging the pre-trained model's visual diversity"
        ),
        "reading_guide": (
            "Section 1 frames the connection between video generation and robotic policy learning. "
            "Section 3 covers the video generation architecture and pre-training procedure. Section 4 "
            "on action conditioning and robot fine-tuning is where the video model becomes a world model. "
            "Section 5 evaluates policies trained inside the world model. Focus on the scaling experiments "
            "showing how video model quality correlates with policy performance."
        ),
        "questions": [
            {
                "question": "GR-2 pre-trains on internet video then fine-tunes with robot actions. Why does the pre-training distribution (passive third-person video) create a systematic bias when used as a first-person robot world model?",
                "choices": [
                    "Internet video has lower resolution than robot cameras",
                    "Internet video predominantly shows objects being manipulated by human hands from a third-person viewpoint, creating a viewpoint and embodiment distribution mismatch with first-person robot manipulation where the robot arm itself is visible",
                    "Internet video does not contain any physical interactions with objects",
                    "The pre-training distribution is perfectly matched to robot deployment",
                ],
                "correct_index": 1,
                "explanation": "Internet video shows kitchens, workshops, and manipulation from human perspective -- typically third-person, with human hands. Robots see from mounted cameras with their own arm visible, from different viewpoints, and with different end-effectors. The pre-trained model has strong priors for human-hand manipulation from third-person views but weaker priors for robot-arm manipulation from first-person views. Fine-tuning bridges this gap but cannot fully overcome the distribution mismatch.",
                "why_it_matters": "This viewpoint/embodiment mismatch is often underappreciated in video-model-as-simulator papers. The implicit assumption that 'internet video teaches physics' overlooks the fact that it teaches physics from a specific viewpoint distribution. Understanding this bias helps explain why fine-tuning on robot data is essential (not optional) and why transfer performance degrades for embodiments/viewpoints far from the pre-training distribution.",
            },
            {
                "question": "GR-2 shows that larger video generation models produce better world models that train better policies. What is the key caveat about this 'scale improves policies' finding?",
                "choices": [
                    "Larger models always produce proportionally better policies regardless of task",
                    "Scaling video generation quality improves visual fidelity, but physically-critical dynamics accuracy may not scale at the same rate -- the model gets better at 'looking right' faster than it gets better at 'being right'",
                    "Scaling only helps for navigation tasks, not manipulation",
                    "The scaling relationship reverses at very large model sizes",
                ],
                "correct_index": 1,
                "explanation": "Video generation quality (FID, FVD) measures visual fidelity -- how realistic frames look. But policy performance depends on dynamics accuracy -- whether predicted interactions are physically correct. A larger model may produce much more photorealistic kitchen scenes (better FID) with only marginally improved understanding of how a spatula responds to being pushed (dynamics accuracy). The correlation between visual quality and dynamics accuracy is positive but imperfect.",
                "why_it_matters": "This distinction between visual quality and dynamics accuracy is critical for evaluating any video-model-as-world-model paper. Many papers report impressive video quality metrics without demonstrating that this translates to physically accurate simulation. Understanding that these are correlated but not equivalent prevents overestimating the value of scaling video models for robotics.",
            },
            {
                "question": "GR-2 demonstrates generalization to novel objects by leveraging the pre-trained model's visual diversity. What type of generalization does this visual diversity NOT provide?",
                "choices": [
                    "Generalization to objects with different colors than seen in training",
                    "Generalization to objects with novel physical properties (unusual mass, friction, compliance) that cannot be inferred from visual appearance alone",
                    "Generalization to different lighting conditions",
                    "Generalization to different table surfaces",
                ],
                "correct_index": 1,
                "explanation": "Visual diversity helps the model recognize and generate images of novel objects -- an unusual tool, a new food item. But physical properties that affect manipulation dynamics (an unexpectedly heavy mug, a slippery surface, a deformable material) cannot be inferred from appearance. The world model will predict dynamics based on visual similarity to training objects, which fails when novel objects have unexpected physical properties.",
                "why_it_matters": "This highlights a fundamental limitation of vision-only world models: physical properties that do not have reliable visual correlates (weight, friction, compliance, center of mass) cannot be predicted from images. This is why tactile sensing, force feedback, and explicit physics models remain important for manipulation -- they capture information that video models inherently cannot. Understanding this limitation scopes where video-based world models are and are not sufficient.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #50  FLOWER
    # ------------------------------------------------------------------
    {
        "number": 50,
        "summary": (
            "FLOWER (Flow-based Open-World Embodied Robot) is an open-source initiative to democratize "
            "generalist robot policy learning. It provides open models, training pipelines, and datasets "
            "for cross-embodiment policy learning, aiming to be a community-driven alternative to "
            "proprietary generalist robot models like RT-2 or GR00T. FLOWER emphasizes reproducibility, "
            "data sharing, and modular design."
        ),
        "key_learnings": (
            "- Open-source release of models, data pipelines, and training code enables community replication and extension of cross-embodiment research\n"
            "- Flow matching for action generation provides a balance between diffusion quality and inference speed, important for real-time robot control\n"
            "- Modular architecture design allows swapping components (visual encoders, action heads, language models) without retraining the full system\n"
            "- Cross-embodiment data standardization is as important as model architecture -- FLOWER defines shared data formats for heterogeneous robot datasets\n"
            "- Democratization requires not just open weights but open training recipes, evaluation protocols, and computing-efficient training strategies"
        ),
        "reading_guide": (
            "Section 1 for the democratization motivation and comparison to proprietary alternatives. "
            "Section 3 covers the modular architecture -- understand how components can be independently "
            "upgraded. Section 4 on data pipeline and standardization is often underappreciated but "
            "critical. Section 5 evaluations should focus on comparison against closed-source models "
            "to assess the performance gap. Read Section 6 on community and reproducibility efforts."
        ),
        "questions": [
            {
                "question": "FLOWER emphasizes modularity -- the ability to swap components (visual encoders, action heads, language models) independently. What is the key technical challenge that makes true modularity difficult in practice?",
                "choices": [
                    "Different components have different programming language requirements",
                    "Components trained end-to-end develop co-adapted representations that break when any single component is replaced, because the interfaces between modules are learned rather than standardized",
                    "Modular systems always have higher latency than monolithic systems",
                    "Modularity requires each component to have the same parameter count",
                ],
                "correct_index": 1,
                "explanation": "When a visual encoder and action head are trained end-to-end, the encoder learns to produce representations that the specific action head expects, and vice versa. Replacing either component introduces a representation mismatch that degrades performance. True modularity requires standardized interfaces between components, but this constrains the representation space and may reduce end-to-end performance compared to fully co-adapted systems.",
                "why_it_matters": "The modularity vs. end-to-end trade-off pervades ML systems design. End-to-end training produces better performance through co-adaptation but creates monolithic systems that are hard to upgrade. Modular systems are flexible but sacrifice co-adaptation benefits. Understanding this trade-off is essential for evaluating open-source projects like FLOWER and for designing systems that balance research flexibility with deployment performance.",
            },
            {
                "question": "FLOWER uses flow matching for action generation. Compared to the diffusion-based action generation used in pi0 and GR00T, what is the practical advantage of flow matching?",
                "choices": [
                    "Flow matching produces higher-quality action distributions than diffusion",
                    "Flow matching requires fewer denoising steps to produce good samples, enabling faster inference critical for real-time robot control without significant quality loss",
                    "Flow matching does not require any training, only inference",
                    "Flow matching can only be used with discrete action spaces",
                ],
                "correct_index": 1,
                "explanation": "Flow matching learns a direct transport map from noise to the target distribution using straight-line interpolation paths (conditional optimal transport). This simpler path structure requires fewer steps to traverse than diffusion's more complex noise schedule, typically producing good action samples in 5-10 steps vs. 20-50 for diffusion. For real-time control at 10+ Hz, this speed advantage is practically significant.",
                "why_it_matters": "Inference speed is a critical practical constraint for robot policies. A model that produces better actions but takes 100ms per step may be worse in practice than a faster model running at 50 Hz. Flow matching's speed advantage over diffusion explains its adoption in several recent VLAs (pi0 also uses flow matching). Understanding the quality-speed trade-off between different generative action models is essential for practical system design.",
            },
            {
                "question": "FLOWER aims to democratize generalist robot policies through open-source release. What is the biggest barrier to democratization that open-sourcing the model alone does NOT solve?",
                "choices": [
                    "Open-source models cannot run on NVIDIA GPUs",
                    "Training and fine-tuning generalist robot models requires large-scale compute and diverse robot hardware that most research labs lack, creating a resource barrier independent of model accessibility",
                    "Open-source licenses prevent commercial use of the models",
                    "The model weights are too large to download",
                ],
                "correct_index": 1,
                "explanation": "Open-sourcing a pre-trained model helps with inference and fine-tuning, but advancing the field requires training new models -- which demands massive compute (hundreds of GPUs for weeks) and diverse robot hardware (multiple platforms generating data). Most academic labs have neither. True democratization requires not just open models but compute grants, shared data infrastructure, and efficient training methods.",
                "why_it_matters": "The 'democratization' narrative in AI often focuses on model access while ignoring compute and data access. In robotics, the barrier is even higher because you also need physical hardware. Understanding these structural barriers helps evaluate whether open-source efforts genuinely democratize research or primarily benefit well-resourced labs that can fine-tune large models. This shapes research strategy for groups with limited resources.",
            },
        ],
    },
    # ------------------------------------------------------------------
    # #52  ST4VLA
    # ------------------------------------------------------------------
    {
        "number": 52,
        "summary": (
            "ST4VLA (Spatial-Temporal Representations for Vision-Language-Action Models) enhances VLA "
            "architectures by incorporating explicit spatial-temporal features that standard VLMs lack. "
            "While VLMs pre-trained on static images and text have limited understanding of 3D spatial "
            "relationships and temporal dynamics, ST4VLA adds dedicated spatial-temporal modules that "
            "provide the policy with geometry-aware and motion-aware representations essential for "
            "precise manipulation."
        ),
        "key_learnings": (
            "- Standard VLM visual encoders (trained on static images) lack explicit spatial-temporal features needed for manipulation: 3D geometry, depth, motion, temporal continuity\n"
            "- Adding dedicated spatial modules (depth estimation, 3D feature extraction) provides geometric understanding that image-only encoders must learn implicitly\n"
            "- Temporal modules that explicitly track object motion and state changes across frames give the policy velocity and acceleration information absent from single-frame features\n"
            "- The spatial-temporal features are complementary to, not replacements for, the semantic features from VLM encoders\n"
            "- Explicit spatial-temporal representations most benefit tasks requiring precise positioning, contact prediction, or dynamic interaction timing"
        ),
        "reading_guide": (
            "Section 1 identifies the spatial-temporal gap in current VLAs. Section 3 is the core "
            "contribution: understand the spatial module (Section 3.1) for geometry features and the "
            "temporal module (Section 3.2) for motion features. Section 3.3 on how these features are "
            "fused with the VLM backbone is architecturally important. Section 4 experiments should "
            "focus on the ablation showing which tasks benefit most from spatial vs. temporal features."
        ),
        "questions": [
            {
                "question": "ST4VLA adds explicit spatial-temporal modules to VLAs. Why do VLMs pre-trained on static images and text struggle to implicitly learn the spatial-temporal features needed for manipulation?",
                "choices": [
                    "VLMs cannot process image inputs at all",
                    "Static image-text pre-training teaches object recognition and semantic relationships but not 3D geometry, physical depth, or temporal dynamics -- these require either multi-view, video, or depth supervision that static image-text pairs do not provide",
                    "VLMs have too few parameters to learn spatial features",
                    "Spatial-temporal features are irrelevant for manipulation tasks",
                ],
                "correct_index": 1,
                "explanation": "VLM pre-training (CLIP, SigLIP) optimizes for matching images to text descriptions -- this teaches 'what is in the image' (semantics) but not 'where things are in 3D' (geometry) or 'how things are moving' (dynamics). Learning 3D geometry from 2D images alone is theoretically possible but requires specific architectural inductive biases or training objectives that standard VLMs lack. ST4VLA provides these through dedicated modules.",
                "why_it_matters": "This identifies a fundamental limitation of the VLA paradigm: VLM backbones provide excellent semantic features but poor geometric and temporal features. Every VLA implicitly asks its VLM to understand 3D space from 2D images -- which is an underspecified problem. ST4VLA's approach of adding explicit spatial-temporal modules is one solution, but others include using 3D visual encoders, depth cameras, or multi-view inputs. Understanding this gap is essential for diagnosing VLA failure modes.",
            },
            {
                "question": "ST4VLA shows that explicit spatial features (depth, 3D geometry) help most on tasks requiring precise positioning. What does this suggest about VLAs that perform well on such tasks WITHOUT explicit spatial features?",
                "choices": [
                    "Those VLAs have implicitly learned 3D geometry from their training data, likely through large-scale diverse robot data that provides indirect supervision",
                    "Precise positioning tasks do not actually require 3D understanding",
                    "Those VLAs use higher resolution cameras, which is equivalent to spatial features",
                    "The results are inconsistent and should be ignored",
                ],
                "correct_index": 0,
                "explanation": "If a VLA without explicit spatial features succeeds at precise positioning, it must be extracting 3D information implicitly from 2D images -- through monocular depth cues, perspective geometry, or learned object size priors. This implicit learning is possible but requires diverse data that exercises 3D reasoning. ST4VLA's explicit modules provide this knowledge more efficiently, especially in data-limited regimes.",
                "why_it_matters": "This question gets at a deep architectural debate: should capabilities be explicitly engineered (modular, interpretable, data-efficient) or implicitly learned (flexible, end-to-end, data-hungry)? Understanding that both can work -- but with different data efficiency and failure modes -- helps you evaluate claims from both camps. Models with explicit spatial features fail gracefully (the module works or it does not), while implicit features fail unpredictably.",
            },
            {
                "question": "ST4VLA adds temporal modules for tracking motion and state changes across frames. Why are explicit temporal features particularly important for dynamic manipulation tasks like catching or inserting moving objects?",
                "choices": [
                    "Temporal features are only needed for video generation, not policy learning",
                    "Without explicit velocity and acceleration estimates, the policy must infer object dynamics from the difference between consecutive static frame encodings, which loses temporal resolution and cannot distinguish between fast and slow motions within a single observation window",
                    "Temporal features are redundant if the policy processes multiple frames",
                    "Dynamic tasks do not exist in real robot deployment scenarios",
                ],
                "correct_index": 1,
                "explanation": "Standard VLAs that concatenate frame features rely on the Transformer to implicitly compute temporal differences. But temporal differences between static image features are a poor proxy for velocity: they mix motion information with appearance changes (lighting, shadows) and lack explicit temporal resolution. Dedicated temporal modules (optical flow, temporal convolutions) extract clean velocity and acceleration signals essential for timing-critical tasks.",
                "why_it_matters": "Temporal reasoning is increasingly recognized as a bottleneck for VLA performance on dynamic tasks. Most VLAs process 1-2 Hz observations with static image encoders -- this is adequate for quasi-static manipulation (pick and place) but inadequate for dynamic tasks (catching, pouring, handover). ST4VLA's explicit temporal features point toward a broader need for temporal awareness in robot perception. As manipulation tasks become more dynamic, this limitation will become more acute.",
            },
        ],
    },
]


def main():
    client = get_admin_client()
    for paper in PAPERS:
        num = paper["number"]
        client.table("papers").update({
            "summary": paper["summary"],
            "key_learnings": paper["key_learnings"],
            "reading_guide": paper["reading_guide"],
        }).eq("number", num).execute()
        for q in paper["questions"]:
            client.table("questions").insert({
                "paper_number": num,
                "question": q["question"],
                "choices": json.dumps(q["choices"]),
                "correct_index": q["correct_index"],
                "explanation": q["explanation"],
                "why_it_matters": q["why_it_matters"],
            }).execute()
        print(f"  #{num}: seeded")
    print(f"Done: {len(PAPERS)} papers")


if __name__ == "__main__":
    main()
