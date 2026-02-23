#!/usr/bin/env python3
"""Replace all questions with expert-level content + why_it_matters context."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_admin_client

PAPERS = [
    {
        "number": 1,
        "questions": [
            {
                "question": "RT-1 uses a relatively small 35M-parameter model yet generalizes well. What does this suggest about the scaling axis that matters most for robotic manipulation?",
                "choices": [
                    "Model capacity is the primary bottleneck — larger Transformers would do even better",
                    "Data diversity (more tasks, environments, objects) is more impactful than model scale",
                    "Pre-training on ImageNet is the critical factor enabling generalization",
                    "The Transformer architecture itself accounts for most of the gains over prior work",
                ],
                "correct_index": 1,
                "explanation": "RT-1 demonstrated that with 130k diverse demonstrations across 700+ tasks, even a 35M-param model achieves strong generalization. Subsequent work (RT-2, OpenVLA) confirmed this: data diversity — not model size alone — is the primary driver of robotic generalization.",
                "why_it_matters": "This question tests whether you understand the scaling laws for robotics, which differ from LLMs. In language, model scale reliably improves performance. In robotics, the bottleneck is data diversity — you need varied environments, objects, and tasks. This insight shaped the entire field's direction, leading to efforts like Open X-Embodiment to collect diverse cross-robot data.",
            },
            {
                "question": "RT-1 predicts actions as discrete token sequences. What is the key limitation of this action representation that motivated later work like Diffusion Policy and pi0?",
                "choices": [
                    "Discrete tokens can't represent high-frequency control signals",
                    "Tokenization destroys the continuous, multimodal structure of action distributions",
                    "Token-based models are too slow for real-time control",
                    "Discrete actions require a separate tokenizer for each robot morphology",
                ],
                "correct_index": 1,
                "explanation": "Discretizing actions into tokens forces a unimodal output — the model picks one token at a time. But manipulation often has multiple valid action trajectories (multimodal distributions). Diffusion Policy and pi0's flow matching preserve this continuous, multimodal structure.",
                "why_it_matters": "Action representation is one of the most consequential design choices in VLA architectures. Token-based actions (RT-1/RT-2) vs. diffusion-based (Diffusion Policy) vs. flow matching (pi0) represent fundamentally different trade-offs between leveraging LLM pre-training and capturing the true geometry of action spaces. Understanding this trade-off is essential for evaluating any new VLA paper.",
            },
            {
                "question": "RT-1 uses TokenLearner between the image encoder and the Transformer. What architectural problem does this solve?",
                "choices": [
                    "It converts RGB images into depth maps for better spatial reasoning",
                    "It compresses visual tokens to a fixed small set, making the Transformer tractable at high image resolutions",
                    "It aligns visual features with language embeddings via contrastive learning",
                    "It provides temporal attention across consecutive image frames",
                ],
                "correct_index": 1,
                "explanation": "TokenLearner adaptively selects and compresses image tokens from the EfficientNet features into a small fixed set (~8 tokens). Without it, the Transformer would need to attend over hundreds of spatial tokens per frame, making real-time 3Hz inference impossible.",
                "why_it_matters": "Visual token compression is a recurring challenge in VLAs. Every model — from RT-1's TokenLearner to OpenVLA's projection layers to pi0's visual resampler — must solve the same problem: images produce far more tokens than language, but the Transformer's quadratic attention cost makes this expensive. Understanding this bottleneck helps you evaluate architectural claims in new papers.",
            },
        ],
    },
    {
        "number": 2,
        "questions": [
            {
                "question": "RT-2 co-fine-tunes a VLM on both web data and robot trajectories, with actions as text tokens. Why is co-training on web data critical rather than just fine-tuning on robot data alone?",
                "choices": [
                    "Web data provides motion planning demonstrations that transfer to robots",
                    "It prevents catastrophic forgetting of semantic knowledge needed for instruction following and reasoning",
                    "Web images provide better visual features than robot camera images",
                    "Co-training is just a regularization technique to prevent overfitting on small robot datasets",
                ],
                "correct_index": 1,
                "explanation": "Fine-tuning solely on robot data causes catastrophic forgetting — the model loses the rich semantic understanding from VLM pre-training. Co-training preserves this knowledge, enabling emergent capabilities like understanding 'pick up something you'd use as an improvised hammer' which require web-scale commonsense.",
                "why_it_matters": "Catastrophic forgetting vs. knowledge retention is a central tension in all VLA work. The entire VLA paradigm rests on the assumption that web knowledge transfers to robotic control. If fine-tuning destroys that knowledge, the approach collapses. Papers like Knowledge-Insulating VLA (#8) and OpenVLA tackle this exact problem with different strategies — LoRA, frozen encoders, or co-training ratios.",
            },
            {
                "question": "RT-2 demonstrates 'emergent' robotic capabilities not present in the robot training data. What is the most precise explanation of how these emerge?",
                "choices": [
                    "The model memorizes internet demonstrations of tool use and manipulation",
                    "The VLM's semantic/conceptual knowledge composes with the robot's sensorimotor skills through the shared token space",
                    "The model learns physics simulation from web videos",
                    "Emergent capabilities are due to the 55B parameter model having enough capacity to learn any task",
                ],
                "correct_index": 1,
                "explanation": "The shared token space is key: actions become just another 'language' the VLM speaks. The model's conceptual knowledge ('rocks are hard and heavy, like a hammer') composes with its learned motor primitives ('grasp and pick up object') — neither alone produces the behavior.",
                "why_it_matters": "Understanding compositionality in VLAs is crucial for predicting what these models can and can't do. The emergent behavior isn't magic — it's compositional generalization through a shared representation space. This is also why VLAs sometimes fail in surprising ways: when the composition doesn't hold (e.g., the semantic concept doesn't map cleanly to a motor skill), the model produces incoherent behavior.",
            },
            {
                "question": "RT-2 is built on PaLM-E (12B) and PaLI-X (55B). What fundamental trade-off does this reveal about the VLA approach compared to smaller specialist models?",
                "choices": [
                    "VLAs are always better than specialist models given enough parameters",
                    "Massive compute at both training and inference is required to achieve the semantic reasoning benefits of the VLA paradigm",
                    "Larger base VLMs always produce proportionally better robot policies",
                    "The VLA approach only works with proprietary closed-source models",
                ],
                "correct_index": 1,
                "explanation": "RT-2's 55B model requires enormous compute — making it impractical for many robots and labs. This compute cost is the price of semantic reasoning. This trade-off directly motivated OpenVLA (7B, open-source) and TinyVLA, which ask: how much of the benefit can we retain at practical scale?",
                "why_it_matters": "Every VLA paper implicitly makes a bet on this trade-off. RT-2 bet big on scale. OpenVLA showed you can get 80% of the benefit at 13% of the parameters. Pi0 found flow matching more parameter-efficient than tokenization. Understanding this trade-off helps you evaluate whether a paper's claimed improvements are practically relevant or just require more compute.",
            },
        ],
    },
    {
        "number": 4,
        "questions": [
            {
                "question": "OpenVLA uses both DINOv2 and SigLIP as visual encoders. What complementary properties do these provide?",
                "choices": [
                    "DINOv2 handles RGB and SigLIP handles depth images",
                    "DINOv2 provides spatially-rich self-supervised features while SigLIP provides language-aligned semantic features",
                    "DINOv2 encodes the current frame and SigLIP encodes the goal image",
                    "They are redundant — the dual encoder is just an ensemble for robustness",
                ],
                "correct_index": 1,
                "explanation": "DINOv2 (self-supervised) learns fine-grained spatial features — object boundaries, textures, geometric structure. SigLIP (contrastive language-image) learns semantic features aligned with language. Together they give the policy both 'where things are' and 'what things mean.'",
                "why_it_matters": "Visual encoder choice is one of the most impactful architectural decisions in VLAs. Pure language-aligned encoders (CLIP/SigLIP) lose spatial detail needed for precise manipulation. Pure self-supervised encoders (DINOv2/MAE) lack semantic grounding. The dual-encoder pattern appears across the field — understanding why tells you what each encoder family is actually good at.",
            },
            {
                "question": "OpenVLA (7B) outperforms RT-2-X (55B) by 16.5% despite being 7x smaller. What is the most likely explanation?",
                "choices": [
                    "The Llama 2 base model is inherently superior to PaLM-E for robotics",
                    "OpenVLA's training data (Open X-Embodiment) is more diverse and better curated than RT-2-X's data",
                    "OpenVLA's dual visual encoder captures manipulation-relevant features that RT-2-X's single encoder misses",
                    "All of the above likely contribute, but the open question is their relative importance",
                ],
                "correct_index": 3,
                "explanation": "Multiple factors contribute: better visual encoder (dual DINOv2+SigLIP vs. single ViT), improved training data (full Open X-Embodiment), and potentially better fine-tuning recipes. Disentangling these factors is genuinely difficult, and the paper's ablations don't fully isolate each.",
                "why_it_matters": "In ML research, it's tempting to attribute improvements to the paper's novel contribution. But real improvements often come from confounded factors — better data, more compute, engineering details. Training yourself to ask 'what's actually driving the gains?' is essential for reading papers critically rather than just accepting headline numbers.",
            },
            {
                "question": "OpenVLA fine-tunes the entire 7B model on robot data. What risk does this pose compared to freezing the VLM backbone, and how does LoRA fine-tuning mitigate it?",
                "choices": [
                    "Risk: the visual encoder degrades; LoRA: only tunes the language model",
                    "Risk: catastrophic forgetting of web knowledge; LoRA: constrains updates to a low-rank subspace preserving most pre-trained weights",
                    "Risk: overfitting to the training robot; LoRA: adds dropout regularization",
                    "Risk: training instability at 7B scale; LoRA: reduces the effective learning rate",
                ],
                "correct_index": 1,
                "explanation": "Full fine-tuning risks overwriting the VLM's pre-trained knowledge with robot-specific patterns. LoRA constrains parameter updates to low-rank matrices, modifying the model's behavior while preserving most of the original weights — enabling adaptation with 10-20 demos without catastrophic forgetting.",
                "why_it_matters": "The fine-tuning strategy directly determines how well a VLA transfers to new robots — arguably the most practically important property. Full fine-tuning, LoRA, frozen backbone + adapter, co-training — each makes different trade-offs between adaptation capacity and knowledge preservation. This is an active research frontier where OpenVLA, pi0, and others reach different conclusions.",
            },
        ],
    },
    {
        "number": 9,
        "questions": [
            {
                "question": "Diffusion Policy achieves 46.9% average improvement. What property of the diffusion process is most responsible for this gain over regression-based policies (e.g., MSE-trained MLPs)?",
                "choices": [
                    "Diffusion models are deeper networks with more parameters",
                    "The iterative denoising process can represent multimodal action distributions, avoiding mode-averaging",
                    "Diffusion provides built-in data augmentation through noise injection",
                    "Diffusion models learn faster due to the simplified training objective",
                ],
                "correct_index": 1,
                "explanation": "When a task has multiple valid solutions (e.g., going left or right around an obstacle), regression-based policies average the modes, producing an invalid midpoint. Diffusion samples from the full distribution, cleanly selecting one valid mode per rollout.",
                "why_it_matters": "Mode-averaging is arguably the single biggest failure mode in imitation learning. If you train an MLP with MSE loss on bimodal demonstrations, it literally predicts the average — which may be physically impossible (e.g., driving straight into an obstacle that demos go around). Understanding why diffusion solves this tells you when it matters (multimodal tasks) and when simpler methods suffice (unimodal tasks).",
            },
            {
                "question": "Diffusion Policy predicts action *sequences* (chunks) rather than single actions. Why is this critical for manipulation tasks?",
                "choices": [
                    "It reduces the number of forward passes, improving inference speed",
                    "Temporal action consistency prevents jittery oscillations and enables capturing the temporal correlation structure of skilled behavior",
                    "Sequence prediction is required for the diffusion formulation to work mathematically",
                    "It allows the policy to plan further into the future, replacing the need for a planner",
                ],
                "correct_index": 1,
                "explanation": "Single-step prediction with closed-loop feedback often causes oscillation — small observation noise flips the action, then the next observation flips it back. Predicting coherent action chunks and executing them with receding-horizon control maintains temporal smoothness, matching how demonstrations actually look.",
                "why_it_matters": "Action chunking appears in most modern robot policies (ACT, Diffusion Policy, pi0) precisely because single-step policies are jittery. This connects to a deeper issue: the Markov assumption (next action depends only on current state) doesn't hold when demonstrations have temporal coherence. Understanding this helps you diagnose real robot failures and choose appropriate prediction horizons.",
            },
            {
                "question": "The paper compares CNN-based (temporal U-Net) and Transformer-based Diffusion Policy. Under what conditions would you choose the Transformer variant?",
                "choices": [
                    "Always — Transformers are strictly superior for all tasks",
                    "When the observation space includes language conditioning or multiple heterogeneous input modalities",
                    "When training data is very limited and you need parameter efficiency",
                    "When real-time inference under 10ms is required",
                ],
                "correct_index": 1,
                "explanation": "The Transformer variant uses cross-attention to condition on observations, making it natural to incorporate language instructions, multi-view images, or other modalities. The CNN variant is simpler and more sample-efficient for fixed observation spaces but less flexible for conditioning on varied inputs.",
                "why_it_matters": "Architecture selection in practice depends on the conditioning structure: what inputs the policy must attend to. The CNN variant's inductive bias (locality, translation equivariance in time) helps with limited data. The Transformer variant's flexibility with cross-attention is why it became the backbone for VLA-style models that condition on language + vision. Knowing which to use when saves you from both under- and over-engineering.",
            },
        ],
    },
    {
        "number": 31,
        "questions": [
            {
                "question": "DreamerV3 works across 150+ tasks with fixed hyperparameters. What is the core technical challenge this solves that prevents most RL algorithms from generalizing across domains?",
                "choices": [
                    "Different domains require different neural network architectures",
                    "Reward scales, observation distributions, and dynamics vary wildly — standard objectives become numerically unstable or poorly conditioned",
                    "Different domains require different exploration strategies",
                    "Compute requirements vary too much between domains for a single configuration",
                ],
                "correct_index": 1,
                "explanation": "A reward of +1 in Atari and +1000 in DMC makes the same loss function behave completely differently. DreamerV3 addresses this with symlog predictions (scale-invariant targets), free bits (balanced KL), and percentile return normalization — making learning dynamics consistent regardless of the domain's numerical properties.",
                "why_it_matters": "Hyperparameter sensitivity is the dirty secret of RL research. Most papers tune per-environment, making results unreproducible and impractical. DreamerV3's approach — making the algorithm invariant to scale and distribution — is the kind of fundamental engineering that enables real deployment. If you can't run the same config on your robot as in sim, the algorithm isn't actually general.",
            },
            {
                "question": "DreamerV3 learns from 'imagined' trajectories in a world model rather than real interaction. What is the fundamental advantage and risk of this approach?",
                "choices": [
                    "Advantage: unlimited free data; Risk: imagined trajectories may diverge from reality (model exploitation)",
                    "Advantage: faster wall-clock training; Risk: higher memory requirements",
                    "Advantage: no need for a reward function; Risk: requires demonstrations",
                    "Advantage: automatically handles partial observability; Risk: only works in continuous action spaces",
                ],
                "correct_index": 0,
                "explanation": "World model learning generates unlimited training data for the actor-critic at zero environment cost. But if the policy exploits inaccuracies in the world model — finding 'cheats' that work in the model but not reality — performance degrades. DreamerV3 manages this through careful KL balancing and short imagination horizons.",
                "why_it_matters": "This advantage/risk trade-off is the central tension in all model-based RL. It's also why world models are critical for robotics: real robot interaction is expensive and slow, so learning from imagination is enormously valuable — but only if the world model is accurate enough. This directly connects to why NVIDIA's Cosmos and other video world models are being pursued for robotic pre-training.",
            },
            {
                "question": "DreamerV3 uses the RSSM (Recurrent State-Space Model) as its world model. Why use a recurrent latent state rather than just predicting next observations directly?",
                "choices": [
                    "Recurrent states are computationally cheaper than pixel prediction",
                    "The compact latent state captures sufficient statistics of history while being tractable for long-horizon imagination and actor-critic learning",
                    "RSSMs can handle multi-modal observations like RGB + proprioception",
                    "Direct observation prediction would require generative adversarial training",
                ],
                "correct_index": 1,
                "explanation": "Predicting raw pixels is high-dimensional and noisy. The RSSM compresses observations into a compact latent state that retains decision-relevant information while discarding irrelevant detail. This makes imagining 15-step trajectories tractable and gives the actor-critic a clean, informative state representation to learn from.",
                "why_it_matters": "The choice between latent-space vs. observation-space world models is a key design axis. Latent models (RSSM, TDMPC2) are efficient but may miss visually important details. Observation-space models (video prediction, Cosmos) preserve visual fidelity but are expensive. Understanding this trade-off helps you evaluate the recent wave of video world models and whether they actually improve downstream policy learning.",
            },
        ],
    },
    {
        "number": 44,
        "questions": [
            {
                "question": "Open X-Embodiment shows positive transfer across 22 robot types. What is the most surprising implication of this result?",
                "choices": [
                    "Robots with completely different morphologies share useful low-level motor primitives",
                    "There exist transferable representations of manipulation behavior that generalize across different observation spaces, action spaces, and embodiment morphologies",
                    "Having more data always helps regardless of domain mismatch",
                    "Language conditioning is the key to cross-embodiment transfer",
                ],
                "correct_index": 1,
                "explanation": "The non-obvious finding is that despite different cameras, grippers, arm kinematics, and action definitions, there is a shared structure in manipulation behavior that a single model can capture. This suggests manipulation has universal principles that transcend specific hardware — analogous to how language structure transfers across domains.",
                "why_it_matters": "This result shapes the field's entire strategy. If cross-embodiment transfer works, we should invest in diverse data collection (many robots, many labs). If it doesn't, each robot needs its own dataset. Open X-Embodiment provided strong evidence for the former, launching a wave of cross-embodiment models (Octo, HPT, CrossFormer) and making 'train on everything' the default approach.",
            },
            {
                "question": "The Open X-Embodiment dataset combines data from 21 different institutions. What is the biggest practical challenge this creates that doesn't exist with single-lab datasets?",
                "choices": [
                    "Different labs use different programming languages",
                    "Heterogeneous data formats, action definitions, camera setups, and annotation standards that must be unified without losing information",
                    "Network bandwidth for transferring large datasets",
                    "Licensing and IP restrictions on sharing data",
                ],
                "correct_index": 1,
                "explanation": "Different labs define actions differently (joint angles vs. end-effector poses, absolute vs. relative), use different camera placements and resolutions, and have inconsistent task annotations. Unifying this without losing information or introducing systematic biases is an ongoing engineering and research challenge.",
                "why_it_matters": "Data standardization is an unglamorous but critical bottleneck for robotics. Unlike NLP (text is text) or vision (images are images), robot data has no standard format. Action spaces vary by robot, observations vary by sensor setup. This is why efforts like RLDS (the data format used by Open X-Embodiment) matter — and why evaluating cross-embodiment claims requires understanding what normalization was applied.",
            },
            {
                "question": "RT-X trained on Open X-Embodiment outperforms single-robot policies. But does this guarantee positive transfer for any new robot added to the dataset?",
                "choices": [
                    "Yes — more data always helps due to the blessing of dimensionality",
                    "No — negative transfer is possible if the new robot's data distribution conflicts with existing data or the model capacity is insufficient to represent all embodiments",
                    "Yes — the Transformer architecture prevents negative transfer through attention masking",
                    "No — but only if the new robot uses a different action space",
                ],
                "correct_index": 1,
                "explanation": "Positive transfer is not guaranteed. Adding data from a very different embodiment could hurt performance on existing robots if the model conflates incompatible behaviors, or if capacity is insufficient. The paper shows average positive transfer but acknowledges some robot-specific metrics decrease.",
                "why_it_matters": "Negative transfer is a real risk in multi-task and multi-robot learning. The 'average improvement' headline can mask that some specific robots got worse. This is critical for deployment: if you're building a product on one robot, you need to know whether adding diverse data helps your robot specifically, not just the average across all robots.",
            },
        ],
    },
    {
        "number": 66,
        "questions": [
            {
                "question": "SayCan factors the action selection as p(skill|instruction) * p(success|skill, state). Why is the multiplicative combination essential rather than just using the LLM's p(skill|instruction) alone?",
                "choices": [
                    "The LLM alone doesn't generate valid robot commands",
                    "The LLM has no model of the physical world — it might suggest semantically correct but physically infeasible actions given the current state",
                    "Multiplication is computationally faster than other combination methods",
                    "The affordance function improves the LLM's language understanding",
                ],
                "correct_index": 1,
                "explanation": "An LLM might suggest 'pick up the cup' when the cup is across the room and unreachable. The affordance function (trained value function) gates this: it knows whether the robot can actually succeed at that skill right now. Without this grounding, the LLM produces coherent but infeasible plans.",
                "why_it_matters": "This is the core insight behind the entire 'LLM + robot' paradigm: LLMs know what makes sense semantically, but not what's physically possible. Every system combining LLMs with robots must solve this grounding problem somehow. SayCan's approach (learned value functions as affordances) is one solution; others use VLMs, vision-based feasibility checks, or embodied simulation. Understanding the problem lets you evaluate all of them.",
            },
            {
                "question": "SayCan requires a pre-trained library of manipulation skills with associated value functions. What fundamental limitation does this impose?",
                "choices": [
                    "The system can only compose behaviors from its fixed skill library — it cannot learn new motor primitives at inference time",
                    "Value functions are too slow to evaluate in real time",
                    "The skill library must be manually programmed by roboticists",
                    "SayCan only works with mobile manipulation robots",
                ],
                "correct_index": 0,
                "explanation": "SayCan's expressiveness is bounded by its skill library. If no skill exists for a required subtask, the system fails — no matter how good the LLM's plan is. This is why end-to-end VLAs (RT-2, pi0) were seen as the next step: they can generate arbitrary actions without a fixed skill vocabulary.",
                "why_it_matters": "This highlights the modularity vs. end-to-end debate in robotics. SayCan (modular: LLM planner + skill library) is interpretable and composable but rigid. End-to-end VLAs are flexible but opaque. Most deployed systems still use modular approaches because they're debuggable — but the field is moving toward end-to-end as VLAs improve. Understanding this trade-off is essential for system design.",
            },
            {
                "question": "SayCan uses the LLM to score all available skills at each step. How does this scale as the skill library grows, and what does this imply?",
                "choices": [
                    "Scoring is O(1) regardless of library size since the LLM processes all skills in parallel",
                    "Linearly — each skill requires an LLM query, making large skill libraries expensive; this motivates hierarchical skill organization",
                    "Logarithmically — the LLM uses binary search over the skill space",
                    "It doesn't scale at all — SayCan only works with fewer than 20 skills",
                ],
                "correct_index": 1,
                "explanation": "Each skill must be independently scored by the LLM and the affordance function, giving O(n) cost. With hundreds of skills, this becomes a bottleneck. This motivates hierarchical planning approaches where skills are organized into categories, reducing the search space at each level.",
                "why_it_matters": "Scalability of the planning loop determines whether an approach works in real kitchens (hundreds of possible actions) vs. toy benchmarks (10-20 actions). This is a practical concern that's often overlooked in paper evaluations. Hierarchical planning, skill abstraction, and retrieval-based approaches all address this — and the choice matters for deployment.",
            },
        ],
    },
    {
        "number": 5,
        "questions": [
            {
                "question": "Pi0 uses flow matching instead of autoregressive token prediction for actions. What is the key technical advantage of flow matching over diffusion for action generation?",
                "choices": [
                    "Flow matching requires fewer denoising steps for equivalent quality, enabling faster inference",
                    "Flow matching doesn't require neural networks",
                    "Flow matching can only generate discrete actions",
                    "Flow matching is identical to diffusion but with a different name",
                ],
                "correct_index": 0,
                "explanation": "Flow matching learns a direct transport map from noise to actions (straight-line flows in most cases), requiring fewer steps than DDPM's iterative denoising. This is critical for real-time robot control where you need actions at 5-50Hz and can't afford 50+ denoising steps.",
                "why_it_matters": "Inference speed directly determines which robots can use a model. A method requiring 1 second per action is useless for dexterous manipulation (50Hz control). Flow matching's efficiency over diffusion is why Physical Intelligence chose it for pi0 — and why the field is shifting from DDPM to flow-based action generation. The speed-quality trade-off in generative models is a recurring theme across VLA papers.",
            },
            {
                "question": "Pi0 follows a 'pre-train then post-train' recipe analogous to LLMs. What does 'post-training' mean in the context of a robot policy, and why is it necessary?",
                "choices": [
                    "Post-training is RLHF applied to robot trajectories",
                    "Post-training fine-tunes the pre-trained generalist model on specific high-quality task data to achieve expert-level performance on target tasks",
                    "Post-training converts the model from simulation to real-world",
                    "Post-training replaces the VLM backbone with a smaller model for deployment",
                ],
                "correct_index": 1,
                "explanation": "Pre-training on diverse cross-embodiment data gives broad capabilities. Post-training on curated, high-quality data for specific tasks (e.g., laundry folding) refines this into expert performance — similar to how LLMs go from general text completion to helpful assistants via SFT/RLHF.",
                "why_it_matters": "The pre-train/post-train paradigm is becoming the standard recipe in robotics, borrowed from LLMs. Understanding it lets you predict the workflow for deploying any VLA: pre-train on broad data for generalization, post-train on your specific task for precision. The quality and quantity of post-training data often matters more than the base model's capabilities.",
            },
            {
                "question": "Pi0 was evaluated on tasks like laundry folding and box assembly that previous VLAs couldn't solve. What property of these tasks makes them particularly challenging for prior VLA architectures?",
                "choices": [
                    "They require bimanual coordination with complex contact dynamics and long action horizons that expose the limitations of discrete token-based action representations",
                    "They require visual reasoning about 3D geometry",
                    "They require processing natural language instructions",
                    "They need faster inference than prior models could provide",
                ],
                "correct_index": 0,
                "explanation": "Laundry folding requires smooth bimanual coordination over long horizons with continuous contact — precisely where discrete token actions fail (quantization artifacts) and where flow matching's continuous, multimodal output shines. These tasks are the 'stress test' that differentiates action representations.",
                "why_it_matters": "Task selection in evaluations reveals (or hides) a model's limitations. Simple pick-and-place tasks don't differentiate action representations — any method works. Complex dexterous tasks expose the differences. When reading VLA papers, always ask: are the evaluation tasks hard enough to distinguish the contribution from prior work?",
            },
        ],
    },
    {
        "number": 21,
        "questions": [
            {
                "question": "Gemini Robotics builds a VLA on top of a general-purpose foundation model (Gemini 2.0) rather than training a robotics model from scratch. What is the strongest argument FOR this approach?",
                "choices": [
                    "Gemini 2.0 already has robotics data in its pre-training set",
                    "The world knowledge, reasoning, and multi-modal understanding in a frontier model provides a richer prior than any robotics-only dataset could",
                    "It's cheaper to fine-tune an existing model than train from scratch",
                    "Foundation models have better optimizers and training infrastructure",
                ],
                "correct_index": 1,
                "explanation": "A frontier model has been trained on trillions of tokens spanning science, engineering, physics, language, and vision. This represents a far richer prior about how the world works than any robotics dataset. The bet is that this knowledge — even though not explicitly robotic — provides a better starting point than random initialization.",
                "why_it_matters": "This is the central philosophical debate in robot learning: should we build domain-specific models with strong robotic inductive biases, or co-opt the most general models available? Gemini Robotics represents one extreme; classical control represents the other. Most practical systems will likely be somewhere between — and your position on this spectrum determines your entire research strategy.",
            },
            {
                "question": "Gemini Robotics-ER (Embodied Reasoning) separates spatial/temporal understanding from direct action generation. Why might a separate reasoning module help compared to a single end-to-end VLA?",
                "choices": [
                    "Separate modules can be trained on different datasets — spatial reasoning from vision data, action generation from robot data — avoiding the bottleneck of scarce robot trajectories",
                    "Separate modules are always faster at inference",
                    "End-to-end models can't process multiple camera views",
                    "Embodied reasoning doesn't require neural networks",
                ],
                "correct_index": 0,
                "explanation": "Spatial reasoning (object detection, 3D understanding, temporal tracking) can be trained on abundant vision data. Action generation requires scarce robot data. Separating them allows each component to train on its natural data distribution, then compose at inference time.",
                "why_it_matters": "Data scarcity is robotics' biggest bottleneck. Architectures that can leverage abundant non-robot data (vision, language) for sub-problems while using scarce robot data only where necessary have a fundamental advantage. This principle — match each component to its most abundant data source — is a powerful lens for evaluating any multi-module robot system.",
            },
            {
                "question": "Google has an advantage in building Gemini Robotics that most labs don't. What is it, and why does it matter for evaluating the paper's results?",
                "choices": [
                    "Access to Google's TPU infrastructure for training at massive scale",
                    "Access to both a frontier foundation model and large-scale robot data collection (from Everyday Robots), making results hard to reproduce independently",
                    "Google's superior software engineering culture",
                    "Access to the latest GPU architectures",
                ],
                "correct_index": 1,
                "explanation": "Gemini Robotics requires both Gemini 2.0 (a model only Google has) and extensive robot data. Independent researchers can't reproduce the foundation model or the data collection. This means the approach's generality is hard to verify — it might work because of Gemini's specific capabilities rather than the general recipe.",
                "why_it_matters": "Reproducibility is a cornerstone of science. When evaluating papers from large labs, always ask: can this be replicated? If the results depend on proprietary models and infrastructure, the contribution is the idea (which may or may not transfer to open models), not a reproducible system. This is why open-source efforts like OpenVLA and Octo are so valuable — they let the community verify and build on claims.",
            },
        ],
    },
    {
        "number": 74,
        "questions": [
            {
                "question": "Decision Transformer conditions on desired return-to-go to control behavior. What fundamental assumption does this make about the training data?",
                "choices": [
                    "The training data must come from an optimal policy",
                    "The training data must contain trajectories spanning a range of returns, so the model can learn the mapping from desired return to corresponding behavior quality",
                    "The training data must be collected online with exploration",
                    "The training data must include reward labels from a human evaluator",
                ],
                "correct_index": 1,
                "explanation": "If all training trajectories have similar returns, the model can't distinguish between 'aiming for high return' and 'aiming for low return.' It needs a spread of sub-optimal to near-optimal behavior to learn that higher return-to-go conditioning leads to better actions. This is why it works well with mixed-quality offline datasets.",
                "why_it_matters": "This connects to a broader question in offline RL: what properties must the dataset have for learning to work? Decision Transformer's requirement (diverse returns) differs from TD-learning methods (need good state-action coverage). Understanding what each method needs from data helps you choose the right algorithm for your available dataset — a very practical consideration.",
            },
            {
                "question": "Decision Transformer frames RL as sequence modeling rather than dynamic programming. What is the biggest limitation this introduces compared to TD-learning methods like CQL?",
                "choices": [
                    "It cannot process image observations",
                    "It can only produce actions as good as the best trajectories in the training data — it cannot 'stitch' together good sub-sequences from different trajectories to exceed any single demonstration",
                    "It requires more training data",
                    "It cannot handle continuous action spaces",
                ],
                "correct_index": 1,
                "explanation": "TD-learning can combine the best parts of different trajectories through value function generalization (trajectory stitching). Decision Transformer, lacking a value function, is bounded by the best complete trajectory in the data. In sparse-reward long-horizon tasks, this is a significant limitation.",
                "why_it_matters": "Trajectory stitching is the key theoretical advantage of value-based offline RL. Decision Transformer trades this for simplicity and stability. Knowing this trade-off is essential: if your offline dataset has no single good trajectory but has good sub-sequences, TD-methods will likely outperform DT. If your data has some expert demonstrations, DT's simplicity wins. This is one of the most practically important distinctions in offline RL.",
            },
            {
                "question": "Decision Transformer uses a causal Transformer (GPT-style). Why is the causal masking important rather than using bidirectional attention (BERT-style)?",
                "choices": [
                    "Causal masking is faster to train",
                    "The model must generate actions autoregressively at test time — it can only condition on past and present context, not future states, matching the causal structure of decision-making",
                    "Bidirectional attention would cause information leakage from the reward signal",
                    "Causal masking provides better gradient flow during training",
                ],
                "correct_index": 1,
                "explanation": "Decision-making is inherently causal: you choose actions based on past observations, not future ones. Causal masking ensures the model respects this temporal structure during both training and inference, enabling autoregressive generation of (return, state, action) sequences.",
                "why_it_matters": "The choice between causal and bidirectional attention reflects a deep modeling assumption. Causal models (GPT, Decision Transformer) match sequential decision-making. Bidirectional models (BERT) are better for understanding complete sequences. In robotics, this distinction matters: planning (considering the whole trajectory) vs. reactive control (acting on current state) may benefit from different attention patterns.",
            },
        ],
    },
]


def main():
    client = get_admin_client()

    # Delete all existing questions
    client.table("quiz_attempts").delete().neq("id", 0).execute()
    client.table("questions").delete().neq("id", 0).execute()
    print("Cleared existing questions and attempts")

    total = 0
    for paper in PAPERS:
        num = paper["number"]
        for q in paper["questions"]:
            client.table("questions").insert({
                "paper_number": num,
                "question": q["question"],
                "choices": json.dumps(q["choices"]),
                "correct_index": q["correct_index"],
                "explanation": q["explanation"],
                "why_it_matters": q["why_it_matters"],
            }).execute()
            total += 1
        print(f"  #{num}: {len(paper['questions'])} questions")

    print(f"\nDone: {total} expert-level questions seeded across {len(PAPERS)} papers")


if __name__ == "__main__":
    main()
