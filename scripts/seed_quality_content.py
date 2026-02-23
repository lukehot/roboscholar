#!/usr/bin/env python3
"""Seed quality summaries and quiz questions for 10 key papers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_admin_client

PAPERS = [
    {
        "number": 1,
        "summary": (
            "RT-1 (Robotics Transformer) is a scalable model for real-world robot control, "
            "trained on 130k demonstrations across 700+ tasks collected from 13 robots over 17 months. "
            "The architecture uses a FiLM-conditioned EfficientNet for image encoding, TokenLearner "
            "for token compression, and a Transformer backbone — totaling only 35M parameters while "
            "running at 3Hz inference. RT-1 demonstrates that scaling up data diversity (not just model size) "
            "leads to significant improvements in generalization to new tasks, objects, and environments. "
            "Key findings include strong zero-shot generalization to novel instructions and robustness "
            "to background changes and distractors."
        ),
        "questions": [
            {
                "question": "What is the core architectural backbone of RT-1?",
                "choices": [
                    "FiLM-conditioned EfficientNet + TokenLearner + Transformer",
                    "ResNet-50 + LSTM + MLP action head",
                    "Vision Transformer (ViT) + GPT-2 decoder",
                    "CLIP encoder + diffusion policy head",
                ],
                "correct_index": 0,
                "explanation": "RT-1 uses a FiLM-conditioned EfficientNet to process images, TokenLearner for token compression, and a Transformer backbone, totaling 35M parameters.",
            },
            {
                "question": "How many robot demonstrations were used to train RT-1?",
                "choices": ["13,000", "130,000", "1.3 million", "700"],
                "correct_index": 1,
                "explanation": "RT-1 was trained on 130k real-world demonstrations collected from 13 robots over 17 months, covering 700+ tasks.",
            },
            {
                "question": "What was RT-1's key insight about scaling?",
                "choices": [
                    "Larger models always perform better",
                    "Scaling data diversity matters more than model size",
                    "Pre-training on ImageNet is essential",
                    "Simulation data is more important than real data",
                ],
                "correct_index": 1,
                "explanation": "RT-1 showed that scaling up data diversity (more tasks, objects, environments) led to significant generalization improvements, even with a relatively small 35M parameter model.",
            },
            {
                "question": "At what frequency does RT-1 run inference for robot control?",
                "choices": ["1 Hz", "3 Hz", "10 Hz", "30 Hz"],
                "correct_index": 1,
                "explanation": "RT-1 operates at 3Hz inference, meaning it produces a new action 3 times per second.",
            },
        ],
    },
    {
        "number": 2,
        "summary": (
            "RT-2 introduces the concept of Vision-Language-Action (VLA) models — taking large "
            "vision-language models (VLMs) pre-trained on internet-scale data and co-fine-tuning them "
            "on robotic trajectory data. Actions are represented as text tokens (e.g., 'move arm 0.1 0.2 0.3'), "
            "allowing the model to leverage web knowledge for robotic control. RT-2 demonstrates emergent "
            "capabilities: it can perform chain-of-thought reasoning, understand novel semantic concepts "
            "(e.g., 'pick up the object you would use as an improvised hammer' → picks up a rock), "
            "and generalize to tasks never seen during robot training. This paper established the VLA "
            "paradigm that subsequent work builds upon."
        ),
        "questions": [
            {
                "question": "How does RT-2 represent robot actions?",
                "choices": [
                    "As continuous vectors in a learned action space",
                    "As text tokens co-trained with language tasks",
                    "As discrete bins from a fixed vocabulary",
                    "As diffusion noise predictions",
                ],
                "correct_index": 1,
                "explanation": "RT-2 represents actions as text tokens (strings of numbers), allowing joint training with internet-scale vision-language data.",
            },
            {
                "question": "What emergent capability does RT-2 demonstrate that RT-1 cannot?",
                "choices": [
                    "Faster inference speed",
                    "Multi-robot coordination",
                    "Semantic reasoning about novel concepts (e.g., improvised tools)",
                    "Learning from human video demonstrations",
                ],
                "correct_index": 2,
                "explanation": "RT-2 shows emergent semantic reasoning — for example, understanding that a rock could serve as an improvised hammer — by leveraging web knowledge from VLM pre-training.",
            },
            {
                "question": "What paradigm did RT-2 establish for robotics?",
                "choices": [
                    "Reinforcement Learning from Human Feedback (RLHF)",
                    "Vision-Language-Action (VLA) models",
                    "Sim-to-real transfer",
                    "Behavior cloning with data augmentation",
                ],
                "correct_index": 1,
                "explanation": "RT-2 introduced the VLA paradigm: fine-tuning vision-language models on robotic data so they can directly output actions while retaining web knowledge.",
            },
        ],
    },
    {
        "number": 4,
        "summary": (
            "OpenVLA is a 7-billion parameter open-source Vision-Language-Action model built on "
            "Llama 2 with a dual visual encoder (DINOv2 + SigLIP). Trained on 970k real-world robot "
            "demonstrations from the Open X-Embodiment dataset, OpenVLA outperforms the much larger "
            "RT-2-X (55B parameters) by 16.5% absolute on generalization benchmarks — while being 7x smaller. "
            "The model supports fine-tuning on new robot setups with as few as 10-20 demonstrations using LoRA. "
            "As an open-source release, OpenVLA democratized VLA research by providing weights, training code, "
            "and fine-tuning recipes that previously required Google-scale infrastructure."
        ),
        "questions": [
            {
                "question": "What visual encoders does OpenVLA use?",
                "choices": [
                    "CLIP ViT-L only",
                    "DINOv2 + SigLIP dual encoder",
                    "ResNet-50 + EfficientNet",
                    "ViT-G with MAE pre-training",
                ],
                "correct_index": 1,
                "explanation": "OpenVLA uses a dual visual encoder combining DINOv2 (self-supervised) and SigLIP (language-aligned) for complementary visual representations.",
            },
            {
                "question": "How does OpenVLA compare to RT-2-X in size and performance?",
                "choices": [
                    "Larger model, worse performance",
                    "Same size, same performance",
                    "7x smaller, 16.5% better on generalization",
                    "7x larger, 16.5% better on generalization",
                ],
                "correct_index": 2,
                "explanation": "OpenVLA (7B params) outperforms RT-2-X (55B params) by 16.5% absolute on generalization benchmarks while being 7x smaller.",
            },
            {
                "question": "What technique does OpenVLA use for efficient fine-tuning on new robots?",
                "choices": [
                    "Full model retraining",
                    "LoRA (Low-Rank Adaptation)",
                    "Prompt tuning only",
                    "Knowledge distillation",
                ],
                "correct_index": 1,
                "explanation": "OpenVLA supports LoRA fine-tuning, allowing adaptation to new robot setups with as few as 10-20 demonstrations.",
            },
        ],
    },
    {
        "number": 9,
        "summary": (
            "Diffusion Policy reformulates robot behavior generation as conditional denoising diffusion "
            "over action sequences. Instead of predicting a single action, it learns the full distribution "
            "of possible action trajectories and samples from it. The paper presents two architectures: "
            "CNN-based (using 1D temporal convolutions) and Transformer-based (using cross-attention between "
            "noise and observation tokens). Key advantages include natural handling of multimodal action "
            "distributions (multiple valid ways to do a task), stable training, and the ability to predict "
            "action sequences rather than single steps. Evaluated across 15 tasks from 4 benchmarks, "
            "Diffusion Policy achieves 46.9% average improvement over prior state-of-the-art."
        ),
        "questions": [
            {
                "question": "What is the fundamental insight behind Diffusion Policy?",
                "choices": [
                    "Robot actions should be predicted one at a time for maximum precision",
                    "Action generation can be framed as denoising diffusion over action sequences",
                    "Reinforcement learning is always better than behavior cloning",
                    "Actions should be discretized into a fixed vocabulary",
                ],
                "correct_index": 1,
                "explanation": "Diffusion Policy treats action generation as conditional denoising diffusion, learning the full distribution over action trajectories rather than predicting single actions.",
            },
            {
                "question": "Why is diffusion particularly well-suited for robot policy learning?",
                "choices": [
                    "It requires less training data",
                    "It naturally handles multimodal action distributions",
                    "It runs faster than other methods",
                    "It doesn't need visual observations",
                ],
                "correct_index": 1,
                "explanation": "Diffusion naturally represents multimodal distributions — when there are multiple valid ways to complete a task, diffusion can capture all modes rather than averaging them.",
            },
            {
                "question": "What two architecture variants does Diffusion Policy propose?",
                "choices": [
                    "RNN-based and MLP-based",
                    "CNN-based (temporal convolutions) and Transformer-based (cross-attention)",
                    "ResNet-based and ViT-based",
                    "LSTM-based and GRU-based",
                ],
                "correct_index": 1,
                "explanation": "Diffusion Policy presents CNN-based (1D temporal convolutions) and Transformer-based (cross-attention between noise and observation tokens) architectures.",
            },
            {
                "question": "By how much did Diffusion Policy improve over prior state-of-the-art on average?",
                "choices": ["12.3%", "25.1%", "46.9%", "73.5%"],
                "correct_index": 2,
                "explanation": "Across 15 tasks from 4 benchmarks, Diffusion Policy achieved a 46.9% average improvement over prior SOTA methods.",
            },
        ],
    },
    {
        "number": 31,
        "summary": (
            "DreamerV3 is a general reinforcement learning algorithm based on world models that masters "
            "150+ diverse tasks with a single set of hyperparameters — no per-domain tuning. It is the first "
            "algorithm to collect diamonds in Minecraft from scratch without human data or curriculum. "
            "The architecture uses RSSM (Recurrent State-Space Model) as the world model, learning a "
            "compact latent representation of environment dynamics. An actor-critic pair then learns entirely "
            "from imagined trajectories in the learned world model. Key techniques include symlog predictions "
            "for scale-invariant learning, free bits for KL balancing, and percentile-based return normalization. "
            "These robustness techniques eliminate the need for hyperparameter tuning across vastly different domains."
        ),
        "questions": [
            {
                "question": "What makes DreamerV3 unique compared to prior RL algorithms?",
                "choices": [
                    "It uses the largest neural network ever trained",
                    "It works across 150+ diverse tasks with a single hyperparameter configuration",
                    "It only works in simulation environments",
                    "It requires human demonstrations for each new task",
                ],
                "correct_index": 1,
                "explanation": "DreamerV3's key contribution is generality: one set of hyperparameters works across 150+ tasks spanning continuous control, Atari, Minecraft, and more.",
            },
            {
                "question": "How does DreamerV3's actor-critic learn without interacting with the real environment?",
                "choices": [
                    "It uses offline data from expert demonstrations",
                    "It learns from imagined trajectories in the learned world model",
                    "It copies policies from other agents",
                    "It uses evolutionary strategies",
                ],
                "correct_index": 1,
                "explanation": "DreamerV3 learns a world model (RSSM), then trains the actor-critic entirely from imagined trajectories within that model — dreaming about what would happen.",
            },
            {
                "question": "What landmark achievement did DreamerV3 accomplish in Minecraft?",
                "choices": [
                    "Building a house from scratch",
                    "Defeating the Ender Dragon",
                    "Collecting diamonds from scratch without human data",
                    "Completing all achievements in the game",
                ],
                "correct_index": 2,
                "explanation": "DreamerV3 was the first algorithm to collect diamonds in Minecraft from scratch — a long-horizon sparse-reward task — without any human data or handcrafted curriculum.",
            },
        ],
    },
    {
        "number": 44,
        "summary": (
            "The Open X-Embodiment project assembled the largest open robot learning dataset: data from "
            "22 different robot embodiments across 21 institutions, covering 527 skills and 160,266 tasks. "
            "The paper demonstrates that training a single policy (RT-X) on this diverse cross-embodiment "
            "data leads to positive transfer — the unified model outperforms policies trained on any single "
            "robot's data alone. This is analogous to how large language models benefit from diverse text data. "
            "The dataset includes single-arm robots, bimanual systems, quadrupeds, and mobile manipulators, "
            "establishing cross-embodiment transfer as a viable paradigm for scaling robot learning."
        ),
        "questions": [
            {
                "question": "How many different robot embodiments contributed data to Open X-Embodiment?",
                "choices": ["5", "12", "22", "50"],
                "correct_index": 2,
                "explanation": "Open X-Embodiment collected data from 22 different robot embodiments across 21 institutions.",
            },
            {
                "question": "What key finding did the Open X-Embodiment project demonstrate?",
                "choices": [
                    "Each robot needs its own specialized model",
                    "Cross-embodiment training shows positive transfer, outperforming single-robot policies",
                    "Simulation data is sufficient for real robot deployment",
                    "Smaller datasets produce better generalization",
                ],
                "correct_index": 1,
                "explanation": "The RT-X model trained on diverse cross-embodiment data outperformed policies trained on any single robot's data alone, demonstrating positive transfer.",
            },
            {
                "question": "What analogy does Open X-Embodiment draw to explain cross-embodiment benefits?",
                "choices": [
                    "Like transfer learning in computer vision from ImageNet",
                    "Like how large language models benefit from diverse text data",
                    "Like ensemble methods combining multiple weak learners",
                    "Like curriculum learning in education",
                ],
                "correct_index": 1,
                "explanation": "The paper draws an analogy to LLMs: just as language models improve by training on diverse text from many sources, robot policies improve from diverse embodiment data.",
            },
        ],
    },
    {
        "number": 66,
        "summary": (
            "SayCan bridges the gap between large language models (LLMs) and physical robot capabilities. "
            "LLMs have broad semantic knowledge but lack grounding in what a robot can actually do. SayCan "
            "solves this by combining two signals: the LLM scores how useful each available skill is for the "
            "current instruction (task grounding), while learned affordance/value functions score how likely "
            "each skill is to succeed in the current physical state (world grounding). Formally: "
            "p(skill|instruction, state) ~ p(skill|instruction) * p(success|skill, state). "
            "This allows a mobile robot to execute long-horizon tasks like 'I spilled my drink, can you help?' "
            "by chaining together feasible primitive skills in a semantically meaningful order."
        ),
        "questions": [
            {
                "question": "What problem does SayCan solve?",
                "choices": [
                    "Training robots from scratch without any data",
                    "Grounding LLM knowledge in what a robot can physically do",
                    "Generating realistic robot simulations",
                    "Learning visual representations for manipulation",
                ],
                "correct_index": 1,
                "explanation": "SayCan grounds LLMs in physical reality by combining semantic knowledge (what's useful) with affordance functions (what's feasible).",
            },
            {
                "question": "How does SayCan combine task grounding and world grounding?",
                "choices": [
                    "It trains a single end-to-end model",
                    "It multiplies LLM skill scoring with affordance/value function scoring",
                    "It uses reinforcement learning to learn the combination",
                    "It alternates between LLM planning and random exploration",
                ],
                "correct_index": 1,
                "explanation": "SayCan multiplies two probabilities: p(skill|instruction) from the LLM (task grounding) and p(success|skill, state) from value functions (world grounding).",
            },
            {
                "question": "What type of tasks can SayCan handle that simple skill execution cannot?",
                "choices": [
                    "Single-step pick-and-place tasks",
                    "Long-horizon multi-step tasks requiring semantic reasoning",
                    "High-speed manipulation tasks",
                    "Tasks requiring force feedback",
                ],
                "correct_index": 1,
                "explanation": "SayCan chains together feasible primitive skills in a semantically meaningful order, handling complex instructions like 'I spilled my drink, can you help?'",
            },
        ],
    },
    {
        "number": 5,
        "summary": (
            "Pi-Zero (pi0) from Physical Intelligence introduces a novel VLA architecture combining a "
            "pre-trained vision-language model (VLM) with flow matching for continuous action generation. "
            "Unlike prior VLAs that discretize actions into tokens, pi0 uses flow matching to produce smooth, "
            "continuous action trajectories — better suited for dexterous manipulation. The model is pre-trained "
            "on diverse cross-embodiment data (single-arm, dual-arm, mobile manipulators) and then post-trained "
            "on specific tasks, following a recipe analogous to LLM pre-training/fine-tuning. Pi-Zero demonstrates "
            "strong results on complex dexterous tasks including laundry folding, table clearing, and box assembly "
            "that were previously unsolved by generalist models."
        ),
        "questions": [
            {
                "question": "How does pi0 generate actions differently from RT-2?",
                "choices": [
                    "It uses reinforcement learning instead of imitation",
                    "It uses flow matching for continuous actions instead of discretizing into tokens",
                    "It generates actions from text descriptions",
                    "It uses a separate action prediction head for each robot",
                ],
                "correct_index": 1,
                "explanation": "While RT-2 discretizes actions into text tokens, pi0 uses flow matching to produce smooth continuous action trajectories, better suited for dexterous control.",
            },
            {
                "question": "What training recipe does pi0 follow?",
                "choices": [
                    "End-to-end training from scratch on each task",
                    "Pre-training on diverse data, then post-training on specific tasks (like LLMs)",
                    "Self-supervised learning on video only",
                    "Reinforcement learning with reward shaping",
                ],
                "correct_index": 1,
                "explanation": "Pi0 follows a pre-training/post-training recipe analogous to LLMs: broad pre-training on diverse cross-embodiment data, then task-specific post-training.",
            },
            {
                "question": "Which of these dexterous tasks did pi0 demonstrate success on?",
                "choices": [
                    "Only simple pick-and-place",
                    "Laundry folding, table clearing, and box assembly",
                    "Autonomous driving",
                    "Bipedal walking and running",
                ],
                "correct_index": 1,
                "explanation": "Pi0 showed strong results on complex dexterous tasks including laundry folding, table clearing, and box assembly — tasks previously unsolved by generalist models.",
            },
        ],
    },
    {
        "number": 21,
        "summary": (
            "Gemini Robotics from Google DeepMind builds on the Gemini 2.0 foundation model to create "
            "a family of models for physical-world AI. It includes Gemini Robotics (a VLA for direct robot "
            "control) and Gemini Robotics-ER (for embodied reasoning with spatial and temporal understanding). "
            "The system demonstrates strong capabilities in 2D/3D object detection, pointing, multi-view "
            "correspondence, and can follow complex natural language instructions. The architecture represents "
            "the trend of adapting the most capable general-purpose AI models for robotics, rather than "
            "building robotics-specific models from scratch. The paper emphasizes responsible development "
            "practices for deploying AI in the physical world."
        ),
        "questions": [
            {
                "question": "What foundation model is Gemini Robotics built upon?",
                "choices": [
                    "GPT-4",
                    "Gemini 2.0",
                    "PaLM 2",
                    "LLaMA 3",
                ],
                "correct_index": 1,
                "explanation": "Gemini Robotics is built on Google DeepMind's Gemini 2.0 foundation model, adapted for physical-world control.",
            },
            {
                "question": "What are the two main model variants in the Gemini Robotics family?",
                "choices": [
                    "Gemini-Small and Gemini-Large",
                    "Gemini Robotics (VLA) and Gemini Robotics-ER (embodied reasoning)",
                    "Gemini-Sim and Gemini-Real",
                    "Gemini-Vision and Gemini-Action",
                ],
                "correct_index": 1,
                "explanation": "The family includes Gemini Robotics (a VLA for direct control) and Gemini Robotics-ER (for embodied reasoning with spatial/temporal understanding).",
            },
            {
                "question": "What trend does Gemini Robotics represent in the field?",
                "choices": [
                    "Building small task-specific models for each robot",
                    "Adapting the most capable general AI models for robotics",
                    "Replacing neural networks with classical control",
                    "Focusing exclusively on simulation-to-real transfer",
                ],
                "correct_index": 1,
                "explanation": "Gemini Robotics represents the trend of taking the most capable general-purpose AI models and adapting them for robotics, rather than building robotics-specific models from scratch.",
            },
        ],
    },
    {
        "number": 74,
        "summary": (
            "Decision Transformer reframes reinforcement learning as sequence modeling. Instead of "
            "learning value functions or policy gradients, it trains a causal Transformer on sequences of "
            "(return-to-go, state, action) tokens. At test time, you condition on a desired return and the "
            "model autoregressively generates actions to achieve it. This simple approach matches or exceeds "
            "specialized offline RL algorithms (like CQL) on Atari, OpenAI Gym, and Key-to-Door tasks. "
            "The key insight is that Transformers' self-attention naturally handles credit assignment over "
            "long horizons — a problem that temporal-difference methods struggle with. Decision Transformer "
            "sparked a line of work applying sequence modeling to decision-making problems."
        ),
        "questions": [
            {
                "question": "How does Decision Transformer reframe reinforcement learning?",
                "choices": [
                    "As a classification problem",
                    "As a sequence modeling problem",
                    "As a graph search problem",
                    "As a generative adversarial problem",
                ],
                "correct_index": 1,
                "explanation": "Decision Transformer reframes RL as sequence modeling, training a causal Transformer on (return-to-go, state, action) sequences.",
            },
            {
                "question": "How do you control Decision Transformer's behavior at test time?",
                "choices": [
                    "By adjusting the temperature parameter",
                    "By conditioning on a desired return-to-go",
                    "By providing reward shaping signals",
                    "By selecting from a set of pre-trained skills",
                ],
                "correct_index": 1,
                "explanation": "At test time, you specify a desired return-to-go and the model generates actions to achieve that target return.",
            },
            {
                "question": "Why are Transformers well-suited for the RL credit assignment problem?",
                "choices": [
                    "They are faster to train than RNNs",
                    "Self-attention naturally handles long-horizon dependencies",
                    "They require less memory than other architectures",
                    "They can process multiple modalities simultaneously",
                ],
                "correct_index": 1,
                "explanation": "Transformers' self-attention mechanism can directly attend to relevant past events across long horizons, naturally handling credit assignment that TD methods struggle with.",
            },
        ],
    },
]


def main():
    client = get_admin_client()

    for paper in PAPERS:
        num = paper["number"]

        # Update summary
        client.table("papers").update({
            "summary": paper["summary"],
        }).eq("number", num).execute()
        print(f"  #{num}: Updated summary")

        # Insert questions
        for q in paper["questions"]:
            client.table("questions").insert({
                "paper_number": num,
                "question": q["question"],
                "choices": json.dumps(q["choices"]),
                "correct_index": q["correct_index"],
                "explanation": q["explanation"],
            }).execute()

        print(f"  #{num}: Inserted {len(paper['questions'])} questions")

    print(f"\nDone: {len(PAPERS)} papers updated with quality content")


if __name__ == "__main__":
    main()
