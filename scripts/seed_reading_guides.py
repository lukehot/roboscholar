#!/usr/bin/env python3
"""Add reading guides and key learnings for the 10 key papers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_admin_client

GUIDES = [
    {
        "number": 1,
        "summary": "RT-1 showed that a compact Transformer (35M params) trained on diverse real-world demonstrations (130k demos, 700+ tasks) can generalize to new instructions, objects, and environments. It established that data diversity — not model size — is the primary driver of robotic generalization.",
        "key_learnings": """- Data diversity > model scale for robotic generalization
- TokenLearner compresses visual tokens to make Transformer inference tractable at 3Hz
- FiLM conditioning is an effective way to fuse language with visual features
- Multi-task training on real robots requires careful data balancing across task difficulties
- Generalization to new objects and backgrounds emerges from data diversity, not explicit training""",
        "reading_guide": """Start with Section 1 (Introduction) for motivation, then jump to Section 3 (System Overview) for the architecture. Section 4 (Data) is surprisingly important — the data collection strategy is RT-1's real contribution. Skim Section 5 (Experiments) for the generalization results. Skip Section 2 (Related Work) unless you need references.""",
    },
    {
        "number": 2,
        "summary": "RT-2 created the VLA paradigm: take a vision-language model pre-trained on web data, represent robot actions as text tokens, and co-fine-tune on both web and robot data. This lets the model leverage internet-scale commonsense for robotic tasks, enabling emergent capabilities like semantic reasoning about novel concepts.",
        "key_learnings": """- Actions can be represented as text tokens, unifying language and robotic control in one model
- Co-training on web + robot data prevents catastrophic forgetting of semantic knowledge
- Emergent capabilities arise from composing the VLM's world knowledge with learned motor skills
- The VLA paradigm trades compute efficiency for semantic richness
- Scale matters: 55B parameters enable reasoning capabilities that smaller models lack""",
        "reading_guide": """Read Section 1 for the VLA concept, then Section 3 (Method) for how actions become tokens. Section 4.3 (Emergent Capabilities) is the most interesting part — this is what made the paper famous. Section 4.2 (Quantitative Results) shows where VLAs beat and lose to RT-1. Skip the supplementary unless you need implementation details.""",
    },
    {
        "number": 4,
        "summary": "OpenVLA democratized VLA research with a 7B open-source model that outperforms RT-2-X (55B) by 16.5%. Built on Llama 2 with DINOv2+SigLIP dual visual encoders, it showed that smart architecture choices (dual visual encoders, better data) can beat brute-force scaling. Supports LoRA fine-tuning with as few as 10 demos.",
        "key_learnings": """- Dual visual encoders (self-supervised + language-aligned) provide complementary features for manipulation
- A 7B model can outperform 55B through better architecture and data choices
- LoRA enables practical fine-tuning to new robots with minimal demonstrations
- Open-source weights and training recipes accelerate the entire field
- Training data quality and diversity matter more than raw quantity""",
        "reading_guide": """Section 3 (Architecture) is essential — understand the dual encoder design. Section 4 (Training) covers the Open X-Embodiment data strategy. Section 5.2 (Fine-tuning) is the most practically useful part if you want to deploy a VLA. The ablation studies (Section 5.3) reveal which components actually drive performance.""",
    },
    {
        "number": 9,
        "summary": "Diffusion Policy reframes robot action generation as conditional denoising diffusion over action sequences. This naturally handles multimodal distributions (multiple valid ways to do a task) and avoids the mode-averaging problem that plagues regression-based policies. Achieves 46.9% average improvement across 15 tasks.",
        "key_learnings": """- Diffusion solves the mode-averaging problem: when multiple valid actions exist, it samples cleanly from one mode instead of averaging
- Action chunking (predicting sequences) prevents jittery oscillations from single-step prediction
- The CNN variant is simpler and more sample-efficient; the Transformer variant is more flexible for multi-modal conditioning
- Receding-horizon control balances action consistency with reactivity
- The DDPM noise schedule and number of denoising steps are critical hyperparameters""",
        "reading_guide": """Section 3 (Method) is the core — understand DDPM action denoising and the two architecture variants. Section 3.3 (design decisions) is full of practical wisdom. Section 4 (Experiments) — focus on the ablations (Section 4.3) rather than raw numbers. The Appendix has implementation details critical for reproduction.""",
    },
    {
        "number": 31,
        "summary": "DreamerV3 is a world-model-based RL algorithm that works across 150+ diverse tasks with a single hyperparameter configuration. It learns a latent dynamics model (RSSM), then trains an actor-critic entirely from imagined trajectories. Key innovations (symlog predictions, free bits, percentile normalization) make it scale-invariant across domains. First to collect diamonds in Minecraft from scratch.",
        "key_learnings": """- World models enable sample-efficient learning by generating unlimited imagined experience
- Scale-invariant objectives (symlog, percentile normalization) are essential for cross-domain generality
- The RSSM compresses observations into compact latent states sufficient for decision-making
- Fixed hyperparameters across domains means the algorithm is truly general, not just well-tuned
- Model exploitation (policy finding 'cheats' in the world model) is managed through careful KL balancing""",
        "reading_guide": """Section 3 (Method) is dense but essential — focus on the three key techniques: symlog predictions (3.1), free bits (3.2), and return normalization (3.3). Section 4.1 (Minecraft) is the headline result. Section 4.3 (Ablations) shows each technique's contribution. Skip the domain-specific result tables unless you work in those areas.""",
    },
    {
        "number": 44,
        "summary": "The Open X-Embodiment project assembled the largest open robot learning dataset — data from 22 robot types across 21 institutions covering 527 skills. Training a single RT-X model on this data shows positive transfer: it outperforms policies trained on any single robot's data alone, establishing cross-embodiment transfer as a viable scaling strategy.",
        "key_learnings": """- Cross-embodiment transfer works: a shared model outperforms single-robot specialists on average
- Data diversity across embodiments provides a form of regularization and generalization
- Data standardization (formats, action spaces, annotations) is a major practical bottleneck
- Negative transfer is possible for specific robots even when average performance improves
- The dataset itself (not just the model) is the lasting contribution""",
        "reading_guide": """Section 3 (Dataset) is the most important — understand the data diversity and standardization challenges. Section 4 (Models) describes RT-1-X and RT-2-X training. Section 5 (Experiments) — focus on Figure 4 (per-robot transfer results) which shows both wins and losses. The appendix dataset tables are a valuable reference for anyone collecting robot data.""",
    },
    {
        "number": 66,
        "summary": "SayCan bridges LLMs and robots by combining semantic scoring (LLM: 'what's useful?') with affordance scoring (value function: 'what's feasible?'). The multiplication p(skill|instruction) * p(success|skill, state) grounds abstract language plans in physical reality, enabling long-horizon tasks like 'I spilled my drink, can you help?'",
        "key_learnings": """- LLMs have broad knowledge but zero physical grounding — affordance functions provide that grounding
- Multiplicative combination of semantic and physical scores is simple but effective
- The approach is bounded by the pre-trained skill library — it can't synthesize new motor primitives
- Scalability of skill scoring becomes a bottleneck with large skill vocabularies
- This modular approach (LLM planner + skill executor) is still dominant in deployed systems""",
        "reading_guide": """Section 1 gives excellent motivation for the grounding problem. Section 3 (Method) — the factored scoring formula is the key idea (one page). Section 4 (Experiments) — the qualitative examples (Table 1) are more informative than the metrics. Section 5 (Discussion) covers limitations honestly. Skip Section 2 unless you need the LLM/robotics context.""",
    },
    {
        "number": 5,
        "summary": "Pi-Zero from Physical Intelligence combines a pre-trained VLM with flow matching to generate continuous robot actions. Unlike token-based VLAs, flow matching produces smooth action trajectories suited for dexterous manipulation. Pre-trained on diverse cross-embodiment data and post-trained on specific tasks, it solves complex tasks (laundry folding, box assembly) that prior generalist models couldn't.",
        "key_learnings": """- Flow matching generates continuous actions more efficiently than DDPM diffusion (fewer steps needed)
- The pre-train/post-train recipe from LLMs transfers directly to robot policies
- VLM pre-training provides a strong initialization even when the output modality (actions) differs from language
- Complex dexterous tasks (bimanual, contact-rich) are the true test of action representations
- Cross-embodiment pre-training provides a general foundation that post-training specializes""",
        "reading_guide": """Section 3 (Architecture) is critical — understand how flow matching integrates with the VLM backbone. Section 4 (Training Recipe) explains the pre-train/post-train strategy. Section 5 (Experiments) — focus on the dexterous tasks (5.2) which differentiate pi0 from prior work. The comparison with Diffusion Policy (5.3) clarifies the flow matching advantage.""",
    },
    {
        "number": 21,
        "summary": "Gemini Robotics adapts Google's frontier Gemini 2.0 model for physical-world control, with two variants: a VLA for direct robot control and an embodied reasoning module for spatial/temporal understanding. Represents the bet that the richest possible foundation model — with world knowledge from trillions of tokens — provides the best starting point for robotics.",
        "key_learnings": """- Frontier foundation models provide an enormously rich prior for robotic reasoning
- Separating embodied reasoning from action generation allows each to leverage different data sources
- Responsible AI development is a real constraint in physical-world deployment
- Reproducibility challenges: results depend on proprietary models and infrastructure
- The approach validates the 'adapt the best general model' strategy over 'build robotics-specific from scratch'""",
        "reading_guide": """Read Section 2 (Approach) for the model family overview. Section 3 (Gemini Robotics-ER) if you care about spatial reasoning capabilities. Section 4 (Gemini Robotics VLA) for the action generation approach. Section 6 (Responsible Development) is unusually substantive for a robotics paper. Skim the benchmark results — the qualitative capabilities are more informative.""",
    },
    {
        "number": 74,
        "summary": "Decision Transformer reframes RL as sequence modeling: a causal Transformer is trained on (return-to-go, state, action) sequences, and at test time you specify a desired return to elicit corresponding behavior quality. Matches specialized offline RL algorithms (CQL, IQL) with a much simpler approach — no value functions, no Bellman updates, just supervised sequence prediction.",
        "key_learnings": """- RL can be reduced to supervised sequence prediction with return conditioning
- Self-attention handles long-horizon credit assignment that TD-methods struggle with
- The model cannot exceed the best trajectory in the data (no trajectory stitching)
- Return conditioning provides an intuitive way to control behavior quality at test time
- The simplicity of the approach (just sequence modeling!) is its biggest strength and weakness""",
        "reading_guide": """Section 3 (Method) is short and elegant — read it carefully. Section 4 (Evaluations) — Atari results (4.1) show where DT shines (sparse rewards, long horizons). Gym results (4.2) show where TD-methods still win (stitching). Section 5 (Discussion) on stitching limitations is the most important part for understanding when to use DT vs. traditional offline RL.""",
    },
]


def main():
    client = get_admin_client()

    for g in GUIDES:
        client.table("papers").update({
            "summary": g["summary"],
            "key_learnings": g["key_learnings"],
            "reading_guide": g["reading_guide"],
        }).eq("number", g["number"]).execute()
        print(f"  #{g['number']}: Updated summary, key learnings, and reading guide")

    print(f"\nDone: {len(GUIDES)} papers updated")


if __name__ == "__main__":
    main()
