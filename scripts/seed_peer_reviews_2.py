#!/usr/bin/env python3
"""Seed peer reviews for papers 29-56."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_admin_client

REVIEWS = {
    29: """STRENGTHS:
PerAct introduced a compelling approach to multi-task manipulation by lifting the problem into a discretized 3D voxel space, allowing the policy to reason about spatial relationships directly rather than through 2D image projections. The use of Perceiver Transformer over voxelized point clouds was elegant — it sidesteps the quadratic cost of full 3D attention while retaining global context. The demonstration that a single model can handle 18 RLBench tasks with language conditioning was a strong result at the time, and the 3D inductive bias clearly helped with tasks requiring precise spatial reasoning (e.g., stacking, insertion).

LIMITATIONS:
The voxel discretization (100x100x100 at 1cm resolution) fundamentally limits workspace size and precision — you cannot simultaneously cover a large area and maintain fine-grained accuracy. The method requires calibrated depth cameras and known camera extrinsics, which is a strong assumption that limits real-world deployment. Inference speed is slow due to the dense 3D representation. Perhaps most critically, PerAct predicts a single next-best pose (position + rotation) rather than a continuous trajectory, making it unsuitable for tasks requiring smooth, dynamic motions like wiping or pouring.

COMMUNITY RECEPTION:
Well-received as a principled argument for 3D representations in manipulation. The robotics community appreciated the clean experimental setup and the direct comparison showing 3D methods outperforming 2D image-based policies on spatial tasks. However, the sim-only evaluation on RLBench was noted as a limitation. GNFactor, Act3D, and RVT built directly on PerAct's insights while addressing its resolution and speed limitations. The broader trend moved toward methods that incorporate 3D understanding without committing to explicit voxelization.

OPEN QUESTIONS:
Can 3D voxel-based methods scale to mobile manipulation where the workspace is room-sized? Is explicit 3D discretization necessary, or can implicit 3D representations (neural fields, 3D-aware features) achieve the same spatial reasoning benefits without the resolution trade-off? How should 3D representations be combined with language models that operate in token space?""",

    30: """STRENGTHS:
GNFactor made a compelling case for combining generalizable neural feature fields with robotic manipulation policies. By distilling vision-language features (from CLIP/LSeg) into a 3D neural field, the method enables the policy to reason about semantics in 3D space rather than in projected 2D views. The multi-task performance on both simulated and real robot tasks was solid, and the approach demonstrated meaningful generalization to novel object configurations. The integration of language-grounded features directly into the 3D representation was a genuine architectural contribution.

LIMITATIONS:
Neural field reconstruction is computationally expensive — requiring multiple camera views and non-trivial optimization per scene, making real-time deployment challenging. The method assumes access to multiple calibrated RGB-D cameras, which is a significant hardware requirement. The real-world experiments were limited in scope (a few tabletop tasks) and did not stress-test generalization to truly novel objects or cluttered environments. The reliance on pre-trained vision-language features means the quality ceiling is set by those foundation models.

COMMUNITY RECEPTION:
Recognized as a creative fusion of the neural radiance field boom with robotic manipulation. The paper appeared at a time when the community was actively exploring how NeRF-style representations could benefit robotics. Researchers appreciated the 3D semantic reasoning capability but questioned the practical overhead of neural field reconstruction for real-time control. Subsequent work explored more efficient ways to get 3D-aware features (e.g., using feed-forward models rather than per-scene optimization).

OPEN QUESTIONS:
Can neural feature field reconstruction be made fast enough for closed-loop control at reasonable frequencies? Is per-scene optimization fundamentally at odds with real-time robotic control, or can amortized inference close the gap? How do these 3D feature representations compare to simply using multi-view 2D features as in RVT or multi-camera setups?""",

    31: """STRENGTHS:
Dreamer V3 is a landmark in model-based reinforcement learning. By introducing symlog predictions, a fixed hyperparameter set that works across domains, and a robust KL-balancing scheme for the world model's latent dynamics, Hafner et al. achieved something no prior MBRL method had: a single algorithm with a single set of hyperparameters that masters Atari, continuous control, DMLab, Minecraft, and Crafter. The demonstration of diamond mining in Minecraft from scratch — a long-horizon, sparse-reward task requiring complex sequential reasoning — was a breakthrough result. The engineering discipline of zero domain-specific tuning is arguably the paper's most important contribution.

LIMITATIONS:
The world model's accuracy degrades significantly with complex contact dynamics — precisely the regime that matters most for manipulation and locomotion in the real world. The latent dynamics model learns a compressed representation that smooths over discontinuous contact events (collisions, grasping, breaking). This means Dreamer V3 excels in domains where dynamics are relatively smooth or where visual prediction suffices (Atari, navigation) but struggles with the hard physics of dexterous manipulation. The imagination-based training also accumulates compounding errors over long rollouts, and the method requires substantial compute for the world model training.

COMMUNITY RECEPTION:
Extremely well-received. The paper was widely seen as the strongest evidence yet that MBRL can be a general-purpose algorithm. The Minecraft result captured broad attention. However, the robotics community noted the gap between simulated benchmarks and real-world contact-rich tasks. DayDreamer (paper #32) was the direct attempt to bridge this gap, and TD-MPC2 (#33) offered an alternative that avoids explicit image reconstruction. Dreamer V3 became the standard baseline for MBRL comparisons.

OPEN QUESTIONS:
Can world models learn accurate contact dynamics, or is this a fundamental limitation of learning dynamics in latent space? How should world models handle the transition from smooth dynamics to discontinuous contact events? Is the symlog trick sufficient for the reward scales encountered in real-world robotics, or do real-world rewards require different normalization?""",

    32: """STRENGTHS:
DayDreamer was the first convincing demonstration that Dreamer-style world models can learn directly on physical robots with only a few hours of real-world interaction. The results on a quadruped (locomotion), a robotic arm (manipulation), and a ground vehicle were impressive for their sample efficiency — the quadruped learned to walk in about one hour of real experience. The paper validated the core promise of MBRL: by learning a model of the world, the agent can perform most of its learning "in imagination," drastically reducing the need for costly real-world trials.

LIMITATIONS:
The tasks demonstrated are relatively simple and short-horizon — walking, basic pick-and-place, and driving. The world model's imagination quality was not evaluated rigorously against ground truth dynamics; it is unclear how much the model captures versus how much the RL optimization is robust to model inaccuracies. The reset problem (needing human intervention between episodes) was glossed over. Furthermore, the method requires careful engineering of the observation space and action space for each robot platform, undermining the "general purpose" narrative somewhat.

COMMUNITY RECEPTION:
The robotics community was enthusiastic but cautious. DayDreamer showed real-world MBRL is feasible, which was a meaningful milestone, but the simple task suite left open whether the approach would scale to contact-rich manipulation, multi-step assembly, or tasks requiring fine motor control. The paper motivated further work on real-world world models (TD-MPC2, RoboDreamer) and raised the profile of model-based methods in a field dominated by imitation learning.

OPEN QUESTIONS:
Can DayDreamer-style approaches scale to dexterous manipulation with complex contact? How do we handle the sim-to-real gap in world model accuracy when the real world has unmodeled physics? Can the world model be pre-trained on simulation data and then fine-tuned on real data, combining the best of both paradigms?""",

    33: """STRENGTHS:
TD-MPC2 demonstrated that temporal difference learning combined with model-predictive control can scale to a single generalist agent handling 80+ continuous control tasks across multiple domains. The key insight — using a task-conditioned latent dynamics model with learned value functions for planning — elegantly avoids the need for explicit pixel reconstruction that burdens Dreamer-style methods. The scaling experiments showing consistent improvement from 1M to 317M parameters were convincing, and the single-hyperparameter-set result (like Dreamer V3) was impressive. The planning at test time allows the agent to adapt behavior without retraining.

LIMITATIONS:
The reliance on proprioceptive state (not pixels) for most benchmarks limits direct comparison with vision-based methods and real-world applicability where state estimation is itself a challenge. The model-predictive control loop at inference time adds latency and computational cost — each action requires running multiple planning iterations through the learned model. The method has not been convincingly demonstrated on real robots with contact-rich tasks. The learned latent space is opaque, making it difficult to diagnose failures or understand what the world model has captured.

COMMUNITY RECEPTION:
Highly regarded in the MBRL community as a strong alternative to the Dreamer line. Researchers appreciated the clean scaling results and the principled combination of learning and planning. Some noted that the comparison with Dreamer V3 was not entirely apples-to-apples due to different observation spaces. The method influenced subsequent work on scalable world models for control, and the codebase became widely used for benchmarking.

OPEN QUESTIONS:
Can TD-MPC2's approach scale to vision-based inputs while maintaining its efficiency advantages over Dreamer? How does the planning horizon interact with the world model's accuracy — is there a principled way to truncate planning when the model becomes unreliable? Can this framework be extended to multi-agent or multi-robot settings?""",

    34: """STRENGTHS:
UniSim proposed an ambitious vision: a single generative model that can simulate diverse real-world interactions — both visual outcomes of actions and the dynamics of scenes. By conditioning a video generation model on actions and language, UniSim showed how generative models could serve as universal simulators for training robot policies, planning, and data augmentation. The breadth of demonstrations (navigation, manipulation, human-scene interaction) was impressive, and the paper articulated a compelling long-term vision for replacing hand-crafted simulators with learned ones.

LIMITATIONS:
The gap between visually plausible simulation and physically accurate simulation is enormous and was not adequately addressed. UniSim generates videos that look reasonable but may violate physics — objects can interpenetrate, masses and forces are not conserved, and contact dynamics are approximate at best. Policies trained in UniSim's imagination may learn visually-correlated shortcuts rather than physically-grounded behaviors. The evaluation was largely qualitative, with limited quantitative assessment of simulation fidelity or downstream policy quality. Computational cost of running a large generative model as a simulator is prohibitive for the millions of rollouts RL typically requires.

COMMUNITY RECEPTION:
The community recognized UniSim as a thought-provoking vision paper. It energized the growing intersection of generative models and robotics. However, many researchers were skeptical about the physical fidelity question — a simulator that produces visually plausible but physically incorrect dynamics could be worse than useless for policy learning. The paper sparked productive debate about what "simulation" means and whether learned simulators need to capture physics or just statistical regularities of the world.

OPEN QUESTIONS:
How do we evaluate whether a learned simulator is physically accurate enough for policy training? Can we combine the visual realism of generative models with the physical correctness of physics engines? What is the minimum level of physical fidelity required for policies trained in imagination to transfer to reality?""",

    35: """STRENGTHS:
Genie 2 from DeepMind extended the original Genie framework to generate interactive, explorable 3D environments from a single image prompt. The ability to create consistent, navigable 3D worlds that respond to agent actions — without any explicit 3D modeling or game engine — represents a genuine technical achievement. The environments exhibit remarkable visual coherence with consistent geometry and object permanence across many steps of interaction. The potential applications for training embodied agents in diverse procedurally-generated worlds are significant.

LIMITATIONS:
Physical accuracy is fundamentally limited by the generative model's training distribution. Generated environments look plausible but do not obey consistent physics — gravity, collision, and material properties are approximate and can be inconsistent across a long interaction. The computational cost of generating environments in real-time (or near-real-time) is enormous. The evaluation was primarily qualitative, making it difficult to assess how useful these environments are for actual agent training. Limited public technical detail makes it hard for the community to build on the work.

COMMUNITY RECEPTION:
Generated significant excitement as a demonstration of what generative models can do for world simulation. The demo videos were widely shared and discussed. However, the research community noted the lack of rigorous evaluation, the closed nature of the work, and the fundamental question of whether visually coherent but physically approximate environments are useful for training robots that must operate in the real, physical world. Some viewed it as more relevant to gaming and virtual environments than robotics.

OPEN QUESTIONS:
Can generated 3D environments be grounded in physics well enough to train manipulation or locomotion policies? How does agent performance in generated environments transfer to real or conventionally-simulated environments? Is there a path to making generation efficient enough for large-scale RL training?""",

    36: """STRENGTHS:
DIAMOND (Diffusion for World Modeling) made a clean contribution by showing that diffusion models can serve as effective world models for reinforcement learning. By using a diffusion model to predict next observations conditioned on actions, DIAMOND achieved strong performance on Atari benchmarks, demonstrating that the high-fidelity generation capabilities of diffusion models translate into better world model quality. The visual quality of imagined rollouts was notably superior to prior autoregressive or VAE-based world models, and this visual fidelity correlated with improved policy performance.

LIMITATIONS:
Diffusion model inference requires multiple denoising steps per prediction, making imagination-based training significantly slower than single-pass models like Dreamer's RSSM. This computational cost scales poorly with the number of imagination rollouts needed for policy optimization. The evaluation was limited to Atari — a domain with relatively simple, deterministic dynamics — and it remains unclear how well diffusion world models handle the stochasticity and complexity of real-world dynamics. The method does not address how to plan with a diffusion world model efficiently.

COMMUNITY RECEPTION:
Well-received as a solid technical contribution connecting the diffusion model literature with MBRL. Researchers appreciated the clear experimental setup and the insight that world model fidelity directly impacts policy quality. However, the Atari-only evaluation and the speed concerns limited enthusiasm for immediate practical adoption. The paper contributed to the broader trend of exploring generative models as world models, alongside UniSim, Genie, and Cosmos.

OPEN QUESTIONS:
Can the inference cost of diffusion world models be amortized or reduced (via distillation, consistency models, etc.) to make large-scale imagination-based training feasible? Do diffusion world models maintain their fidelity advantage in continuous control and real-world visual domains? How should planning algorithms be adapted to work with the iterative generation process of diffusion models?""",

    37: """STRENGTHS:
NVIDIA's Cosmos represents one of the most ambitious efforts to build "world foundation models" — large-scale video generation models specifically designed for physical AI applications. The emphasis on curating a massive, physics-relevant training dataset and designing tokenizers optimized for physical scene understanding is commendable. The explicit goal of creating pre-trained world models that can be fine-tuned for downstream robotics, autonomous driving, and simulation tasks addresses a real gap in the field. The scale of compute and data invested signals serious long-term commitment.

LIMITATIONS:
The gap between generating visually plausible video and modeling physically accurate dynamics remains the central unresolved issue. Cosmos can produce impressive-looking videos of physical scenes, but there is limited evidence that the internal representations capture the causal physical structure needed for planning and control. The benchmark evaluations focus on video quality metrics (FVD, FID) rather than downstream task performance, which is the metric that actually matters for robotics. The closed-weight approach (at least initially) limits community adoption and scientific scrutiny.

COMMUNITY RECEPTION:
The scale and ambition of Cosmos attracted attention, and NVIDIA's positioning of world foundation models as infrastructure for physical AI resonated with the community's direction. However, researchers remained skeptical about whether scaling video generation is the right path to world models — the concern being that these models learn correlations in pixel space rather than causal physical structure. The open-sourcing of model weights and tokenizers was welcomed when it occurred, enabling independent evaluation.

OPEN QUESTIONS:
Do video generation models implicitly learn physics, or do they learn visual shortcuts that approximate physics? How should world foundation models be evaluated for robotics utility — what benchmarks actually predict downstream policy quality? Can Cosmos-style models be fine-tuned efficiently for specific robot embodiments and tasks, or does the generality come at the cost of task-specific performance?""",

    38: """STRENGTHS:
Cosmos Policy builds on the Cosmos world foundation model by demonstrating how a pre-trained video generation model can be adapted for robot policy learning. The approach of using the world model's learned representations as a perceptual backbone for policy networks is principled — if the world model has learned useful physical scene understanding, those representations should transfer to control. The integration with NVIDIA's broader Cosmos ecosystem provides a coherent pipeline from pre-training to deployment.

LIMITATIONS:
The evaluation is limited and does not convincingly demonstrate that the Cosmos-derived representations are superior to standard pre-trained visual features (CLIP, DINOv2, etc.) for policy learning. The computational overhead of running a large world foundation model as a feature extractor is significant and may not justify the marginal gains. The tight coupling to NVIDIA's ecosystem raises concerns about reproducibility and generalizability. The real-world experiments are narrow in scope.

COMMUNITY RECEPTION:
Viewed as a natural extension of the Cosmos line, but the community wanted stronger evidence that world model pre-training provides a meaningful advantage over simpler visual pre-training for policy learning. The comparison baselines were seen as insufficient — a rigorous comparison against state-of-the-art VLAs using standard visual encoders would be more convincing. The work contributes to the important question of how to leverage world models for control but does not yet provide a definitive answer.

OPEN QUESTIONS:
Does pre-training a world model provide better representations for policy learning than visual-only pre-training (e.g., masked autoencoding, contrastive learning)? What is the optimal way to extract and use representations from a world model for downstream control — latent features, predicted futures, or both? How much world model pre-training data is needed to see benefits for specific robot tasks?""",

    39: """STRENGTHS:
RoboDreamer addressed a key limitation of world models for robotics: the inability to compose novel behaviors from learned dynamics primitives. By introducing compositional world models that can combine learned sub-skills in imagination, RoboDreamer enabled robots to "dream up" solutions to tasks that require sequencing multiple manipulation skills — even tasks not seen during training. The language-conditioned composition mechanism was a creative way to leverage LLM planning with learned dynamics.

LIMITATIONS:
The compositional assumption — that complex tasks can be decomposed into clean, sequential sub-skills — is strong and often violated in real manipulation. Many tasks require simultaneous coordination (e.g., holding an object steady while screwing) rather than sequential composition. The world model's accuracy in composed sequences degrades due to compounding errors, especially at transition points between sub-skills. The evaluation was primarily in simulation with relatively structured task decompositions.

COMMUNITY RECEPTION:
Appreciated as a creative contribution at the intersection of world models, language-guided planning, and compositional reasoning. The community recognized the importance of the compositional generalization problem but noted that the approach works best for tasks that naturally decompose into sequential stages. The reliance on language-specified decomposition was seen as both a strength (interpretable) and a limitation (requires accurate task decomposition).

OPEN QUESTIONS:
How can world models handle tasks that require simultaneous rather than sequential composition of skills? Can the compositional framework be extended to handle failure recovery and replanning when a sub-skill fails? What happens when the language decomposition does not match the actual physical decomposition of the task?""",

    40: """STRENGTHS:
IRASim focused on a specific and important problem: generating videos that accurately depict robot-object interactions, including the physical consequences of robot actions. By explicitly modeling the interaction between the robot and objects in the scene, IRASim produces more physically plausible predictions than generic video generation models. The approach of conditioning video generation on robot actions and modeling the resulting object state changes is well-motivated for policy learning and planning.

LIMITATIONS:
The quality of interaction modeling is limited by the training data — the model can only predict interactions similar to those seen during training. Truly novel interactions (new objects, new manipulation strategies) may produce implausible results. The evaluation focused on visual quality of generated interactions rather than whether the predictions are accurate enough for downstream policy improvement. The method requires paired robot-action and video data, which is expensive to collect at scale.

COMMUNITY RECEPTION:
Recognized as a useful contribution to the growing field of action-conditioned video generation for robotics. The explicit focus on interaction quality (rather than just visual quality) was appreciated. However, the practical impact was seen as incremental — the method improves prediction quality for seen-distribution interactions but does not solve the fundamental problem of generalizing to novel physical interactions.

OPEN QUESTIONS:
Can interaction-aware video generation models generalize to truly novel object categories and manipulation strategies? How should the quality of interaction predictions be evaluated for policy learning — what error metrics predict downstream task success? Can interaction-aware generation be combined with physics priors to improve out-of-distribution prediction?""",

    41: """STRENGTHS:
PhysDreamer tackled the important challenge of learning physically grounded dynamics models for object manipulation. By incorporating physics-based priors (material properties, deformation models) into a generative framework, PhysDreamer aimed to produce imagination rollouts that respect physical laws rather than merely looking visually plausible. The ability to predict how objects deform and respond to applied forces is critical for manipulation tasks involving soft objects, articulated mechanisms, and granular materials.

LIMITATIONS:
The physics modeling is approximate and limited to specific material types and interaction modes. Complex phenomena like fracture, fluid dynamics, and multi-body contact are not captured. The computational cost of combining physics simulation with neural generation is high. The method was demonstrated on a narrow set of scenarios, and it is unclear how the approach generalizes across the diversity of materials and interactions encountered in real-world manipulation. The reliance on known material properties limits applicability to novel objects.

COMMUNITY RECEPTION:
The community appreciated the physics-aware direction and recognized it as a step toward more grounded world models. However, many noted that the gap between the simplified physics in PhysDreamer and the full complexity of real-world dynamics remains large. The work was seen as complementary to data-driven approaches — ideally, learned world models would discover physical regularities from data rather than requiring explicit physics priors.

OPEN QUESTIONS:
Is it better to inject physics priors into generative models or to let them discover physical regularities from data? Can physics-aware generation scale to the full diversity of materials and interactions in household environments? How should we evaluate whether a physics-grounded model is accurate enough for policy training?""",

    42: """STRENGTHS:
Genesis provides a differentiable physics engine implemented as a generative framework, allowing gradient-based optimization through physics simulation. This is a significant engineering contribution — differentiable simulation has long been desired for robot learning but existing physics engines (MuJoCo, PyBullet, Isaac Gym) offer limited or no differentiability. The ability to compute gradients through contact dynamics, material deformation, and multi-body interactions enables new approaches to trajectory optimization, system identification, and policy learning. The unified treatment of rigid bodies, soft bodies, and fluids in a single differentiable framework is technically ambitious.

LIMITATIONS:
Differentiable physics engines suffer from well-known issues: gradients through contact dynamics are often noisy, discontinuous, or uninformative due to the discrete nature of contact events. The sim-to-real gap may be larger than for non-differentiable simulators that have been carefully tuned over decades. The computational cost of backpropagating through long simulation horizons is substantial. The accuracy of the physics (especially for soft bodies and fluids) relative to established simulators requires more extensive validation.

COMMUNITY RECEPTION:
Welcomed as an ambitious and useful open-source tool for the robot learning community. Researchers appreciated the unified differentiable framework and the potential for gradient-based policy optimization. However, the community remains cautious about the practical utility of differentiable simulation — the theoretical advantages of gradients through physics are clear, but empirical results have been mixed, with RL-based approaches often matching or exceeding gradient-based methods in practice. Genesis is viewed as promising infrastructure that needs more downstream validation.

OPEN QUESTIONS:
Can gradients through contact dynamics be made reliable enough for practical policy optimization, or will the discontinuities always limit their utility? How does Genesis's sim-to-real transfer compare to established, non-differentiable simulators with carefully tuned parameters? Can differentiable simulation enable fundamentally new capabilities (e.g., online system identification) that RL-based approaches cannot match?""",

    43: """STRENGTHS:
VideoVLA proposed an intriguing paradigm: using video generation models as generalizable robot manipulation policies. Instead of the standard VLA approach (vision-language model predicting actions), VideoVLA generates future video conditioned on the task instruction and then extracts actions from the generated video. This is compelling because video generation models, pre-trained on vast internet data, capture rich knowledge about how objects move and interact. The method demonstrated that this approach can generalize to novel objects and scenes not present in the robot training data.

LIMITATIONS:
The action extraction pipeline — going from generated video to executable robot actions — is fragile and introduces significant latency. Generated videos may be visually plausible but kinematically infeasible for the robot, leading to extraction failures. The computational cost of running a video generation model in the control loop makes real-time operation challenging. The method's performance is upper-bounded by the video generator's physical accuracy, which, as discussed in the world model literature, remains limited for contact-rich interactions.

COMMUNITY RECEPTION:
The idea generated significant interest as a creative inversion of the world-model-for-control paradigm. Researchers appreciated the conceptual elegance of using internet-scale video generation as an implicit world model. However, the practical challenges — latency, action extraction reliability, physical accuracy — tempered enthusiasm. The work is seen as an interesting research direction rather than a practical alternative to direct action prediction VLAs.

OPEN QUESTIONS:
Can action extraction from generated video be made robust and real-time? Is the video generation pathway fundamentally limited by the gap between visual plausibility and physical accuracy? Could a hybrid approach — using video generation for high-level planning and direct action prediction for low-level control — combine the best of both worlds?""",

    44: """STRENGTHS:
The Open X-Embodiment (OXE) dataset is one of the most significant community contributions in robot learning. By aggregating manipulation data from 22 robot embodiments across 21 institutions, the dataset (with over 1 million episodes) enabled the first large-scale study of cross-embodiment transfer. The RT-X experiments demonstrated that training on diverse cross-embodiment data improves performance compared to training on any single dataset, providing empirical evidence for the "data diversity" hypothesis. The collaborative, open-science approach set an important precedent for the field.

LIMITATIONS:
Data quality varies enormously across contributing labs — different collection protocols, skill levels of human operators, camera configurations, and task complexities create significant distributional inconsistencies. The dataset is heavily biased toward tabletop manipulation with parallel-jaw grippers; mobile manipulation, dexterous hand manipulation, and locomotion are underrepresented. The action spaces across embodiments are not standardized, requiring heuristic normalization that may lose important information. Quality control and curation at this scale is extremely challenging, and some contributing datasets contain significant noise or failure trajectories labeled as successes.

COMMUNITY RECEPTION:
Widely celebrated as a milestone for open-source robot learning. The community embraced the dataset as essential infrastructure, and it became the standard training set for new VLA models (OpenVLA, Octo, CrossFormer). However, researchers who worked closely with the data identified significant quality issues and biases. The tabletop bias in particular has been criticized — models trained on OXE may appear general but are actually specialized for a narrow manipulation regime. Despite these issues, OXE's impact on the field has been substantial and clearly positive.

OPEN QUESTIONS:
How should cross-embodiment datasets be curated to ensure quality without excluding valuable but noisy data? Can the tabletop manipulation bias be corrected by targeted data collection, or does it require fundamentally different data collection strategies? What is the right abstraction for cross-embodiment action representation — should actions be standardized, or should models learn to translate between action spaces?""",

    45: """STRENGTHS:
HPT (Heterogeneous Pre-trained Transformers) introduced a principled architecture for pre-training on data from diverse robot embodiments with different observation and action spaces. The key innovation — using modality-specific tokenizers (stem modules) that project heterogeneous inputs into a shared token space processed by a single Transformer trunk — is a clean engineering solution to the cross-embodiment problem. The method demonstrated that scaling pre-training across heterogeneous datasets improves transfer to new embodiments and tasks, validating the cross-embodiment pre-training hypothesis with a practical architecture.

LIMITATIONS:
The stem module design requires hand-crafting tokenizers for each new observation and action modality, which limits the "universality" of the approach. The shared trunk may learn lowest-common-denominator representations that fail to capture modality-specific nuances. The evaluation focused primarily on simulated benchmarks and limited real-world experiments, leaving open questions about real-world transfer. The method does not address how to handle the quality disparity in heterogeneous datasets — noisy data from one embodiment may degrade performance on others.

COMMUNITY RECEPTION:
HPT was well-received as a practical architectural contribution. The robotics community appreciated the clean separation between modality-specific tokenization and shared representation learning. It was seen as complementary to OXE (data) and CrossFormer (architecture), contributing to the growing ecosystem of cross-embodiment tools. Some researchers questioned whether the architecture was complex enough to capture the deep differences between embodiments (e.g., the dynamics of a quadruped vs. a manipulator arm).

OPEN QUESTIONS:
Can stem modules be learned automatically rather than designed manually for each embodiment? Does the shared Transformer trunk actually learn embodiment-agnostic features, or does it implicitly partition its capacity by embodiment? How should heterogeneous data be weighted during pre-training to prevent dominant embodiments from drowning out rare ones?""",

    46: """STRENGTHS:
CrossFormer directly tackled the cross-embodiment transfer problem with an architecture designed to handle varying observation and action dimensions across robots. The use of attention mechanisms that can dynamically attend to variable-length observation and action tokens is more principled than fixed-size projection approaches. The paper demonstrated meaningful positive transfer between substantially different robot morphologies, showing that shared manipulation knowledge exists across embodiments. The evaluation across multiple OXE embodiments was thorough.

LIMITATIONS:
The "cross-embodiment transfer" demonstrated is primarily between manipulator arms with similar kinematic structures — transfer to fundamentally different embodiments (legged robots, flying robots, humanoids) was not demonstrated and may require deeper architectural innovations. The computational cost of the flexible attention mechanism scales poorly with the number of observation and action dimensions. The method assumes that shared manipulation primitives exist across embodiments, which may not hold for highly dissimilar robot morphologies. Fine-tuning on target embodiment data is still necessary, making the "transfer" more like "better initialization" than zero-shot generalization.

COMMUNITY RECEPTION:
Recognized as a solid contribution to the cross-embodiment learning literature. The community appreciated the principled architectural approach and the thorough evaluation on OXE data. CrossFormer became one of the standard baselines for cross-embodiment experiments. However, some researchers noted that the improvements over simply training a larger model on pooled data were modest, raising questions about whether specialized architectures are necessary or whether scale alone can solve cross-embodiment transfer.

OPEN QUESTIONS:
Is cross-embodiment transfer between fundamentally different robot morphologies possible with current approaches, or does it require new inductive biases? What is the "common language" of manipulation that transfers across embodiments — is it task-level semantics, motion primitives, or something else? At what point does embodiment-specific fine-tuning dominate the pre-trained representation, making the transfer benefit negligible?""",

    47: """STRENGTHS:
NVIDIA's GR00T N1 represents a well-resourced industry effort to build a generalist humanoid robot foundation model. The integration of vision, language, and proprioceptive inputs into a unified architecture for humanoid control is technically challenging, and GR00T N1 demonstrated basic humanoid manipulation and locomotion behaviors. The backing of NVIDIA's compute infrastructure and the integration with their simulation ecosystem (Isaac, Cosmos) provides a viable path from sim-to-real transfer for humanoid robots.

LIMITATIONS:
The public evaluation data is severely limited — most demonstrations involve simple tasks (grasping objects, basic walking) that do not stress-test the generalist claims. Independent reproduction and evaluation has been impossible due to the closed nature of the model and the requirement for specific NVIDIA humanoid hardware. The tasks shown are substantially simpler than what specialized controllers achieve on humanoid platforms. The gap between the impressive marketing narrative and the available evidence is a concern. Without rigorous, independent benchmarking on diverse tasks, the model's actual capabilities remain unclear.

COMMUNITY RECEPTION:
The announcement generated significant media attention, but the research community's response was more measured. Researchers noted the limited evaluation, the lack of published ablations or detailed technical methodology, and the absence of open weights or reproducible benchmarks. The work was viewed as an engineering achievement and a statement of intent rather than a scientific contribution. Comparisons with open efforts (HPT, CrossFormer, Octo) highlighted the tension between industry scale and academic rigor/openness.

OPEN QUESTIONS:
What is GR00T N1's actual performance on standardized manipulation and locomotion benchmarks? Can the model handle dynamic, reactive tasks (catching, balancing recovery, tool use) or is it limited to quasi-static manipulation? How does the performance compare quantitatively with specialized controllers and with other generalist models when evaluated on the same tasks?""",

    48: """STRENGTHS:
Helix addresses the relatively underexplored problem of heterogeneous multi-robot learning — training policies that enable different types of robots (manipulators, mobile bases, drones) to coordinate on shared tasks. This is a practically important problem for real-world deployment where fleets consist of diverse robot types. The approach of learning shared task representations that can be decoded into embodiment-specific actions is architecturally sound. Demonstrating multi-robot coordination with heterogeneous morphologies goes beyond the standard single-robot paradigm that dominates the field.

LIMITATIONS:
The coordination scenarios demonstrated are relatively simple — typically two robots performing complementary actions in a shared workspace. Scaling to many heterogeneous robots with complex interdependencies remains an open challenge. The communication protocol between robots is learned end-to-end, making it opaque and difficult to debug or guarantee safety properties. The sim-to-real transfer of coordinated multi-robot policies is significantly harder than single-robot transfer due to compounding uncertainties.

COMMUNITY RECEPTION:
Appreciated as an interesting step beyond single-robot policy learning. The multi-robot research community welcomed the heterogeneous approach, as most prior multi-robot learning work assumes homogeneous agents. However, the limited scale and simplicity of the coordination tasks kept expectations tempered. The work highlighted an important gap in the field — most foundation model efforts focus on single robots, but real-world deployment increasingly requires multi-robot coordination.

OPEN QUESTIONS:
How does heterogeneous multi-robot coordination scale with the number and diversity of robot types? Can learned communication protocols provide any safety or interpretability guarantees? How should credit assignment work when multiple heterogeneous robots contribute to a shared task outcome?""",

    49: """STRENGTHS:
GR-2 built on the GR-1 approach by scaling up the integration of video generation and robot policy learning. The key idea of pre-training on large-scale video data to learn world dynamics and then fine-tuning for robot control is well-motivated. GR-2 demonstrated improved performance on humanoid manipulation tasks and showed that video pre-training provides useful representations for control. The scaling behavior — larger video models yielding better robot policies — provided evidence for the world model pre-training hypothesis.

LIMITATIONS:
The evaluation is narrow, focusing on a limited set of humanoid manipulation tasks. The connection between video prediction quality and policy quality is correlational rather than causal — it is not clear whether the video pre-training is providing physics understanding or merely good visual features. The method requires substantial compute for video pre-training. The real-world demonstrations are short-horizon and do not test robustness, recovery from errors, or long-horizon task completion. Direct comparison with methods using standard visual pre-training (CLIP, DINOv2) rather than video generation would strengthen the claims.

COMMUNITY RECEPTION:
Viewed as a reasonable iteration on the video-generation-for-robotics theme. The scaling results attracted attention, but the community noted the same concerns that apply to all video-to-policy approaches: does video generation actually capture the right inductive biases for control? The work contributed to the ongoing debate about whether video pre-training or visual-language pre-training provides better foundations for robot policies.

OPEN QUESTIONS:
Is video pre-training fundamentally better than image-language pre-training for robot control, or are they complementary? What aspects of video dynamics actually transfer to policy learning — is it object permanence, physics intuition, or just visual feature quality? Can the video pre-training approach match the data efficiency of methods that directly learn from demonstrations?""",

    50: """STRENGTHS:
FLOWER aims to democratize generalist robot policy development by providing an open, accessible framework for training and deploying generalist policies. The emphasis on lowering the barrier to entry — providing pretrained models, standardized training pipelines, and evaluation protocols — addresses a real need in a field where reproducing results requires substantial engineering effort and compute. The focus on practical usability and community adoption reflects a mature understanding of what makes research impactful.

LIMITATIONS:
Democratization frameworks risk trading performance for accessibility. The standardized pipeline may constrain architectural innovation, pushing users toward a particular design paradigm that may not be optimal. The evaluation of "democratization" is inherently difficult — usage metrics and community adoption are lagging indicators. The pretrained models may encode biases from training data that are hard for downstream users to identify or correct. The performance gap between FLOWER models and state-of-the-art specialized models needs to be clearly characterized.

COMMUNITY RECEPTION:
The open-source and democratization angle was well-received, particularly by researchers at smaller labs without access to large compute clusters. The community appreciated the focus on reproducibility and accessibility. However, more established labs questioned whether the standardized pipeline was too constraining and whether the models were competitive with purpose-built systems. FLOWER represents an important trend toward open infrastructure for robot learning.

OPEN QUESTIONS:
Can democratized frameworks keep pace with rapidly evolving architectural innovations? How should pre-trained models be distributed responsibly, given potential safety concerns with robot policies? What is the minimum compute and data budget needed to fine-tune FLOWER models for new tasks and embodiments?""",

    51: """STRENGTHS:
SmolVLA directly addresses one of the most important practical challenges in VLA deployment: computational efficiency. By designing a compact VLA architecture that can run on edge devices with limited compute, SmolVLA makes the VLA paradigm practical for real robots that cannot afford to run 7B+ parameter models in real-time. The careful distillation and architecture search to maintain performance while dramatically reducing model size is a valuable engineering contribution. Demonstrating competitive performance at a fraction of the compute budget challenges the assumption that VLA performance requires large models.

LIMITATIONS:
The efficiency gains inevitably come with some performance loss, particularly on tasks requiring complex reasoning or long-horizon planning where larger models excel. The distillation quality is upper-bounded by the teacher model's capabilities. The architecture optimizations may be specific to current hardware and become less relevant as edge compute improves. The evaluation may not adequately test the boundary cases where model capacity matters — simple pick-and-place tasks may not reveal the performance gap with larger models that more complex tasks would expose.

COMMUNITY RECEPTION:
Enthusiastically received by the practical robotics community, which has long been frustrated by the compute requirements of state-of-the-art VLAs. SmolVLA demonstrated that the field was taking deployment seriously, not just scaling for benchmark numbers. Academic labs with limited GPU access found it particularly useful. The work contributed to a healthy counter-narrative to the "bigger is better" trend, showing that careful architecture design can substitute for raw scale.

OPEN QUESTIONS:
What is the minimum model size that retains meaningful language grounding and visual reasoning for manipulation? Can efficiency-focused architectures close the gap with large models as they scale, or is there a fundamental accuracy-efficiency frontier? How should we evaluate VLAs for deployment — wall-clock time per action, memory footprint, or energy consumption?""",

    52: """STRENGTHS:
ST4VLA (Spatial-Temporal for VLA) tackles the important problem of incorporating both spatial and temporal information into vision-language-action models. Most VLAs process single frames or short windows, losing the temporal context that is critical for understanding dynamic scenes — object velocities, ongoing interactions, and task progress. By explicitly designing spatial-temporal attention mechanisms for VLA architectures, ST4VLA improved performance on tasks requiring temporal reasoning (e.g., catching moving objects, responding to dynamic scenes).

LIMITATIONS:
The additional computational cost of spatial-temporal attention is significant and may negate some of the practical benefits. The temporal context window is fixed and may not capture the right timescale for all tasks — some require millisecond-level dynamics, others require minute-level context. The evaluation focused on tasks specifically designed to require temporal reasoning, potentially overstating the general benefit. The interaction between spatial-temporal features and language conditioning needs more analysis — it is unclear whether the temporal features are actually being used for language-grounded reasoning or just low-level motion prediction.

COMMUNITY RECEPTION:
Acknowledged as addressing a genuine gap in VLA architectures. The community recognized that temporal reasoning is important but noted that the added complexity may not be justified for the majority of manipulation tasks that can be solved with reactive, single-frame policies. The work contributed to the discussion about what information VLAs actually need versus what architectural complexity is justified by empirical gains.

OPEN QUESTIONS:
How much temporal context do VLAs actually need for different task categories? Can temporal reasoning be achieved more efficiently through recurrent state or memory mechanisms rather than explicit spatial-temporal attention? Does the benefit of temporal reasoning increase with task complexity, suggesting it will become more important as the field tackles harder problems?""",

    53: """STRENGTHS:
DROID provides a large-scale, in-the-wild robot manipulation dataset collected across diverse real environments — not just lab settings. With data from 50+ environments, multiple robot setups, and a wide variety of manipulation tasks, DROID addresses the critical need for out-of-distribution robustness that lab-collected datasets cannot provide. The dataset design, with multiple camera angles, wrist cameras, and detailed annotations, is thoughtful. The demonstration that training on DROID data improves generalization to novel environments is a valuable empirical contribution.

LIMITATIONS:
The collection methodology introduces biases from operator skill variation — different human teleoperators have different manipulation strategies, speeds, and error patterns, creating distributional noise that is difficult to separate from genuine behavioral diversity. The quality control challenge at scale means some demonstrations contain suboptimal or failed behaviors. The hardware standardization (specific robot arm, specific gripper) limits cross-embodiment applicability. Despite the "in-the-wild" framing, the environments are still primarily indoor settings with structured furniture and common household objects, not truly unconstrained environments.

COMMUNITY RECEPTION:
Valued highly as a step toward more ecologically valid robot learning datasets. The emphasis on environmental diversity (as opposed to task diversity in OXE) was seen as complementary to existing efforts. Researchers appreciated the careful dataset documentation and the open release. The operator skill variation issue was identified early by users training on the data, leading to community discussions about data filtering and quality weighting strategies. DROID has become a standard component of multi-dataset training pipelines.

OPEN QUESTIONS:
How should operator skill variation be handled — filtering, weighting, or modeling it as a latent variable? Can in-the-wild data collection be scaled further without sacrificing quality? What is the right balance between environmental diversity and data quality for training generalizable policies? Should robot datasets adopt standardized collection protocols, or does protocol diversity contribute to robustness?""",

    54: """STRENGTHS:
Dobb-E provided an impressively accessible pipeline for collecting and learning household manipulation tasks. The use of an iPhone-mounted stick as a demonstration tool — leveraging the phone's camera and IMU for pose estimation — dramatically lowered the barrier to data collection compared to traditional teleoperation setups. The ability to learn manipulation policies from just a few minutes of demonstrations in a specific home environment is a powerful practical contribution. The open-source release of the full pipeline, from data collection to policy deployment, was exemplary.

LIMITATIONS:
The iPhone-based demonstration tool, while creative, provides noisy pose estimates that limit the precision of learned behaviors. The approach works best for coarse manipulation tasks (opening doors, drawer manipulation) but struggles with precision tasks (inserting keys, threading). The policies are trained per-environment and do not generalize across homes — each new home requires new demonstrations. The dependence on the specific iPhone hardware and software pipeline creates reproducibility concerns as iOS updates change sensor APIs.

COMMUNITY RECEPTION:
Widely celebrated for its practicality and accessibility. The robotics community loved the "anyone can collect robot data" narrative, and several labs reproduced and extended the pipeline. The work was seen as an important contribution to the democratization of robot learning research. However, some researchers noted that the simplicity of the data collection came at the cost of data quality, and that the per-environment training paradigm does not address the generalization challenge that the field is fundamentally trying to solve.

OPEN QUESTIONS:
Can the Dobb-E data collection approach be combined with foundation model pre-training to achieve few-shot adaptation to new homes? How should low-quality, easy-to-collect demonstration data be combined with high-quality, expensive-to-collect data? What is the minimum quality threshold for demonstration data to be useful for policy learning?""",

    55: """STRENGTHS:
RoboCasa represents a significant expansion of simulated benchmark environments for everyday manipulation tasks. By providing a large-scale, diverse set of kitchen and household environments with realistic object sets, articulated furniture, and task specifications, RoboCasa addresses the need for benchmarks that go beyond simple tabletop manipulation. The integration with existing simulation frameworks (robosuite) and the inclusion of diverse kitchen layouts and appliance configurations enable systematic evaluation of generalization. The procedural generation of environment variations allows testing at scale.

LIMITATIONS:
The sim-to-real gap remains the fundamental challenge for any simulation benchmark — policies that succeed in RoboCasa may fail in real kitchens due to unmodeled physics, sensor discrepancies, and visual domain differences. The "everyday tasks" are still simplified versions of real household tasks — the complexity of actual cooking, cleaning, and organizing involves many more edge cases, tool uses, and physical interactions than the benchmark captures. The object and environment diversity, while improved, still does not match the long tail of real-world variation. Physics fidelity for deformable objects, liquids, and granular materials is limited.

COMMUNITY RECEPTION:
Appreciated as a needed benchmark expansion beyond tabletop manipulation. The robotics community recognized that evaluation on simple pick-and-place tasks was insufficient and that household manipulation benchmarks were needed. RoboCasa became a standard evaluation environment alongside LIBERO and RLBench. However, some researchers cautioned against over-optimizing for simulation benchmarks at the expense of real-world validation — simulation performance does not reliably predict real-world performance.

OPEN QUESTIONS:
How should simulation benchmarks be validated — what evidence is needed to show that simulation results predict real-world performance? Can procedural generation of environments capture the distributional properties of real-world household variation? What aspects of everyday tasks are fundamentally hard to simulate (deformables, fluids, multi-step cooking) and how should benchmarks handle these?""",

    56: """STRENGTHS:
LIBERO provides a well-structured benchmark suite specifically designed to evaluate knowledge transfer in lifelong robot learning. The benchmark's decomposition into separate sub-suites — LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long — enables fine-grained diagnosis of what types of knowledge transfer succeed or fail. The careful control of task variables (changing only spatial arrangement, or only objects, or only goals) is scientifically valuable for understanding generalization. The standardized evaluation protocol and open-source implementation have made LIBERO a widely-used benchmark.

LIMITATIONS:
The tasks, while well-designed for controlled experiments, are relatively simple compared to real-world manipulation challenges. The tabletop setting with a fixed robot arm limits ecological validity. The benchmark's focus on lifelong learning assumes a sequential task presentation that may not match real deployment scenarios where tasks arrive in arbitrary orders. The simulated physics, while adequate for the tasks included, does not stress-test methods on contact-rich or deformable object manipulation. The benchmark may favor methods that are good at the specific types of transfer tested while missing other important aspects of generalization.

COMMUNITY RECEPTION:
LIBERO quickly became one of the standard benchmarks for evaluating VLA and policy learning methods, alongside RLBench and RoboCasa. The research community valued the controlled experimental design, which allows meaningful comparisons between methods on specific transfer axes. The benchmark's popularity has led to extensive leaderboard competition, though some researchers worry that methods are being over-fit to LIBERO's specific task distributions. The separate sub-suites were particularly appreciated for enabling nuanced analysis of model capabilities.

OPEN QUESTIONS:
Is strong performance on LIBERO predictive of real-world generalization, or are the benchmark's tasks too clean and controlled? How should lifelong learning benchmarks handle the evaluation of catastrophic forgetting when the task distribution is non-stationary? Should benchmarks include explicit "distractor" tasks that share visual similarity but require different skills, to test whether models learn generalizable representations or surface-level correlations?""",
}


def seed():
    admin = get_admin_client()
    for number, review in REVIEWS.items():
        admin.table("papers").update({"peer_reviews": review}).eq("number", number).execute()
        print(f"  Seeded peer review for paper #{number}")


if __name__ == "__main__":
    seed()
    print("Done!")
