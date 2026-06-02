# Experiment 01.5: MoE sensitivity (Llama 4 Maverick)

This experiment extends the layer-sensitivity idea from exp01 to mixture-of-experts
models, using Llama 4 Maverick (17B active parameters, 128 experts per block) as the
target. It is designed and scoped but not yet run, because I do not currently have the
compute to run it. The design is ready to execute the day the hardware is available.

## Why it needs its own experiment

The dense ablator from exp01 does not transfer to MoE. In a dense model, "ablate one
Linear layer" has a single clear meaning. In an MoE block with 128 experts plus a router,
it does not, so this experiment defines three distinct ablation modes:

- Single-expert: quantize one expert and leave the other 127 at fp16. This measures how
  sensitive the model is to the specialization of an individual expert.
- Router: quantize only the gating network. This measures routing fidelity under
  quantization, which has no analog in a dense model.
- Whole-MoE-block: quantize all 128 experts in one block at once. This is the closest
  analog to dense-model layer ablation, and it is what makes the MoE results comparable to
  the dense findings in exp01.

The hypothesis worth testing is whether sensitivity concentrates in the router and a few
experts, which would make MoE models cheaper to protect than dense models of comparable
quality, or whether it spreads across many experts, which would make them harder.

## Why it is blocked

Maverick is roughly 400B total parameters, about 800 GB in fp16. A clean fp16 baseline
needs something on the order of 8x H100 SXM with NVSwitch. With 4x H100 it is possible to
load int4 only and give up the fp16 baseline. At on-demand H100 rates a single full sweep
lands somewhere around 80 to 150 dollars. That is out of reach at the moment.

## What still needs building

- A MoE-aware `Ablator` subclass that understands experts and routers.
- The three ablation primitives above (single-expert, router, whole-block).
- Maverick loading across 8 GPUs.
- The three-mode sweeps.
- A comparison back against the dense Tier 1 findings from exp01.