The following is a detailed synthesis of the current changes require for adding hierarchical processing to NuCLR:

At a high level, this is a summary of the addition:

> We modify NuCLR’s single-resolution temporal pathway into a configurable multistage temporal hierarchy. Neural activity is first tokenized exactly as in NuCLR, then processed through successive fine-to-coarse temporal stages, with local concat-and-project merging between stages. At each temporal scale, permutation-equivariant spatial attention is preserved to incorporate population context. Each stage produces a learned neuron-level summary via attention pooling, and these scale-specific summaries are fused with a cross-scale transformer to produce the final neuron representation used in the original NuCLR contrastive objective.


> We keep NuCLR’s input pipeline, permutation-equivariant population modeling, and contrastive objective, but replace its single-resolution temporal pathway with an explicit **multistage temporal hierarchy**, then fuse **scale-specific neuron representations** with cross-scale attention.

NuCLR currently does: patch tokens per neuron → temporal layers → alternating spatial/temporal layers → final mean-pool over time to one neuron embedding. 
Your refinement changes the middle and late parts of that pipeline.

---

# 1. Keep the input tokenization

NuCLR already has a clean front end:

* bin spikes,
* patch into non-overlapping temporal chunks,
* linearly project each patch into a latent token,
* build a per-neuron sequence of patch tokens. 

The input to the encoder stays something like:

[
Z_n^{(0)} \in \mathbb{R}^{P \times D}
]

for neuron (n), where:

* (P) = number of temporal patches in the context window
* (D) = model dimension

This is good because:

* it preserves NuCLR’s data interface,
* it avoids confounding the experiments with new tokenization choices,
* and it isolates the hypothesis to **temporal hierarchy**, not front-end preprocessing.

---

# 2. Replace the flat temporal stack with explicit temporal stages

This is the main architectural change.

## Current NuCLR behavior

NuCLR processes each neuron’s patch sequence through temporal transformer layers, then through spatiotemporal layers, but the temporal resolution stays conceptually flat until the final pooling step. 

## New behavior

Instead of one temporal sequence resolution all the way through, define **multiple stages**, each corresponding to a distinct temporal scale.

Example:

* **Stage 1:** fine temporal scale
* **Stage 2:** intermediate temporal scale
* **Stage 3:** coarse temporal scale

Each stage should:

* take a sequence of tokens at its current temporal resolution,
* run temporal modeling,
* run spatial attention across neurons at that same temporal resolution,
* output refined tokens for that scale.

Then, downsample temporally and feed the coarser sequence to the next stage.

So instead of:

[
P \rightarrow P \rightarrow P \rightarrow \text{pool}
]

we get something like:

[
P \rightarrow P/2 \rightarrow P/4 \rightarrow \text{fuse scale summaries}
]

## Why this matters

The goal of this is to enforce that deeper stages represent larger temporal receptive fields. That is the real hierarchy. Without the merge/downsampling, “deeper layer” does not necessarily mean “coarser timescale.”

### What to expose in config

We should keep this configurable, where we have something like:

* `num_temporal_stages: 3`
* `stage_depths: [a, b, c]`
* `stage_merge_factors: [2, 2]`
* `stage_dims: [D, D, D]` initially, or allow scaling later
* `stage_temporal_rope: true/false`

A minimal first version can keep all stage dims equal. The most important config is the **number of stages** and **merge factor per stage**.

---

# 3. For temporal merging: local token merge via concat + projection

## What it should do

Between stages, take adjacent temporal tokens and merge them.

If the current stage output for neuron (n) is:

[
Z_n^{(s)} \in \mathbb{R}^{P_s \times D}
]

then with merge factor 2, create:

[
\tilde{Z}_n^{(s+1)} \in \mathbb{R}^{(P_s/2) \times D}
]

by combining pairs:
[
[z_{n,2i}, z_{n,2i+1}] \rightarrow \tilde{z}_{n,i}
]

using:

* concatenation to dimension (2D)
* linear projection (2D \rightarrow D)

So in code terms, for each local group:

1. concatenate neighboring tokens
2. apply linear layer
3. maybe apply norm / activation

## Why this is good

* simple
* stable
* learnable
* preserves more information than raw averaging
* easy to reason about

It also creates a clear interpretation:
each coarser token is a learned summary of a larger temporal chunk.

## Practical implementation detail

If (P_s) is odd, either:

* pad one token,
* or drop last token,
* or handle final window specially

Padding is usually cleanest.

### What to expose in config

Add:

* `merge_type: "concat_linear"`
* `merge_factor: 2` or per-stage list
* `merge_norm: true`
* `merge_activation: gelu` or none

We could at least start with:

* concat
* linear
* layernorm

---

# 4. Make scale identity explicit and configurable

## What this means

Stages shouldn't just be “layers 1, 2, 3.”
They correspond to **explicit temporal abstractions**.

For example, if:

* bin size = 20 ms
* patch length = 2 bins = 40 ms
* stage merge factor = 2 each time

then:

* Stage 1 token represents ~40 ms chunk
* Stage 2 token represents ~80 ms chunk
* Stage 3 token represents ~160 ms chunk

More importantly, after attention, each stage’s *effective receptive field* is larger than those raw chunk sizes.

That gives a meaningful narrative:

* Stage 1 captures local events
* Stage 2 captures short motifs
* Stage 3 captures broader temporal context

## In the model

At minimum, the architecture itself does this via downsampling.

Optionally, there could be a **scale embedding** to the stage summaries before fusion:
[
h_n^{(s)} + e^{(s)}_{\text{scale}}
]
so the cross-scale attention block knows which representation came from which scale.

That might be worth doing.

### What to expose in config

Add:

* `use_scale_embeddings: true`
* `num_temporal_stages`
* `stage_merge_factors`
* maybe computed `stage_patch_spans_ms`

This makes analysis easier later too.

---

# 5. Spatial attention at every scale

## Current NuCLR principle

NuCLR’s core design is that neuron identity should be informed by population context, and it uses alternating temporal and spatial attention to let neurons interact without fixed ordering assumptions. 

## Mmodified version

Preserve that principle **at each temporal scale**.

So each stage should have:

1. temporal modeling within each neuron sequence
2. spatial modeling across neurons at aligned time indices

That means Stage (s) operates on tokens shaped roughly like:
[
(N, P_s, D)
]
where:

* (N) = number of neurons
* (P_s) = number of temporal positions at that stage
* (D) = embedding dim

Then:

* temporal block attends across (P_s) for each neuron independently
* spatial block attends across (N) for each time index independently

This preserves permutation-equivariance across neurons, consistent with NuCLR’s existing architecture. 

## Recommended stage block pattern

A good default stage could be:

* temporal self-attention
* temporal FFN
* spatial self-attention
* spatial FFN

possibly repeated `stage_depths[s]` times.

So a stage becomes a mini NuCLR block at one temporal scale.

## Why this is better than delaying spatial attention

Because population context may matter at multiple timescales:

* local synchrony / short co-activation patterns,
* medium-timescale ensemble motifs,
* broader contextual state

By keeping spatial interaction at every scale, those relationships are to be represented hierarchically too.

### What to expose in config

Add:

* `stage_depths`
* `temporal_heads_per_stage`
* `spatial_heads_per_stage`
* `spatial_every_stage: true`
* `block_order: "temp_then_spatial"`

For the first version, keep block order fixed.

---

# 6. Replace naive final mean-pooling with a learned stage-summary mechanism


## What this stage summary needs to do

At the end of each stage, for neuron (n), you have a token sequence:
[
Z_n^{(s)} \in \mathbb{R}^{P_s \times D}
]

You need to convert that sequence into one stage-level summary vector:
[
h_n^{(s)} \in \mathbb{R}^{D_h}
]

That summary is what gets passed to cross-scale fusion.

The question is: how should (h_n^{(s)}) be formed?

---

## Why plain mean-pooling may be too weak

Mean-pooling assumes all time positions contribute equally and linearly.

That may be okay in vanilla NuCLR’s final readout, but for the hierarchical design, the stage summaries are central. If you summarize each stage badly, then the cross-scale fusion block is operating on poor inputs.

So yes, this is an important step.

---

## Better option than plain mean-pooling could be learned attention pooling


For each stage, learn a query vector (q^{(s)}), or use a small attention-pooling head that computes weights over the temporal positions:
[
\alpha_{n,p}^{(s)} = \text{softmax}(f(z_{n,p}^{(s)}))
]
and then:
[
h_n^{(s)} = \sum_p \alpha_{n,p}^{(s)} z_{n,p}^{(s)}
]

This is better than mean-pooling because:

* the model can emphasize informative moments,
* it stays sequence-to-vector,
* it is lightweight,
* and it is easy to inspect.

So each stage summary module would be:

[
h_n^{(s)} = \text{AttnPool}(Z_n^{(s)})
]

possibly followed by a projection:
[
\hat{h}_n^{(s)} = W_s h_n^{(s)}
]

to normalize all stage summaries into the same fusion dimension.

### What to expose in config

Add:

* `stage_summary_type: "attention_pool"`
* other options: `"mean"`, `"cls"`, `"mean_proj"`
* `fusion_dim`
* `use_stage_specific_summary_heads: true`

This is one of the most important config choices.

---

# 7. Cross-scale attention over stage summaries

## What the inputs are

After summarization, each neuron (n) has:
[
h_n^{(1)}, h_n^{(2)}, \dots, h_n^{(L)}
]

where each (h_n^{(s)}) is the stage-level summary for one temporal scale.

Stack them into:
[
H_n \in \mathbb{R}^{L \times D_f}
]

Then run a small transformer or attention module over those (L) scale tokens.

## What this block learns

This block should learn:

* which scales are most informative,
* whether fine and coarse scales are complementary,
* whether certain scales should be suppressed for some neurons,
* and how to combine scale evidence into the final neuron representation.

This is much stronger than concatenation or averaging because it gives a adaptive fusion.

### What to expose in config

Add:

* `cross_scale_depth`
* `cross_scale_heads`
* `cross_scale_dim`
* `cross_scale_use_cls: true/false`
* `use_scale_embeddings: true`
