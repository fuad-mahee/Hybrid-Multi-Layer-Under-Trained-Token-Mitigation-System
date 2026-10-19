# Initial HMUTTMS run: recorded observations and limitations

This repository-local record was prepared on 19 October 2026 from the **First run** tab of the [project notes](https://docs.google.com/document/d/1l5_AoTypne2BialXow6xIQ0NdDVoBseW7wpJX4OJyAc/edit?tab=t.kff3jhdnw1yz), with interpretation checked against the implementation. It is a curated transcription of the relevant observations, not a raw-log export or a new experiment. The linked document was readable without signing in when checked on that date. Committing this file preserves the stated observations and caveats in repository history.

## Recorded configuration

| Item | Recorded value |
|---|---|
| Target model | Qwen2.5-7B-Instruct |
| Reference model | Qwen2.5-14B-Instruct |
| Vocabulary size | 152,064 |
| Detection candidates | 2,979 |
| Runtime track with recorded attempts | Track 3: embedding injection |
| Embedding source in the initial runtime tests | Contextual, neighboring-token embeddings from the target model |

The reference model was used for cross-validation. The observed embedding intervention should not be described as successful embedding transfer from the larger model. Llama and Kimi examples elsewhere in the code are alternative configurations, not evidence that those models were used in this run.

## Observations carried forward

| Observation in the development notes | Supported interpretation |
|---|---|
| 2,979 candidates out of 152,064 vocabulary entries | The detection procedure flagged these entries. This is not independently verified prevalence of under-training. |
| Eight successful Track 3 operations | The neighboring-token embedding path reportedly constructed and staged patches for eight attempts. Operational success does not establish eight correct semantic repairs. |
| 20 of 21 outputs passed the quality checker (95.24%) | A pass count under a surface-level heuristic, not task accuracy or human-verified coherence. |
| No recorded attempts for Tracks 1, 2, or 4 | These tracks have no demonstrated effectiveness in this runtime sample. |

The quality checker uses length, repetition, and character diversity. Those measures do not establish factual correctness, exact-string preservation, or task completion. The fallback uses surrounding input-token embeddings; references in the historical notes to semantically nearest vocabulary neighbors should not be treated as an accurate description of that implementation.

## Artifact and threshold issues

**Bloom filter.** The notes describe a mismatch between an older filter at `hmuttms/bloom_filter.bin` and a rebuilt registry-based filter at `data/mitigation_system/bloom_filter.bin`. The recorded evaluation therefore cannot establish the corrected pipeline's detection performance. Runtime and evaluation must read the same newly generated registry and filter. Fixing a path does not prove zero false positives.

**Threshold.** The notes describe lowering the reference-model pass threshold from 0.70 to 0.60 to admit scores of 2/3. On 19 October 2026, code inspection found `DecisionConfig.b_pass_above = 0.60` but the Layer 2 CLI still defaulted to 0.70 and passed that value into the configuration. The accompanying correction makes the CLI obtain its defaults from `DecisionConfig`; the actual comparison remains strict (`B > 0.60`). The effective threshold used in any historical result must be established from that run's configuration or printed summary, not inferred from the corrected source.

**Inconsistent classification narrative.** The source notes contain classification counts alongside proposed follow-up changes, including inconsistent descriptions of when substitutable cases were available. Those totals and the reported latency values are not promoted here as corrected benchmark results. They require a fresh run with one recorded configuration and consistent artifact paths.

## What remains to be established

No corrected end-to-end evaluation is supplied by this documentation update. A reproducible result should include the code revision, exact model/tokenizer revisions, environment, probe settings, effective thresholds, prompts, random seeds where applicable, and the generated JSONL files, registry, Bloom filter, and runtime events. It should compare mitigated and unmodified outputs on the same tasks, with independent labels and semantic outcome measures.

This record preserves the initial observations without treating historical explanatory prose, projected improvements, or successful patch construction as evidence of a general mitigation benefit.
