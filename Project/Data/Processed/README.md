# Processed data

Result summaries and cached artefacts produced by the notebooks. These files are
what every number in the thesis is computed from, and they are the only surviving
record of the LLM runs, since all four models have since been deprecated by the
provider.

| File | Contents |
|---|---|
| `integrity_report.json` | Output of the verification notebook (48 checks) |
| `ner_pipeline_summary.json` | spaCy, Stanza, Flair entity counts and agreement |
| `ner_llm_summary.json` | LLM entity extraction results |
| `ner_comparison_summary.json` | Pipeline versus LLM comparison |
| `nb13_scaling_summary.json` | Model-scale comparison |
| `nb13b_summary.json` | Translation ablation, F1 and significance tests |
| `nb14_summary.json` | Dual consensus references, all systems scored against both |
| `nb15_summary.json` | Topic classification, NLI and LLM |
| `ablation_summary.json` | German versus English input, entity-set overlap |
| `error_analysis_summary.json` | Disagreement inspection |
| `topic_modeling_summary.json`, `topic_summary_v2.json` | BERTopic runs |
| `thesis_final_summary.json` | Consolidated figures reported in the thesis |
| `topic_model_v2/`, `bertopic_model` | Saved BERTopic models |
| `topic_embeddings*.npy` | Cached sentence embeddings |
