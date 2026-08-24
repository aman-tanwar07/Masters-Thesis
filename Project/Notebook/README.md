# Notebooks

Numbered in the order they were run. Each notebook is self-contained: it installs
its own dependencies in the first cell, mounts Google Drive, and sets
`PROJECT_ROOT` at the top.

Outputs are committed deliberately. All four LLMs were deprecated by the provider
during the writing of the thesis, so the stored outputs are the only remaining
record of those runs.

A description of each notebook is in the repository README.

## Notes on running

- `qwen/qwen3-32b` requires a `/no_think` prefix to suppress its reasoning output,
  and a larger token budget for NER.
- Inter-request sleeps are set per model in `config.yaml` to stay inside free-tier
  rate limits. Llama 70B needs the longest (35s for NER, 45s for topic).
- Runs checkpoint every 10 articles so that a session timeout does not lose work.
