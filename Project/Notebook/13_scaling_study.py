# ============================================================
# NOTEBOOK 13: LLM SCALING STUDY — JACCARD SIMILARITY CURVE
# ============================================================
# Purpose : Compare pipeline NER systems against LLMs of varying
#           scale using Jaccard similarity (no fixed gold standard).
#           Approach 1 from gold-standard-bias discussion:
#           plot pipeline-LLM convergence across model scales.
#
# New models to run  : Llama 3.2 3B, Llama 3.1 8B, Gemma2 9B (Groq)
#                      Claude Haiku (Anthropic)
# Already done       : Qwen 2.5 7B (EN), Llama 3.3 70B (EN+DE)
#
# Checkpoint policy  : saved every SAVE_EVERY articles per model.
#                      On ANY unrecoverable error the run stops,
#                      checkpoint is flushed, and a clear message
#                      is printed.  Re-running resumes from last
#                      saved position automatically.
# ============================================================


# ============================================================
# CELL 1 : INSTALLATION
# Run once per Colab session.
# ============================================================

# !pip install -q groq anthropic
# !pip install -q 'numpy>=2.0'   # must be last


# ============================================================
# CELL 2 : IMPORTS & CONFIGURATION
# ============================================================

import os, json, pickle, time, re, traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from google.colab import drive, userdata

# Mount Drive
drive.mount('/content/drive', force_remount=False)

# Paths
ROOT        = Path('/content/drive/MyDrive/thesis')
DATA_PROC   = ROOT / 'Project/Data/Processed'
FIGURES_DIR = ROOT / 'Project/Outputs/Figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# API tokens
GROQ_TOKEN      = userdata.get('GROQ_TOKEN')
ANTHROPIC_TOKEN = userdata.get('ANTHROPIC_TOKEN')   # may be None if not set yet

# Checkpointing config
SAVE_EVERY               = 10    # flush checkpoint every N articles
MAX_CONSECUTIVE_FAILURES = 5     # stop run if this many articles fail in a row
MAX_FAILURE_RATE         = 0.20  # stop run if >20% of articles have failed

# Rate-limit / retry config
GROQ_SLEEP_BETWEEN = 2.0         # seconds between successful Groq calls
RETRY_DELAYS       = [4, 8, 16]  # exponential back-off on 429 / timeout

# Target Groq models (70B already done in nb09 - loaded, not re-run)
GROQ_TARGET_MODELS = {
    "llama-3.2-3b-preview" : {"scale_B": 3,  "family": "llama", "input": "en"},
    "llama-3.1-8b-instant" : {"scale_B": 8,  "family": "llama", "input": "en"},
    "gemma2-9b-it"         : {"scale_B": 9,  "family": "gemma", "input": "en"},
}

ANTHROPIC_TARGET_MODELS = {
    # scale_B=20 is a conservative proxy for Haiku's position on the curve
    "claude-haiku-4-5-20251001": {"scale_B": 20, "family": "claude", "input": "en"},
}

# NER prompt — same structure as nb04/nb09 for consistency
NER_SYSTEM_PROMPT = """You are a named entity recognition (NER) system.
Extract ALL named entities from the text provided.

Return ONLY a valid JSON object — no markdown, no explanation, no preamble:
{
  "entities": [
    {"text": "<entity surface form>", "label": "<TYPE>"}
  ]
}

Use exactly these four labels:
  PER   -- person names
  LOC   -- locations, countries, cities, geographical features
  ORG   -- organisations, companies, institutions
  MISC  -- other named entities (events, products, languages, nationalities, etc.)

If no entities are found return:  {"entities": []}"""

NER_USER_TEMPLATE = "Extract named entities from the following text:\n\n{text}"

print("Configuration loaded")
print(f"  DATA_PROC  : {DATA_PROC}")
print(f"  FIGURES    : {FIGURES_DIR}")
print(f"  Groq token : {'SET' if GROQ_TOKEN else 'MISSING'}")
print(f"  Anthropic  : {'SET' if ANTHROPIC_TOKEN else 'NOT SET — Haiku will be skipped'}")


# ============================================================
# CELL 3 : CHECK AVAILABLE GROQ MODELS
# ============================================================

from groq import Groq

groq_client = Groq(api_key=GROQ_TOKEN)

print("Querying Groq for live models...")
try:
    available_groq = {m.id for m in groq_client.models.list().data}
    print(f"  {len(available_groq)} models available\n")
except Exception as e:
    available_groq = set()
    print(f"  Could not list models: {e}")

confirmed, skipped = {}, {}
for model_id, meta in GROQ_TARGET_MODELS.items():
    if model_id in available_groq:
        confirmed[model_id] = meta
        print(f"  CONFIRMED : {model_id}  ({meta['scale_B']}B)")
    else:
        skipped[model_id] = meta
        print(f"  NOT FOUND : {model_id}  -- will skip")

if ANTHROPIC_TOKEN:
    print(f"\n  CONFIRMED : claude-haiku-4-5-20251001  (Anthropic)")
else:
    print(f"\n  SKIPPED   : claude-haiku-4-5-20251001  (set ANTHROPIC_TOKEN to enable)")


# ============================================================
# CELL 4 : LOAD EXISTING DATA
# ============================================================

print("Loading pipeline results...")
pipeline_df = pd.read_pickle(DATA_PROC / 'ner_pipeline_results.pkl')
print(f"  shape: {pipeline_df.shape}")
print(f"  columns: {list(pipeline_df.columns)}")

# NOTE: adjust these column names if yours differ
PIPELINE_ENTITY_COLS = {
    'spaCy'  : 'spacy_entities',
    'Stanza' : 'stanza_entities',
    'Flair'  : 'flair_entities',
}

for name, col in PIPELINE_ENTITY_COLS.items():
    status = "OK" if col in pipeline_df.columns else "MISSING -- adjust PIPELINE_ENTITY_COLS"
    print(f"  {name}: '{col}' -- {status}")

# Existing LLM checkpoints (Qwen 7B, Llama 70B)
EXISTING_LLM_FILES = {
    "qwen-2.5-7b-instruct"    : {"file": DATA_PROC / 'ner_llm_checkpoint.pkl',
                                  "scale_B": 7,  "family": "qwen",  "input": "en"},
    "llama-3.3-70b-versatile" : {"file": DATA_PROC / 'ner_llama70b_checkpoint.pkl',
                                  "scale_B": 70, "family": "llama", "input": "en"},
}

existing_llm_results = {}
for model_id, info in EXISTING_LLM_FILES.items():
    fpath = info["file"]
    if fpath.exists():
        with open(fpath, 'rb') as f:
            ckpt = pickle.load(f)
        existing_llm_results[model_id] = ckpt
        n_ok = len([r for r in ckpt.get('results', []) if r.get('status') == 'ok'])
        print(f"  Loaded {model_id}: {n_ok} articles")
    else:
        print(f"  NOT FOUND: {model_id} at {fpath}")

# Working article set: same 183 used in nb04/nb09
if "qwen-2.5-7b-instruct" in existing_llm_results:
    ARTICLE_IDS_183 = [
        r['article_id'] for r in existing_llm_results["qwen-2.5-7b-instruct"].get('results', [])
        if r.get('status') == 'ok'
    ]
else:
    ARTICLE_IDS_183 = list(pipeline_df['article_id'].unique())
print(f"\n  Working set: {len(ARTICLE_IDS_183)} articles")

# article_id -> translated_text lookup
TEXT_COL = 'translated_text'   # adjust if needed
if TEXT_COL in pipeline_df.columns:
    id_to_text = dict(zip(pipeline_df['article_id'], pipeline_df[TEXT_COL]))
    print(f"  Text column '{TEXT_COL}' ready")
else:
    print(f"  Column '{TEXT_COL}' not found -- check pipeline_df columns above")
    id_to_text = {}


# ============================================================
# CELL 5 : CHECKPOINT UTILITIES
# ============================================================

def ckpt_path(model_id: str) -> Path:
    safe_id = re.sub(r'[/:\\]', '_', model_id)
    return DATA_PROC / f'nb13_{safe_id}_checkpoint.pkl'

def load_checkpoint(model_id: str) -> dict:
    """Load existing checkpoint or return a fresh empty one."""
    fp = ckpt_path(model_id)
    if fp.exists():
        with open(fp, 'rb') as f:
            ckpt = pickle.load(f)
        done = len([r for r in ckpt['results'] if r['status'] == 'ok'])
        fail = len(ckpt['failed_ids'])
        print(f"  Resuming {model_id}: {done} done, {fail} failed previously")
        return ckpt
    return {
        'model_id'   : model_id,
        'results'    : [],
        'failed_ids' : [],
        # stop_reason: None | 'complete' | 'rate_limit' |
        #              'consecutive_failures' | 'high_failure_rate' | 'other_error'
        'stop_reason': None,
        'started_at' : datetime.now().isoformat(),
        'updated_at' : None,
    }

def save_checkpoint(ckpt: dict, model_id: str):
    ckpt['updated_at'] = datetime.now().isoformat()
    fp = ckpt_path(model_id)
    with open(fp, 'wb') as f:
        pickle.dump(ckpt, f)

def is_complete(ckpt: dict) -> bool:
    return ckpt.get('stop_reason') == 'complete'

def processed_ids(ckpt: dict) -> set:
    return {r['article_id'] for r in ckpt['results']} | set(ckpt['failed_ids'])

print("Checkpoint utilities ready")


# ============================================================
# CELL 6 : NER EXTRACTION FUNCTIONS
# ============================================================

def _parse_entities(raw: str) -> list:
    """Parse JSON entity list from model response. Returns [] on any failure."""
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$',          '', raw, flags=re.MULTILINE)
    try:
        data = json.loads(raw)
        entities = data.get('entities', [])
        return [e for e in entities
                if isinstance(e, dict) and 'text' in e and 'label' in e]
    except json.JSONDecodeError:
        return []


def extract_ner_groq(client, model_id: str, text: str):
    """
    Call Groq NER endpoint.
    Returns (entity_list, raw_response_str).
    Raises RuntimeError on unrecoverable failure -- caller handles stopping.
    """
    from groq import RateLimitError, APITimeoutError, APIStatusError

    messages = [
        {"role": "system", "content": NER_SYSTEM_PROMPT},
        {"role": "user",   "content": NER_USER_TEMPLATE.format(text=text[:3000])},
    ]

    last_exc = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS, start=1):
        if delay:
            print(f"    Waiting {delay}s before retry {attempt}...")
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=messages,
                max_tokens=512, temperature=0.0,
            )
            raw = resp.choices[0].message.content or ''
            return _parse_entities(raw), raw

        except RateLimitError as e:
            last_exc = e
            if attempt <= len(RETRY_DELAYS):
                print(f"    429 rate-limit (attempt {attempt}), backing off...")
            else:
                raise RuntimeError(f"RATE_LIMIT after {attempt} attempts: {e}") from e

        except APITimeoutError as e:
            last_exc = e
            if attempt <= len(RETRY_DELAYS):
                print(f"    Timeout (attempt {attempt}), retrying...")
            else:
                raise RuntimeError(f"TIMEOUT after {attempt} attempts: {e}") from e

        except APIStatusError as e:
            # 402 credits exhausted, 503 model unavailable -- stop immediately
            raise RuntimeError(f"API_STATUS_{e.status_code}: {e.message}") from e

        except Exception as e:
            raise RuntimeError(f"UNEXPECTED: {e}") from e

    raise RuntimeError(f"Exhausted retries. Last error: {last_exc}")


def extract_ner_anthropic(client, model_id: str, text: str):
    """
    Call Anthropic NER endpoint. Same return contract as extract_ner_groq.
    """
    import anthropic as _anthropic
    try:
        resp = client.messages.create(
            model=model_id, max_tokens=512,
            system=NER_SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": NER_USER_TEMPLATE.format(text=text[:3000])}],
        )
        raw = resp.content[0].text if resp.content else ''
        return _parse_entities(raw), raw

    except _anthropic.RateLimitError as e:
        raise RuntimeError(f"ANTHROPIC_RATE_LIMIT: {e}") from e
    except _anthropic.APIStatusError as e:
        raise RuntimeError(f"ANTHROPIC_STATUS_{e.status_code}: {e.message}") from e
    except Exception as e:
        raise RuntimeError(f"ANTHROPIC_UNEXPECTED: {e}") from e


print("NER extraction functions ready")


# ============================================================
# CELL 7 : RUN LOOP WITH CHECKPOINTING
# ============================================================

def run_model(model_id, extract_fn, article_ids, id_to_text,
              sleep_between=GROQ_SLEEP_BETWEEN):
    """
    Generic run loop. Resumes automatically from checkpoint.

    Stopping conditions (saves checkpoint before stopping):
      - rate_limit          : 429 or 402 from API
      - consecutive_failures: MAX_CONSECUTIVE_FAILURES in a row
      - high_failure_rate   : >MAX_FAILURE_RATE of total articles failed
      - complete            : all articles processed

    Re-running this cell after any stop will resume from where it left off.
    """
    ckpt = load_checkpoint(model_id)

    if is_complete(ckpt):
        print(f"  {model_id} already complete -- skipping")
        return ckpt

    already_done = processed_ids(ckpt)
    remaining    = [aid for aid in article_ids if aid not in already_done]

    if not remaining:
        ckpt['stop_reason'] = 'complete'
        save_checkpoint(ckpt, model_id)
        print(f"  {model_id} -- all articles processed, marked complete")
        return ckpt

    print(f"\n{'='*60}")
    print(f"  MODEL   : {model_id}")
    print(f"  Pending : {len(remaining)} articles  (done: {len(already_done)})")
    print(f"  Stop if : {MAX_CONSECUTIVE_FAILURES} consecutive failures "
          f"OR >{MAX_FAILURE_RATE*100:.0f}% failure rate")
    print(f"{'='*60}")

    consec_fail = 0
    total_ok    = len([r for r in ckpt['results'] if r['status'] == 'ok'])
    total_fail  = len(ckpt['failed_ids'])

    for i, article_id in enumerate(remaining, start=1):
        text = id_to_text.get(article_id, '')

        if not text:
            # No text available -- count as failure but don't stop
            ckpt['failed_ids'].append(article_id)
            consec_fail += 1
            total_fail  += 1
            print(f"  [{i}/{len(remaining)}] {article_id} -- NO TEXT")
        else:
            try:
                entities, raw = extract_fn(text)

                ckpt['results'].append({
                    'article_id'   : article_id,
                    'entities'     : entities,
                    'raw_response' : raw,
                    'status'       : 'ok',
                })
                consec_fail  = 0
                total_ok    += 1

                # Progress print every 20 articles
                if i % 20 == 0 or i == len(remaining):
                    print(f"  [{i}/{len(remaining)}] ok={total_ok}  "
                          f"fail={total_fail}  last={len(entities)} entities")

                time.sleep(sleep_between)

            except RuntimeError as exc:
                err_msg = str(exc)
                consec_fail += 1
                total_fail  += 1
                ckpt['failed_ids'].append(article_id)
                print(f"\n  [{i}/{len(remaining)}] FAILED [{article_id}]: {err_msg}")

                # Determine stop reason
                stop_reason = None

                if 'RATE_LIMIT' in err_msg or 'API_STATUS_402' in err_msg:
                    stop_reason = 'rate_limit'
                    print(f"\n  STOPPING -- rate limit / credit exhaustion.")
                    print(f"  Fix: wait for limit reset or switch provider, then re-run.")

                elif consec_fail >= MAX_CONSECUTIVE_FAILURES:
                    stop_reason = 'consecutive_failures'
                    print(f"\n  STOPPING -- {consec_fail} consecutive failures.")
                    print(f"  Fix: check model availability / API status, then re-run.")

                else:
                    total_seen = total_ok + total_fail
                    if total_seen >= 20 and (total_fail / total_seen) > MAX_FAILURE_RATE:
                        stop_reason = 'high_failure_rate'
                        rate = total_fail / total_seen * 100
                        print(f"\n  STOPPING -- failure rate {rate:.1f}% "
                              f"exceeds {MAX_FAILURE_RATE*100:.0f}% threshold.")

                if stop_reason:
                    ckpt['stop_reason'] = stop_reason
                    save_checkpoint(ckpt, model_id)
                    print(f"  Checkpoint saved: {ckpt_path(model_id).name}")
                    print(f"  Status: {total_ok} ok / {total_fail} failed / "
                          f"{len(remaining) - i} not yet attempted")
                    return ckpt

        # Periodic checkpoint save
        if i % SAVE_EVERY == 0:
            save_checkpoint(ckpt, model_id)
            print(f"  [checkpoint saved at article {i}]")

    ckpt['stop_reason'] = 'complete'
    save_checkpoint(ckpt, model_id)
    print(f"\n  COMPLETE: {model_id}")
    print(f"  ok={total_ok}  failed={total_fail}")
    return ckpt


# Run confirmed Groq models
nb13_new_results = {}

for model_id in confirmed:
    extract_fn = (lambda text, mid=model_id:
                  extract_ner_groq(groq_client, mid, text))
    nb13_new_results[model_id] = run_model(
        model_id=model_id, extract_fn=extract_fn,
        article_ids=ARTICLE_IDS_183, id_to_text=id_to_text,
        sleep_between=GROQ_SLEEP_BETWEEN,
    )


# Run Anthropic Haiku if token is available
HAIKU_ID = "claude-haiku-4-5-20251001"

if ANTHROPIC_TOKEN:
    import anthropic as _anthropic
    anthropic_client = _anthropic.Anthropic(api_key=ANTHROPIC_TOKEN)
    extract_haiku    = lambda text: extract_ner_anthropic(anthropic_client, HAIKU_ID, text)
    nb13_new_results[HAIKU_ID] = run_model(
        model_id=HAIKU_ID, extract_fn=extract_haiku,
        article_ids=ARTICLE_IDS_183, id_to_text=id_to_text,
        sleep_between=0.5,
    )
else:
    print(f"\n  Skipping {HAIKU_ID} -- set ANTHROPIC_TOKEN in Colab Secrets to enable")


# ============================================================
# CELL 8 : CONSOLIDATE ALL LLM RESULTS
# ============================================================

# Label normalisation: handles variation in how different LLMs name entity types
LABEL_NORM = {
    "PERSON": "PER", "person": "PER",
    "LOCATION": "LOC", "location": "LOC", "GPE": "LOC", "gpe": "LOC", "FAC": "LOC",
    "ORGANIZATION": "ORG", "organisation": "ORG", "organization": "ORG",
    "MISCELLANEOUS": "MISC", "miscellaneous": "MISC",
    "EVENT": "MISC", "PRODUCT": "MISC", "LANGUAGE": "MISC", "NORP": "MISC",
    "WORK_OF_ART": "MISC", "LAW": "MISC", "DATE": "MISC", "TIME": "MISC",
    # already-standard labels pass through
    "PER": "PER", "LOC": "LOC", "ORG": "ORG", "MISC": "MISC",
}
VALID_LABELS = {"PER", "LOC", "ORG", "MISC"}


def normalize_entities(entity_list: list) -> set:
    """Convert entity list to set of (normalised_text, normalised_label) tuples."""
    out = set()
    for e in (entity_list or []):
        text  = str(e.get('text', '')).strip().lower()
        label = LABEL_NORM.get(str(e.get('label', '')), None)
        if text and label in VALID_LABELS:
            out.add((text, label))
    return out


def build_entity_lookup(ckpt: dict) -> dict:
    """Returns {article_id: set_of_(text,label)_tuples}."""
    return {
        r['article_id']: normalize_entities(r.get('entities', []))
        for r in ckpt.get('results', [])
        if r.get('status') == 'ok'
    }


# Merge existing + new LLM results
all_llm_lookups = {}
all_llm_meta    = {}

for model_id, info in EXISTING_LLM_FILES.items():
    if model_id in existing_llm_results:
        all_llm_lookups[model_id] = build_entity_lookup(existing_llm_results[model_id])
        all_llm_meta[model_id]    = {k: v for k, v in info.items() if k != 'file'}
        print(f"  {model_id}: {len(all_llm_lookups[model_id])} articles")

for model_id, ckpt in nb13_new_results.items():
    n_ok = len([r for r in ckpt['results'] if r['status'] == 'ok'])
    if n_ok > 0:
        all_llm_lookups[model_id] = build_entity_lookup(ckpt)
        meta_src = {**confirmed, **ANTHROPIC_TARGET_MODELS}.get(model_id, {})
        all_llm_meta[model_id]    = meta_src
        print(f"  {model_id}: {n_ok} articles")
    else:
        print(f"  {model_id}: 0 ok articles -- excluded from Jaccard")

print(f"\nTotal LLM systems for Jaccard: {len(all_llm_lookups)}")

# Pipeline entity lookups (raw per-system output)
pipeline_lookups = {}
article_id_set   = set(ARTICLE_IDS_183)

for name, col in PIPELINE_ENTITY_COLS.items():
    if col not in pipeline_df.columns:
        print(f"  Skipping pipeline {name}: column '{col}' missing")
        continue
    lookup = {}
    for _, row in pipeline_df.iterrows():
        aid = row['article_id']
        if aid not in article_id_set:
            continue
        raw_ents = row[col]
        if isinstance(raw_ents, list) and raw_ents:
            first = raw_ents[0]
            if isinstance(first, dict):
                lookup[aid] = normalize_entities(raw_ents)
            elif isinstance(first, (list, tuple)) and len(first) >= 2:
                lookup[aid] = normalize_entities(
                    [{'text': e[0], 'label': e[1]} for e in raw_ents])
            else:
                lookup[aid] = set()
        else:
            lookup[aid] = set()
    pipeline_lookups[name] = lookup
    print(f"  Pipeline {name}: {len(lookup)} articles")

print("\nAll lookups built")


# ============================================================
# CELL 9 : COMPUTE JACCARD SIMILARITY
# ============================================================

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0   # both empty -> perfect agreement by convention
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_jaccard_stats(p_lookup: dict, l_lookup: dict,
                           article_ids: list) -> dict:
    scores = [
        jaccard(p_lookup.get(aid, set()), l_lookup.get(aid, set()))
        for aid in article_ids
        if aid in p_lookup and aid in l_lookup
    ]
    return {
        'mean'  : float(np.mean(scores))   if scores else np.nan,
        'median': float(np.median(scores)) if scores else np.nan,
        'std'   : float(np.std(scores))    if scores else np.nan,
        'n'     : len(scores),
    }


# LLMs ordered by scale for the x-axis
llm_order = sorted(all_llm_meta, key=lambda m: all_llm_meta[m].get('scale_B', 0))

jaccard_records = []
for pipeline_name, p_lookup in pipeline_lookups.items():
    for llm_id in llm_order:
        stats = compute_jaccard_stats(
            p_lookup, all_llm_lookups[llm_id], ARTICLE_IDS_183)
        jaccard_records.append({
            'pipeline': pipeline_name,
            'llm'     : llm_id,
            'scale_B' : all_llm_meta[llm_id].get('scale_B', 0),
            'family'  : all_llm_meta[llm_id].get('family', '?'),
            **stats,
        })

jaccard_df = pd.DataFrame(jaccard_records)

print("Jaccard similarity matrix (mean):")
pivot = jaccard_df.pivot(index='pipeline', columns='llm', values='mean').round(3)
# Reorder columns by scale
col_order = [m for m in llm_order if m in pivot.columns]
print(pivot[col_order].to_string())

jaccard_df.to_csv(DATA_PROC / 'nb13_jaccard_matrix.csv', index=False)
print(f"\nSaved: nb13_jaccard_matrix.csv")


# ============================================================
# CELL 10 : SCALING CURVE PLOT
# ============================================================

PIPELINE_STYLES = {
    'Flair' : {'color': '#1f77b4', 'marker': 'o', 'lw': 2.2, 'ms': 9},
    'Stanza': {'color': '#ff7f0e', 'marker': 's', 'lw': 2.2, 'ms': 9},
    'spaCy' : {'color': '#2ca02c', 'marker': '^', 'lw': 2.2, 'ms': 9},
}

def x_label(model_id: str, scale_B: float) -> str:
    if 'qwen'   in model_id.lower(): return f"Qwen\n{int(scale_B)}B"
    if 'haiku'  in model_id.lower(): return f"Haiku\n~{int(scale_B)}B"
    if 'gemma'  in model_id.lower(): return f"Gemma2\n{int(scale_B)}B"
    if '3b'     in model_id.lower(): return f"Llama\n3B"
    if '8b'     in model_id.lower(): return f"Llama\n8B"
    if '70b'    in model_id.lower(): return f"Llama\n70B"
    return f"{int(scale_B)}B"

fig, ax = plt.subplots(figsize=(10, 6))

for pipeline_name, style in PIPELINE_STYLES.items():
    sub = jaccard_df[jaccard_df['pipeline'] == pipeline_name].sort_values('scale_B')
    if sub.empty:
        continue
    ax.plot(sub['scale_B'], sub['mean'],
            label=pipeline_name,
            color=style['color'], marker=style['marker'],
            linewidth=style['lw'], markersize=style['ms'])
    ax.fill_between(sub['scale_B'],
                    sub['mean'] - sub['std'],
                    sub['mean'] + sub['std'],
                    alpha=0.12, color=style['color'])

# x-axis tick labels
scale_rows = (jaccard_df.drop_duplicates('scale_B')
                         .sort_values('scale_B'))
ax.set_xticks(scale_rows['scale_B'].tolist())
ax.set_xticklabels([x_label(r['llm'], r['scale_B'])
                    for _, r in scale_rows.iterrows()], fontsize=9)

ax.set_xlabel('LLM Scale (approximate parameters)', fontsize=11)
ax.set_ylabel('Mean Jaccard Similarity\n(pipeline entities vs. LLM entities)', fontsize=11)
ax.set_title(
    'Pipeline–LLM Entity Agreement Across Model Scales\n'
    '(no fixed gold standard; 183 articles; EN input; shading = ±1 SD)',
    fontsize=12)
ax.legend(title='Pipeline', fontsize=10)
ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.grid(axis='y', alpha=0.3)
ax.grid(axis='x', alpha=0.15)

plt.tight_layout()
for ext in ['pdf', 'png']:
    out = FIGURES_DIR / f'nb13_scaling_curve.{ext}'
    fig.savefig(out, dpi=300 if ext == 'pdf' else 150, bbox_inches='tight')
plt.show()
print("Figure saved: nb13_scaling_curve.pdf / .png")


# ============================================================
# CELL 11 : SUMMARY JSON
# ============================================================

flair_curve = jaccard_df[jaccard_df['pipeline'] == 'Flair'].sort_values('scale_B')
if not flair_curve.empty:
    peak = flair_curve.loc[flair_curve['mean'].idxmax()]
    convergence = {'scale_B': float(peak['scale_B']),
                   'jaccard': round(float(peak['mean']), 4)}
else:
    convergence = None

model_run_status = {
    mid: {
        'n_ok'       : len([r for r in ckpt['results'] if r['status'] == 'ok']),
        'n_failed'   : len(ckpt['failed_ids']),
        'stop_reason': ckpt.get('stop_reason'),
    }
    for mid, ckpt in nb13_new_results.items()
}

summary = {
    'notebook'       : '13_scaling_study',
    'generated_at'   : datetime.now().isoformat(),
    'n_articles'     : len(ARTICLE_IDS_183),
    'llm_systems'    : {
        mid: {
            'scale_B': all_llm_meta[mid].get('scale_B'),
            'family' : all_llm_meta[mid].get('family'),
            'n_used' : int(jaccard_df[jaccard_df['llm'] == mid]['n'].max())
                       if mid in jaccard_df['llm'].values else 0,
        }
        for mid in llm_order
    },
    'jaccard_matrix' : {
        f"{r['pipeline']}|{r['llm']}": round(r['mean'], 4)
        for _, r in jaccard_df.iterrows()
    },
    'convergence'    : convergence,
    'run_status'     : model_run_status,
}

summary_path = DATA_PROC / 'nb13_scaling_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print("=" * 60)
print("NOTEBOOK 13 COMPLETE")
print("=" * 60)
print(f"  nb13_jaccard_matrix.csv     -- Jaccard scores (all pipeline x LLM pairs)")
print(f"  nb13_scaling_summary.json   -- Summary + convergence estimate")
print(f"  nb13_scaling_curve.pdf/.png -- Main figure")
print()
if convergence:
    print(f"  Flair convergence peak: ~{convergence['scale_B']}B params "
          f"(Jaccard = {convergence['jaccard']})")
print()
print("  New model run status:")
for mid, s in model_run_status.items():
    print(f"    {mid:<42} ok={s['n_ok']}  "
          f"fail={s['n_failed']}  [{s['stop_reason'] or 'not run'}]")
