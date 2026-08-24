# Corpus Statistics

Figures below describe the cleaned working corpus. They are the numbers reported
in the thesis and are verified by `Project/Notebook/12_integrity_check.ipynb`.

## Overview

| Metric | Value |
|---|---|
| Articles (cleaned) | 1,115 |
| Articles collected before cleaning | 1,149 |
| Date range | 2015-01-12 to 2025-06-27 |
| Sources | 4 |
| Mean word count | 779 |
| Median word count | 705 |

Cleaning removed non-German articles (a small number of French-language pieces
captured by the archive queries) and articles under 50 words, which are almost all
captions or truncated retrievals rather than genuine short articles.

## Sources

| Source | Articles | Share |
|---|---|---|
| Luxemburger Wort | 356 | 31.9% |
| Frankfurter Allgemeine Zeitung | 290 | 26.0% |
| SZ Sueddeutsche Zeitung | 270 | 24.2% |
| Trierischer Volksfreund | 199 | 17.8% |
| **Total** | **1,115** | 100% |

## Article sets used in the experiments

Three counts recur in the thesis and are easy to confuse, so they are set out here.

| n | Set | Used for |
|---|---|---|
| 199 | Stratified sample as drawn | - |
| 183 | Articles with predictions from every system | Topic classification, translation ablation |
| 159 | Intersection of the four LLM German-input runs | NER dual-reference comparison |

The reduction from 199 to 183 was caused by API credit exhaustion during an early
run, not by any property of the articles. The further reduction to 159 is the
intersection of the four German-input LLM runs; Llama 70B is the binding
constraint, having failed on some articles under rate-limit pressure. Restricting
to the intersection keeps every system scored over an identical article set.

## Verification

`12_integrity_check.ipynb` recomputes the reported figures from the stored result
files. The most recent run passed all checks.
