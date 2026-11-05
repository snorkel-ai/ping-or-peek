"""The trials parsed, the episodes exported, and the Hugging Face dataset snorkelai/ping-or-peek
built from them.

The readers of a Harbor trial (`trial`, `acp`, `codex_rollout`, `user_agent`) and the episode's
measures and evidence (`episode`, `evidence`) feed the labelers in `labeling`; `export` writes the
labeled episodes; `build` writes the dataset's Parquet shards from the runs and the export, `card`
writes its card, and `check` checks the result before anything is uploaded. `export_check` is
Check A of the safety net. Nothing here uploads.
"""
