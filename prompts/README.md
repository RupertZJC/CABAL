# Published prompt scaffold

`system_prompts.json` contains the honest, aggressive, subtle, and independent
quality-grader roles. `task_prompts.json` contains example bidding, single-review,
batch-review, and grading instructions using placeholder paper fields. The
shared scaffold and behavioral policies are described in Appendix B.3 of the
CABAL paper; full JSON-format instructions are included here for reuse.

Braced fields denote runtime inputs rather than real profiles. Task examples
show one placeholder paper; actual bidding used chunks of ten and reviewing
used per-reviewer batches. No real research backgrounds are instantiated here.

The scaffold builder is available in `collubid/llm/role_descriptions.py`, with
conference context and rubric in `collubid/core/persona.py`. Numerical settings
are recorded in `config/experiment.json`. This release supplies prompts and
configuration, rather than the full API-based generation pipeline.
