"""Train task-oriented decision (TOD) models: pick one candidate, report confidence, defer.

You supply a DecisionSpec (decision kinds and how their candidates are enumerated) and an
adapter that turns your data into Episodes of spec-built Decisions; `chat_traces` derives
both from OpenAI-style tool-calling logs. The SDK owns validation, grouped splits,
coverage reporting, training (`openjev tod train`), calibration, confident-or-defer
serving, and RL in an Environment (`openjev.tod.rl`). Importing this package never loads
a model or an accelerator.
"""
from .schema import (SCHEMA_VERSION, Candidate, Choice, Decision, Episode, NoneCorrect, Outcome,
                     Prediction, Preference, Target)
from .spec import Dataset, DecisionSpec, Kind, build_dataset, load_decisions

__all__ = ['SCHEMA_VERSION', 'Candidate', 'Choice', 'Decision', 'Episode', 'NoneCorrect', 'Outcome',
           'Prediction', 'Preference', 'Target', 'Dataset', 'DecisionSpec', 'Kind',
           'build_dataset', 'load_decisions']
