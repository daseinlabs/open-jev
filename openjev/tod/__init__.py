"""Train task-oriented decision (TOD) models: pick one candidate, report confidence, defer.

You supply a DecisionSpec (decision kinds and how their candidates are enumerated; `ListSpec`
when the candidates are a list in the state, e.g. retrieved passages) and an
adapter that turns your data into Episodes of spec-built Decisions; `chat_traces` derives
both from OpenAI-style tool-calling logs. The SDK owns validation, grouped splits,
coverage reporting, training (`openjev tod train`), calibration, confident-or-defer
serving, and RL in an Environment (`openjev.tod.rl`). Importing this package never loads
a model or an accelerator.
"""
from .schema import (SCHEMA_VERSION, Candidate, Choice, Decision, Episode, NoneCorrect, Outcome,
                     Prediction, Preference, Subset, Target)
from .spec import Dataset, DecisionSpec, Kind, ListSpec, build_dataset, load_decisions, load_spec

__all__ = ['SCHEMA_VERSION', 'Candidate', 'Choice', 'Decision', 'Episode', 'NoneCorrect', 'Outcome',
           'Prediction', 'Preference', 'Subset', 'Target', 'Dataset', 'DecisionSpec', 'Kind', 'ListSpec',
           'build_dataset', 'load_decisions', 'load_spec']
