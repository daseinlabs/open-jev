"""Generic primitives for training task-oriented decision (TOD) models.

The user supplies a DecisionSpec (decision kinds and their candidate sources) and an
extractor that turns their traces into Episodes of spec-built Decisions. The SDK owns
validation, grouped splits, label-coverage reporting, and confident-or-defer serving.
"""
from .schema import (SCHEMA_VERSION, Candidate, Choice, Decision, Episode, NoneCorrect, Outcome,
                     Prediction, Preference, Target)
from .spec import Dataset, DecisionSpec, Kind, build_dataset, load_decisions

__all__ = ['SCHEMA_VERSION', 'Candidate', 'Choice', 'Decision', 'Episode', 'NoneCorrect', 'Outcome',
           'Prediction', 'Preference', 'Target', 'Dataset', 'DecisionSpec', 'Kind',
           'build_dataset', 'load_decisions']
