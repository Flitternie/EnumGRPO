# SemBench Movie oracle programs

This directory contains the fixed, query-specific programs used by the three
oracle baselines in our SemBench Movie experiment. Here, *oracle* means that
the workflow is manually specified for each query; it does not mean that the
program can access ground-truth labels or answers.

- `lotus.py` and `palimpzest.py` are verbatim copies of the corresponding
  Movie runners in SemBench commit
  `c814e3807e72d4cf876b852b17e77f3cc94575c2`:
  `src/scenario/movie/runner/{lotus_runner,palimpzest_runner}/`.
- `blendsql/Q1.sql` through `Q10.sql` are the fixed BlendSQL programs used in
  our evaluation. The runner records BlendSQL version `0.1.26` and the
  implementation revision `6e1b3606e37ac378131e908b40ac290c1440a7d7`.

`../run_oracle_programs.py` loads the vendored LOTUS and Palimpzest query
programs while retaining SemBench's generic runner support. `../run_blendsql.py`
executes the SQL programs in `blendsql/`.
