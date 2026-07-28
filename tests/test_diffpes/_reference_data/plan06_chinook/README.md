# Plan 06 Chinook K-reference

This directory contains the inert numerical output used by Plan 06 gates G6
and G7. The source comparator is Chinook commit
`24913de8cc5b8c162f7c1b4acc64bd1b54dd548b`. The adjacent manifest pins the
archive, model specification, isolated environment, and behavioral scope.

`test_plan06_chinook_parity.py` reconstructs the DiffPES result through public
APIs on every frozen point. It never imports Chinook or trusts a saved DiffPES
replay. The independently generated archive remains immutable; changes to
`matrixel.py` must continue to reproduce it within the registered tolerance.
