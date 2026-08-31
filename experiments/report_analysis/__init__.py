"""Re-evaluation passes over trained checkpoints, for the report's figures.

Nothing in this package trains anything.  Each module loads existing
checkpoints, measures something the original runs did not record, and writes a
CSV under ``results/analysis/``.
"""
