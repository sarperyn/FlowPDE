"""Figure generation for the report.

Each module reads result CSVs (or cached ensembles) and writes vector PDFs into
``report/figures``.  Nothing here trains or samples except where a figure needs
raw fields, in which case it regenerates the dataset — which is seconds, not
hours, because the generators are seeded.
"""
