# Agent Instructions & Workspace Rules

## 1. Language Policy

* **Direct Chat Communication with the User:** You may converse with the user in German (or match the language chosen by the user).
* **Project Documentation & Artifacts (Mandatory English):** All project documentation, README files, markdown reports, summaries, notes, and repository artifacts must be authored **exclusively in English**.
* **Code & Technical Nomenclature (Mandatory English):** All code, identifiers, function and variable names, classes, docstrings, comments, and commit messages must be strictly in English.

## 2. Methodology & Scientific Rigor

* **Time Handling:** Strict adherence to Pure UTC (`+00:00` / `Z`) across all time series data.
* **Look-Ahead Bias Prevention:** Never incorporate future data (e.g., ex-post quality-assured clearing revisions, intraday forecast updates published after gate closure) into prediction features for earlier contract maturities.
