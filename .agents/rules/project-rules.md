# Project Development Rules

These rules apply to all AI coding assistants working on this project.
They exist to keep the project understandable, reproducible, and safe
for both student partners throughout the course.

---

## Rules

1. Work in small, reviewable stages.
2. Do not implement future stages unless explicitly asked.
3. Before changing files, briefly explain your plan.
4. After each stage, explain:
   - what was created or changed,
   - why it was needed,
   - how to run it,
   - and what should happen next.
5. Use clear, student-readable Python.
6. Avoid unnecessary abstractions and overengineering.
7. Add type hints and concise comments where useful.
8. Do not hide important logic inside notebooks.
9. Notebooks are for exploration and presentation.
10. Reusable logic belongs in Python modules under `src/`.
11. Never place tokens, passwords, or secrets directly in code.
12. Use a `.env` file locally and provide only a safe `.env.example`.
13. Do not overwrite existing files without checking them.
14. Do not run destructive terminal commands.
15. Do not push to GitHub or create a remote repository without explicit permission.
16. Run tests after meaningful changes.
17. Keep documentation updated so either project partner can continue later.
18. Record meaningful AI assistance in `docs/gpt_usage.md`.
19. Do not claim that code works unless it was actually executed and verified.
20. Stop after completing the requested stage.

---

## Project Structure Reference

```
github-issue-intelligence/
├── .agents/rules/          ← This file lives here
├── data/
│   ├── raw/                ← Downloaded data (not committed)
│   ├── interim/            ← Partially processed data
│   └── processed/          ← Final model-ready datasets
├── docs/                   ← All project documentation
├── notebooks/              ← Jupyter notebooks
├── reports/figures/        ← Generated charts
├── scripts/                ← One-off helper scripts
├── src/issue_intelligence/ ← Importable Python package
└── tests/                  ← pytest test suite
```

---

## Sensitive Files

| File | Rule |
|------|------|
| `.env` | **Never commit.** Contains real secrets. Listed in `.gitignore`. |
| `.env.example` | **Always commit.** Safe template with no real values. |
| `data/raw/` | **Never commit.** May contain large files or API responses. |
