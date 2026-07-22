# AI Tool Usage Log — GitHub Issue Intelligence

This file records how AI tools (GPT-4, Gemini, Antigravity coding assistant,
etc.) were used during the project, as required by the course lecturer.

The intent is transparency: to show which parts of the project were assisted
by AI and how the output was reviewed, adapted, and integrated.

---

## Entry 001 — Project Initialisation (Stage 0)

**Date:** 2026-07-22
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** 0 — Project Initialisation

### What the AI assisted with

1. **Brainstorming and refining the project idea**
   - We described the general concept (analysing GitHub issues with ML) and
     worked with the AI to sharpen the problem statement, choose candidate
     repositories, and define a useful output (issue category, component,
     similar issues, duplicate detection).

2. **Planning the project architecture**
   - The AI suggested a staged workflow (data → baselines → classical NLP →
     transformers → retrieval → API) and explained the rationale for each stage.
   - We reviewed, challenged, and adapted the plan to match the course
     requirements and our own understanding.

3. **Defining the gradual workflow rules**
   - We iterated with the AI on a set of project development rules (no
     overengineering, small reviewable stages, no secrets in code, etc.)
     to ensure the project remains understandable to both team members.

4. **Initialising the repository structure**
   - The AI generated the initial file tree, all boilerplate files
     (`.gitignore`, `.env.example`, `requirements.txt`, `README.md`, docs),
     and the minimal test suite.
   - We reviewed every file before approving it.
   - The AI executed the virtual environment setup and dependency installation
     and reported the results.

### What we did ourselves

- Defined the overall project goal and course constraints.
- Chose the three candidate repositories (scikit-learn, pandas, vLLM).
- Reviewed and approved every generated file.
- Made the decision to audit the data before committing to a label scheme.

### How we verified the AI output

- Read every generated file before approving.
- Ran `pytest` and `ruff` to confirm the scaffolding works.
- Reviewed the plan in `docs/project_plan.md` against the course requirements.

---

*Future entries will be added at the end of each stage.*
