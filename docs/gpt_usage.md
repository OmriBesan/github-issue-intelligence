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

## Entry 002 — Stage 1A: GitHub Issue Collection Pipeline

**Date:** 2026-07-22
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** 1A — Data Collection

### What the AI assisted with

1. **Designing the GitHub API client module** (`github_client.py`)
   - Suggested using `requests.Session` for connection reuse.
   - Designed the PR filtering logic (checking for the `"pull_request"` key).
   - Designed the Link-header pagination parser.
   - Structured the metadata dictionary to include only safe, non-sensitive fields.

2. **Writing the CLI script** (`collect_issues.py`)
   - Generated the `argparse` interface with `--help` text.
   - Wrote the statistics printer (label frequencies, date range, missing bodies).

3. **Writing the unit test suite** (`test_github_client.py`)
   - Designed the `FakeResponse` class to avoid any real network calls.
   - Wrote all 7 test cases covering PR exclusion, pagination, token safety,
     HTTP errors, and file output verification.

4. **Debugging and fixing a real bug**
   - The AI discovered and fixed a `KeyError: 'page'` bug that appeared when
     following GitHub Link-header pagination (params dict was cleared but the
     debug log still referenced it). Fixed by introducing a separate `page_number`
     counter.

5. **Fixing a Windows encoding issue**
   - The `✅` emoji in print statements caused a `UnicodeEncodeError` on Windows
     cp1252 terminals. The AI replaced them with plain ASCII `[OK]` markers.

### What we did ourselves

- Approved the design of the collector before any code was written.
- Reviewed every generated file.
- Ran the actual collection and inspected the real output.

### How we verified the AI output

- All 16 pytest tests pass.
- `ruff check` reports no issues.
- The real collection produced 500 valid issues with 0 PR entries and no token
  data in any output file, confirmed by a programmatic JSON inspection script.

---

*Future entries will be added at the end of each stage.*
