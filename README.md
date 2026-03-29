# Assignment 1 - LLM Evaluation

Product description generation and evaluation using LLMs.

## Setup

1. **Install dependencies with uv:**

   ```bash
   uv sync
   ```

2. **Configure API credentials:**

   ```bash
   cp .env.example .env
   # Edit .env and add your Nebius Token Factory API key
   ```

3. **Launch Jupyter:**

   ```bash
   jupyter notebook assignment_01_solution.ipynb
   ```

## Project Structure

```
assignment_01-evaluation/
├── Assignment_01_product_dataset.csv   # Input data (51 products)
├── assignment_01_solution.ipynb        # Main notebook (all 6 tasks)
├── assignment_01.xlsx                  # Generated results (Task 2+)
├── pyproject.toml                      # Dependencies
├── .python-version                     # Python 3.11
├── .env.example                        # API credentials template
└── README.md                           # This file
```

## Assignment Tasks

1. **Task 1:** Define evaluation rubric
2. **Task 2:** Generate product descriptions
3. **Task 3:** Manual human evaluation (10-15 products)
4. **Task 4:** Improvement experiments
5. **Task 5:** Create LLM judge model
6. **Task 6:** Judge analysis and comparison

## Dependencies

- `pandas` - Data manipulation and Excel I/O
- `openpyxl` - Excel file format support
- `pydantic` - Structured output schemas
- `openai` - OpenAI REST API
- `python-dotenv` - Environment variable management
- `jupyter` - Notebook environment

## Notes

- Use Nebius Token Factory API (not OpenAI/Anthropic)
- Choose between Gemma-2-9b-it or Meta-Llama-3.1-8B-Instruct
- Judge model should be different from generator model
- All work is contained in a single notebook

**Due Date:** April 5, 2026
