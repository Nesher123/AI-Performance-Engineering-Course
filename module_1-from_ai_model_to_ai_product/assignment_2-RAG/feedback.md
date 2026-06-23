Feedback - Score: 100

Per-Task Feedback
Task 1 — 10 / 10
All deliverables complete: code is clear and correct using Llama-3.3-70B-Instruct, the xlsx table has all 6 required columns with 10 well-judged entries (5 domain-relevant, 5 novel-generated), and the discussion addresses all three questions with concrete observations about refusal patterns and question_type differences.

Task 2 — 5 / 5
All three RAG components (indexing, retrieval, generation) are addressed separately with correct contribution/failure/timing explanations. Concrete failure examples are strong (e.g., "splitting mid-sentence," vocabulary mismatch "cancel" vs "terminate subscription," "lost in the middle").

Task 3 — 15 / 15
Code correctly loads only relevant PDFs with PyPDFLoader, attaches all required metadata (doc_name, company, doc_period, page_number 0-indexed), splits with RecursiveCharacterTextSplitter(1000/150), embeds with BAAI/bge-small-en-v1.5 into FAISS, and saves to disk. Retrieval sanity check on 3 questions shows actual output with metadata and includes a thoughtful markdown observation about the retriever finding the right document but struggling with table-heavy evidence pages.

Task 4 — 25 / 25
System prompt covers all three required elements (answer only from context, explicit refusal when context insufficient, concise with citations). Prompt construction uses clear separators with doc_name metadata and explicitly handles empty retrieval. Code is clean and correct with the right models. The answer_with_rag function has the correct signature and return format with answer (str) and retrieved_chunks (list with doc_name/page_number metadata).

Task 5 — 10 / 10
Code straightforwardly runs the same 10 Task 1 questions through RAG. The xlsx has all required columns. The discussion is excellent — it provides a detailed per-question breakdown with tables showing where RAG helped (e.g., Best Buy cash drop with specific citation) and hurt (e.g., Corning working capital refusal), identifies the domain-relevant vs. novel-generated asymmetry, and offers a well-reasoned hypothesis for why.

Task 6 — 20 / 20
All three metrics implemented correctly: correctness judge uses DeepSeek-V3.2 with a visible binary-verdict prompt, faithfulness uses Ragas with llm_factory wrapping on the first 20 questions, and page-hit@k correctly compares page_number to evidence_page_num for k in {1,3,5} with proper handling of multiple evidence items. The xlsx has all required columns with 100 rows. Aggregate numbers are clearly reported (correctness 0.290, faithfulness 0.423, page-hit@1/3/5 = 0.20/0.33/0.40). The results summary is exceptionally thorough with cross-tabulation of correctness vs. page-hit, refusal analysis, and faithfulness breakdown.

Task 7 — 15 / 15
Three valid experiments (reranker, relaxed prompt, chunk size 500), each changing one variable. Each has a clear hypothesis naming the expected metric and reasoning, and an interpretation grounded in actual numbers. All three metrics are re-computed per experiment. The FAISS index was rebuilt for the chunk size experiment. The results xlsx has a baseline row and all required columns. The wrap-up is grounded in results — identifies retrieval as the main bottleneck (page-hit@5 = 40%), names relaxed_prompt as the best experiment, and proposes concrete next steps (query rewriting, metadata-aware filtering).

Bonus: Not attempted (0/10).

Overall: This is an outstanding submission. The notebook is exceptionally well-organized with clear task separation, markdown headers, and step-by-step sub-sections. Code is clean, correct, and well-documented. Discussions go beyond surface-level observations with insightful analysis grounded in specific results. The Task 7 experiment harness with dataclass-based configuration and cached execution is particularly well-engineered.
