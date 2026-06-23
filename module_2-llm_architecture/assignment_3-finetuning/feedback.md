Feedback - Score: 99

Summary
This is an excellent submission that demonstrates strong technical understanding of both QLoRA fine-tuning and attention mechanisms. The student produced a coherent end-to-end pipeline with thorough evidence (datasets, training configurations, evaluation tables, attention heatmaps) and well-reasoned analysis grounded in observed results.

What Works Well
Task 1.1: Robust dataset generation pipeline using Nebius/Llama-3.3-70B with a clean prompt template producing valid child/student/expert JSON. Both causal and seq2seq formats are pre-built into the data rows. Train (1719) / Test (21) split, JSONL saved + reloaded, displayed as DataFrames.
Task 1.2: Proper QLoRA configuration — 4-bit NF4 with double quantization, fp16 compute dtype, prepare_model_for_kbit_training, LoRA rank 16 / alpha 32, correct task types and target modules for both SmolLM2 (q_proj, v_proj) and flan-t5 (q, v). Different data collators and trainers (SFTTrainer vs Seq2SeqTrainer) correctly chosen.
Task 1.3: Evaluation table contains all required columns (Question, Level, Base Output, Fine-tuned Output, Expected Answer, Better after FT?, Notes). Clever use of disable_adapter() to compare base vs fine-tuned in a single load. 21 test examples cover all three expertise levels.
Task 1.4: Strong, evidence-based discussion citing concrete examples (Duchovny hallucinations, "simile vs syma" degeneration). Correctly attributes flan-t5 failure to learning rate, model size, and pre-training objective mismatch.
Task 1.5: All four questions answered with technical depth — explains 4-bit quantization, LoRA capacity tradeoff, overfitting risk, and prompt-vs-fine-tune economics with concrete numbers from the experiment.
Task 2.1: Mathematically correct scaled dot-product attention with mask support, scaling, softmax, and proper tensor shapes.
Task 2.2: Clear, well-labeled heatmap showing identity-dominant attention pattern (as expected with X=Q=K=V).
Task 2.3: Both experiments well-implemented; uses shared embedding lookup to isolate the effect of the changed word; multi-head experiment uses distinct W_Q/W_K/W_V projections per head. Answers are grounded in actual observed numbers.
What Needs Improvement
Task 1.3: The "Better after FT?" and "Notes" columns appear to be populated from an LLM judge based on the displayed output, but only head(3) is shown for each model. Showing the full 21-row table (or at least summary statistics across all rows) would strengthen the evidence. Minor deduction here.
The text field of Cell 26 markdown begins with a fragment ("e confusing 'metaphor'…") suggesting truncation at the start of question 1's answer — likely a copy/paste artifact. Content is otherwise complete and excellent.
Suggestions
For Task 1.3, render the full evaluation DataFrame (or include aggregate metrics like % "Better after FT?" per level/model) to make the comparative claims in Task 1.4 fully reproducible from the notebook output.
For Task 1.2, consider documenting why a lower learning rate (e.g., 5e-5) might have been better for flan-t5 — your Task 1.4 discussion already correctly diagnoses this, so a brief note in the training config would close the loop.
Fix the small leading-fragment typo at the start of Cell 26.
Grading Summary
Task 1.1 — Custom Dataset: 15 / 15
Task 1.2 — QLoRA Fine-Tuning: 10 / 10
Task 1.3 — Evaluation: 19 / 20 (minor deduction for only displaying head(3) of evaluation tables)
Task 1.4 — Model Comparison: 15 / 15
Task 1.5 — Conceptual Questions: 15 / 15
Task 2.1 — Scaled Dot-Product Attention: 5 / 5
Task 2.2 — Visualize Attention: 5 / 5
Task 2.3 — Experiments: 15 / 15
Final Score: 99 / 100
