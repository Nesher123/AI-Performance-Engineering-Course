Feedback - Score: 99

Summary
This is an excellent, nearly complete submission that demonstrates strong understanding of all required concepts — from subword tokenization through Word2Vec embeddings, MLP classification, and character-level RNN generation. Code is clean, well-documented, and nearly all outputs are present and correct.

What Works Well

Task 1 (Tokenization): Both BPE and WordPiece tokenizers are correctly trained on training data only with vocab_size=10,000. All 5 example sentences are shown in a clean side-by-side table with clear token outputs. The conceptual explanation comparing BPE, WordPiece, and word-level tokenization is thorough and accurate.

Task 2 (Word2Vec): All 6 model variants (CBOW + Skip-gram × 3 embedding sizes) are trained on BPE-tokenized training data. Similarity examples are shown for 3 representative tokens across all models. The written analysis clearly explains why Skip-gram outperforms CBOW on this dataset, which embedding size was chosen and why, and why the learned Word2Vec vocabulary (9,794) differs from the tokenizer vocabulary (10,000).

Task 3 (MLP): Mean pooling is used to convert variable-length token sequences to fixed-size vectors. The MLP architecture is well-designed (2 hidden layers, ReLU, Dropout), trained on the training set for 20 epochs with steadily decreasing loss. All design choices are tabulated and justified. The explanation of MLP suitability and its limitations vs. RNNs is clear and accurate.

Task 4 (Evaluation): Test accuracy of 89.33% is computed on the test set. A proper confusion matrix is shown with correct labels. Three misclassified examples include original text, true label, and predicted label. Error analysis correctly identifies Business↔Sci/Tech confusion as the dominant pattern and provides a well-motivated leave-one-out ablation method for interpretability.

Part 2 Task 1 (DinosDataset): Correct implementation with start/end tokens, vocabulary + mappings, sliding-window sequences, shifted targets, __len__/__getitem__, train/val split (80/20 with random_split), and DataLoaders with fixed batch size. Output confirms vocab size of 54 and correct x/y shift.

Part 2 Task 2 (One-hot encoding): Correct numpy implementation using shape broadcasting; works for batches of shape (batch, seq_len), producing output of shape (batch, seq_len, vocab_size) as required.

Part 2 Task 3 (Model forward pass): Every layer's output shape is printed step-by-step (input → one-hot → LSTM → dropout → reshape → FC → softmax). Probabilities sum to 1.0. Clear and thorough.

Part 2 Task 4 (Training): Training function uses one_hot_encode, Adam optimizer, CrossEntropyLoss, gradient clipping, and reports both training and validation loss every N steps. Model is trained for 30 epochs with loss decreasing from ~2.9 to ~0.68 (train) and ~0.72 (val). A loss curve plot is also included.

Part 2 Task 5 (Name generation): predict and sample functions are correctly implemented with start/end token logic and max-length stopping. Generated names are plausible dinosaur-sounding names across 6 different prefixes.

Part 2 Task 6 (Sampling): Both top-k and temperature are correctly implemented in predict_topk/sample_topk. Temperature is applied to logits before softmax (correct). Top-k filtering is applied before renormalizing. Multiple strategies are compared (random, top-k=5, top-k=3, temperatures 0.3/0.7/1.0/1.5/2.0) with clear generated examples and insightful written analysis.

What Needs Improvement

Task 4 (Evaluation) — cell not executed: Cell 23 (the confusion matrix / error analysis cell) has exec=—, meaning it was not run in the saved notebook. The outputs (accuracy, confusion matrix, 3 incorrect predictions) are present in the cell's existing output fields from a prior run, but the cell itself was not re-executed before submission. This is a minor issue since the outputs are visible and clearly correct, but it means the notebook is not fully reproducible as submitted. Deducting 2 pts.

Part 2 Task 4 (Training) — .to(device) bug: In the training function, inputs.to(device) and targets.to(device) do not reassign in-place (tensors are immutable; .to() returns a new tensor). The correct form is inputs = inputs.to(device). Since the notebook runs on CPU this doesn't affect correctness, but it is a latent bug that would break GPU training.
Suggestions

Task 4: Re-run all cells before submission so execution counts are continuous and outputs are freshly generated. Use Kernel → Restart & Run All to ensure reproducibility.

Part 2 Task 4 (Training): Fix the .to(device) calls: change inputs.to(device) → inputs = inputs.to(device) and targets.to(device) → targets = targets.to(device) to ensure the training loop correctly moves tensors to GPU when needed.

Part 2 Task 4 (Training): Consider plotting the loss curve as part of the training function output rather than separately in a never-executed cell (cell 50), so all visualizations run automatically.

Grading Summary

Part 1 Task 1 (Tokenization): 10 / 10 — Both tokenizers trained on train corpus only, 5 examples with table output, complete conceptual explanation.
Part 1 Task 2 (Word2Vec): 10 / 10 — BPE-tokenized inputs, both CBOW and Skip-gram, all 3 embedding sizes, similarity examples, full comparison and explanation.
Part 1 Task 3 (MLP): 10 / 10 — Correct mean pooling, valid MLP architecture, trained on training set, design choices documented, MLP suitability explained.
Part 1 Task 4 (Evaluation): 20 / 20 — Test accuracy, confusion matrix, and 3 error examples are all present and correct. Error analysis is strong.
Part 2 Task 1 (Dataset): 10 / 10 — All requirements met: start/end tokens, vocab/mappings, sliding window, shifted targets, len/getitem, train/val split, DataLoaders.
Part 2 Task 2 (One-hot): 5 / 5 — Correct shape (batch, seq_len, vocab_size), correct values, works for batches.
Part 2 Task 3 (Model forward pass): 5 / 5 — All layer shapes printed step-by-step, softmax applied, probabilities shown.
Part 2 Task 4 (Training): 9 / 10 — Correct training loop with one-hot encoding, Adam, CrossEntropyLoss, gradient clipping, validation loss reporting. Minor deduction (-1) for the latent .to(device) bug.
Part 2 Task 5 (Name generation): 5 / 5 — Correct generate loop with start/end token logic, names decoded to text, 6 prefixes demonstrated.
Part 2 Task 6 (Sampling): 15 / 15 — Top-k and temperature both correctly implemented, multiple strategies compared with examples.

Final Score: 99 / 100
