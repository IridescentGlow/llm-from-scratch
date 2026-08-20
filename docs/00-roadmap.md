# Roadmap

The map before the territory. Six stages, in order. Each gets its own doc
and its own working code before moving to the next.

1. **Tokenization** — turn text into numbers the model can process.
2. **Data pipeline** — turn a pile of text into batches of training examples.
3. **Architecture** — build the transformer that predicts the next token.
4. **Pretraining loop** — actually train it on raw text, watch loss drop.
5. **Evaluation** — measure whether it's any good, beyond "loss went down."
6. **Fine-tuning** — turn a raw text-predictor into something that follows
   instructions.

That's the whole arc: text → numbers → batches → model → trained model →
measured model → useful model.

See `docs/01-tokenization.md` onward for the "what and why" of each stage.
