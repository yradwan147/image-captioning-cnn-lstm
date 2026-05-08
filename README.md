# Image Captioning — CNN encoder + LSTM decoder

Final project for Udacity's *Computer Vision* nanodegree (nd891). An
encoder-decoder neural-captioning model trained on MS-COCO that takes an
image and emits a fluent English description.

## Architecture

```
                  ┌─────────────────────┐         ┌──────────────────────┐
   image ───────► │   ResNet-18  (frozen)│ ────►  │  Linear → BatchNorm  │ ──► feature
                  └─────────────────────┘         └──────────────────────┘

   feature  ─┬───►   ┌────┐    ┌────┐    ┌────┐
              │     │LSTM├──►─►│LSTM├─►─►│LSTM├─► … → vocab logits
              ▼     └────┘    └────┘    └────┘
        <start> token  →  <word_1> token  →  …
```

* `EncoderCNN`: ResNet-18 (pretrained, frozen) + linear projection +
  BatchNorm to `embed_size`.
* `DecoderRNN`: word embedding → LSTM (configurable layers + hidden) →
  per-token vocab logits.
* Training uses teacher forcing; inference is a simple greedy autoregressive
  loop in `DecoderRNN.sample`.

## Notebooks

```
0_Dataset.ipynb         Walk through MS-COCO + the data loader.
1_Preliminaries.ipynb   Sanity-check the encoder + decoder dimensions.
2_Training.ipynb        Train the model end-to-end.
3_Inference.ipynb       Caption new images interactively.
4_Zip … .ipynb          Bundle results for submission.
```

## Running

```bash
pip install torch torchvision pillow nltk pycocotools

# 1. Download MS-COCO 2014 train/val captions + images (the project README
#    in 0_Dataset.ipynb has the exact paths).

# 2. Build the vocab once (default vocab_threshold=5 yields ~9k tokens).
python -c "from vocabulary import Vocabulary; Vocabulary(vocab_threshold=5)"

# 3. Train (GPU strongly recommended — ~3 hours on a V100).
jupyter notebook 2_Training.ipynb

# 4. Use the trained checkpoints in models/ for captioning.
jupyter notebook 3_Inference.ipynb
```

## Standing-out work

* `EncoderCNN` adds a `BatchNorm1d(embed_size)` after the projection — a
  detail from the original Vinyals "Show and Tell" paper that stabilises
  training when the encoder is frozen.
* `DecoderRNN.forward` uses `captions[:, :-1]` so the LSTM is asked to
  predict the *next* token at every step, using teacher forcing during
  training.
* `sample` is concise — no while-loop, no <end> early-stopping (left to
  the caller so it can pick its own stop token).

## License

Educational submission for Udacity nd891. Starter scaffold © Udacity.
