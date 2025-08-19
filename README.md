# LÉ

LÉ takes a text file as input, assuming each line represents one training
example, and generates new examples in the same style. Under the hood it is an
autoregressive character-level language model with options ranging from simple
bigrams to a Transformer architecture similar to GPT. The project aims to be a
hackable, single-file reference implementation that only depends on
[PyTorch](https://pytorch.org).

On startup the bot checks for an existing model in `<work-dir>/model.pt`. The
working directory can be configured with the `LE_WORK_DIR` environment variable
or by passing `--work-dir` when invoking `le.py`. If the file is missing,
training is launched immediately in the background using `asyncio.create_task`,
allowing the bot to stay responsive while the model is built.

## Usage

The repository ships with a small dataset at `blood/lines01.txt` containing
short lines used for training, e.g.:

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

To train a model and save it to the `names` directory:

```bash
python le.py -i blood/lines01.txt -o names
```

Training progress, logs, and the model are stored under the output directory.
Many more configurations are available via command-line options. Training runs
on CPU but benefits from a GPU when available.

To sample manually from the best model so far, run:

```bash
python le.py -i blood/lines01.txt -o names --sample-only
```

Example generated names after a short training run:

```
dontell
khylum
camatena
aeriline
```

## License

GNU General Public License v3.0

