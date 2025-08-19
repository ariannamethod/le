# LÉ

On startup the bot checks for an existing model in `<work-dir>/model.pt`. The
working directory can be configured with the `LE_WORK_DIR` environment variable
or by passing `--work-dir` when invoking `le.py`. If the file is missing,
training is launched immediately in the background using
`asyncio.create_task`, allowing the bot to stay responsive while the model is
built.
