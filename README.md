# LÉ

On startup the bot checks for an existing model in `names/model.pt`. If the
file is missing, training is launched immediately in the background using
`asyncio.create_task`, allowing the bot to stay responsive while the model is
built.
