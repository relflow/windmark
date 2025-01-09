import json
import os

from fastapi import FastAPI

with open(os.getenv("STASH_FILENAME"), "r") as file:

    stash = json.load(file)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": stash}
