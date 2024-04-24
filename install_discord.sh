#!/usr/bin/env bash
git clone https://github.com/Rapptz/discord.py
cd discord.py
uv pip install -U .[voice]
cd ..