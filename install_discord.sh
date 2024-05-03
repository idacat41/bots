#!/usr/bin/env bash
set -eux
#install discord.py
git clone https://github.com/Rapptz/discord.py
cd discord.py
pip3 install -U .[voice]
cd ..