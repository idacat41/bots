import cv2
from PIL import Image
import pytesseract
from discord.ext import commands
import openai
import discord
import numpy as np
import logging
from logging.handlers import TimedRotatingFileHandler
from io import BytesIO
import requests
import json
import os

# tessdata_path = '/home/josh/.local/lib/python3.10/site-packages/pytesseract/'  # adjust the path according to your Tesseract installation
# pytesseract.tesseract_cmd = tessdata_path + 'tesseract.py'


token = "MTIyNjAyNjQ1OTM5NTY1MzY3NA.GMFevz.GHtTnplJdoFjZSCcU-Lyhc71YBA2rwSWNREJXw"

openai.api_key="sk-111111111111111111111111111111111111111111111111"
openai.base_url='http://192.168.128.134:5000/v1/completions/'

# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a file handler to write logs to a file
file_handler = TimedRotatingFileHandler('app.log', when='midnight', interval=1, backupCount=7)
file_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger = logging.getLogger()
logger.addHandler(file_handler)

# intents = discord.Intents(messages=True, guilds=True)
# bot = commands.Bot(command_prefix='!', intents=intents)
# # message=discord.message

intents = discord.Intents().all()
client = commands.Bot(command_prefix='!', intents=intents)

# @bot.event
# async def on_ready():
#     print('Logged in as {0}'.format(bot.user))



import discord
from PIL import Image
from pytesseract import image_to_string
import requests
import io
import base64
import sys

OpenAI_API_KEY = openai.api_key
OpenCV_LANGUAGE = 'eng' # Set the language for OCR
OpenCV_OCR_PATH = "/usr/bin/tesseract" # Path to Tesseract executable (Windows) or binary file (Linux)

# client = discord.Client()

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    for attachment in message.attachments:
        try:
            # Download the image from Discord's server and convert it to OpenCV format
            img = Image.open(io.BytesIO(base64.b64decode(attachment.filename)))
            
            # Set up Tesseract for OCR
            if sys.platform == 'darwin':  # macOS
                tessdata_path = '/usr/local/Cellar/tesseract/latest/share/tesseract'
            elif sys.platform == 'linux':  # Linux
                tessdata_path = '/usr/share/tesseract-ocr/4'
            
            config = r'--oem 1 --psm 6' if OpenCV_LANGUAGE == 'eng' else ''
            pytesseract.pytesseract.tesseract_cmd = [OpenCV_OCR_PATH, f"-l {OpenCV_LANGUAGE}", '-c', 'tessedit_char_whitelist=0123456789'] if sys.platform == 'win32' else tessdata_path
            
            # Perform OCR and pass the result to OpenAI API for processing
            description = pytesseract.image_to_string(img)
            response = requests.post(f"{openai.base_url}", headers={"Authorization": f"Bearer {OpenAI_API_KEY}"}, data={'model': 'davinci-codex', 'prompt': description, 'max_tokens': 2048})
            response = response.json()['choices'][0]['text']
            
            # Send the result back to the user in a new message
            await message.channel.send(response)
        except Exception as e:
            print("Error:", str(e))

    

client.run(token)