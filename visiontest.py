import cv2
from io import BytesIO
import openai
import discord
from PIL import Image
import requests
from bot_init import init_bot
import numpy as np
import json

image_url=discord.Attachment.url    
async def describe_image(ctx, *, image_url):
    # Download the image from the URL provided by the user
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Load and preprocess the image using OpenCV as you normally would
    gray_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged_image = cv2.Canny(blurred_image, 100, 200)

    # Convert the edged image to a byte array and send it to the OpenAI API using requests
    byte_array = cv2.imencode('.jpg', edged_image)[1].tobytes()
    response = requests.post('http://192.168.128.134:5000/v1/images/generations', headers={'Authorization': 'Bearer sk-111111111111111111111111111111111111111111111111'}, files={"file": ('image.jpg', byte_array, 'application/octet-stream')})

    # Parse the response from the OpenAI API and extract the generated description
    response_json = json.loads(response.content)
    description = response_json['created']['url']

    # Create a Discord embed with the image and its description, then send it to the user using your bot's `send` method
    embed = discord.Embed(title="Image Description", color=0x00AE86)
    embed.set_image(url='attachment://image.jpg')
    embed.add_field(name="Description:", value=description, inline=False)
    await ctx.send(file=discord.File('image.jpg', filename='image.jpg'), embed=embed)

bot = init_bot()
@bot.event
async def on_message(message):
    while True:
        if message.author == bot.user:
            return

        await bot.process_commands(message)

        # Check for image attachments
        if 'attachment' in message.attachments:
            describe_image()
        else:
            init_bot()