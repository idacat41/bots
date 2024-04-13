import discord
from math import log
from openai import OpenAI
from collections import deque
import json
import requests
import io
import base64
import time
import logging
import asyncio
from logging.handlers import TimedRotatingFileHandler
from PIL import Image
import re

# Token Bucket class for rate limiting
class TokenBucket:
	def __init__(self, capacity, refill_rate):
		self.capacity = capacity
		self.tokens = capacity
		self.last_refill_time = time.time()
		self.refill_rate = refill_rate

	def consume(self, tokens):
		current_time = time.time()
		self.tokens = min(self.capacity, self.tokens + (current_time - self.last_refill_time) * self.refill_rate)
		self.last_refill_time = current_time

		if tokens <= self.tokens:
			self.tokens -= tokens
			return True
		else:
			return False

# Add message to message history
def add_message_to_history(role, user_id, user_name, message_content, user_message_histories, history_file_name):
	if user_id not in user_message_histories:
		user_message_histories[user_id] = []
	user_message_histories[user_id].append({'role': role, 'name': user_name, 'content': message_content})
	# Fix the calculation of total_character_count to handle NoneType
	total_character_count = sum(len(entry['content']) for entry in user_message_histories.get(user_id, []) if entry is not None)
	while total_character_count > 6000:
		oldest_entry = user_message_histories[user_id].pop(0)
		total_character_count -= len(oldest_entry['content'])
	try:
		with open(history_file_name, 'w') as file:
			json.dump(user_message_histories, file)
			logging.info("Message added to history and saved to file successfully.")
	except Exception as e:
		logging.error("An error occurred while writing to the JSON file: " + str(e))

# Load configuration from file
def load_config(config_path):
	try:
		with open(config_path, "r") as file:
			config = json.load(file)
		logging.info(f"Config file successfully opened with content: {config}")
		return config
	except FileNotFoundError:
		logging.error("Config file not found. Please check the file path.")
	except PermissionError:
		logging.error("Permission denied. Unable to open Config file.")
	except Exception as e:
		logging.error("An error occurred while loading config: " + str(e))
	return None

# Load existing message histories from file
import os

def load_message_histories(history_file_name):
	try:
		if os.path.exists(history_file_name):
			with open(history_file_name, 'r') as file:
				message_histories = json.load(file)
				return message_histories if message_histories is not None else {}
		else:
			logging.info("Message history file not found. Creating a new one.")
			with open(history_file_name, 'w') as file:
				json.dump({}, file)  # Create a new empty message history file
			return {}
	except Exception as e:
		logging.error("An error occurred while loading message histories: " + str(e))
		return {}


# Save message histories to file
def save_message_histories(history_file_name, user_message_histories):
	try:
		with open(history_file_name, 'w') as file:
			json.dump(user_message_histories, file)
		logging.info("Message histories saved to file successfully.")
	except Exception as e:
		logging.error("An error occurred while saving message histories: " + str(e))

# Generate response using OpenAI API
def generate_response(config, user_id, user_message_histories, additional_instructions=None, prompt=None, include_personality=True):
	try:
		# Initialize OpenAI client with host and API key
		client = OpenAI(
			base_url=config["OpenAPIEndpoint"],
			api_key=config["OpenAPIKey"]
		)
		
		# Check if the user has any message history
		if user_id in user_message_histories and user_message_histories[user_id]:
			# Get the most recent message from the user's message history
			recent_message = user_message_histories[user_id][-1]['content']
		else:
			# If there's no message history for the user, use an empty string as the recent message
			recent_message = ""

		# Include personality if specified
		messages = []
		if include_personality:
			messages.append({'role': 'system', 'content': config["Personality"]})
			logging.info("Included personality message.")

		# Include recent message if not empty
		if recent_message:
			messages.append({'role': 'user', 'content': recent_message})
			logging.info("Included recent user message.")

		# Append additional instructions if provided
		if additional_instructions:
			messages.extend(additional_instructions)
			logging.info("Included additional instructions.")

		# Log the prompt
		if prompt:
			if isinstance(prompt, str):
				logging.info("Using provided prompt:")
				logging.info(prompt)
				messages.append({'role': 'system', 'content': prompt})
			else:
				logging.warning("Prompt must be provided as a string.")
				logging.info("Using constructed messages:")
				logging.info(messages)
		else:
			logging.info("No prompt provided. Using constructed messages:")
			logging.info(messages)

		logging.info(f"Sending data to OpenAI: {messages}")

		# Send request to OpenAI
		response = client.chat.completions.create(
			messages=messages,
			model=config["OpenaiModel"]
		)

		logging.info("API response received.")

		# Check the length of the response
		response_text = response.choices[0].message.content
		if len(response_text) > 2000:
			# Split the response into chunks of 2000 characters
			chunks = [response_text[i:i + 2000] for i in range(0, len(response_text), 2000)]
			return chunks
		else:
			return [response_text]

	except Exception as e:
		logging.error("An error occurred during OpenAI API call: " + str(e))
		return None


# Handle image generation
async def handle_image_generation(config, message, bucket, user_message_histories):
	try:
		if bucket.consume(1):
			logging.info(f"Enough bucket tokens exist, running image generation for message: {message.content}")
			prompt = message.content.replace("draw", "")
			if prompt is not None:
				logging.info(prompt)

			if "selfie" in prompt or "you" in prompt:
				logging.info("The prompt contains Selfie so we are appending Appearance to the prompt.")
				appearance_info = [{'role': 'system', 'content': config.get("Personality", prompt)}]
				prompt += " " + json.dumps(appearance_info)  # Append appearance information to the prompt
						
			additional_instructions = [{'role': 'system', 'content': config.get("SDOpenAI", "")}]  # Fetch additional instructions
			logging.info(f"Additional instructions for OpenAI: {additional_instructions}")

			# Call generate_response to get OpenAI response
			logging.info(prompt)
			openai_response = generate_response(config, message.author.id, user_message_histories, additional_instructions=additional_instructions, prompt=prompt, include_personality=False)
			logging.info(f"OpenAI response: {openai_response}")

			if openai_response:
				# Process the response as needed
				prompt += " " + " ".join(openai_response)
				logging.info("OpenAi Prompt: %s", prompt)

			if "--upscale" in prompt:
				prompt = prompt.replace("--upscale", "")
				# Handle upscaling logic
				image = stable_diffusion_generate_image(config, prompt)
			else:
				# Continue with regular image generation
				image = stable_diffusion_generate_image(config, prompt)

			while not image_generated(image):
				async with message.channel.typing():
					await asyncio.sleep(1)

			image_bytes = io.BytesIO()
			image.save(image_bytes, format='PNG')
			image_bytes.seek(0)
			file = discord.File(image_bytes, filename='output.png')

			if openai_response:
				# Extract the string from the list
				openai_response_str = openai_response[0]

				# Remove newlines inside square brackets
				response_without_newlines = re.sub(r'\[([^]]+)\]', lambda x: x.group(0).replace('\n', ' '), openai_response_str)

				if isinstance(prompt, str):
					prompt += " " + response_without_newlines
				else:
					prompt = " ".join(prompt) + " " + response_without_newlines

				# logging.info("OpenAi Prompt: %s", prompt)

				if isinstance(message.channel, discord.DMChannel):
					await message.author.send(file=file)
					await message.author.send(openai_response_str)
				else:
					await message.channel.send(file=file)
					await message.channel.send(openai_response_str)

			else:
				# If there's no OpenAI response, just send the image
				if isinstance(message.channel, discord.DMChannel):
					await message.author.send(file=file)
				else:
					await message.channel.send(file=file)
		else:
			await message.channel.send("I'm busy sketching for you. Please wait until I finish this one before asking for another.")
			logging.info("Image drawing throttled. Skipping draw request")
	except Exception as e:
		logging.error("An error occurred during image generation: " + str(e))
		pass

# Generate image using Stable Diffusion API
def stable_diffusion_generate_image(config, prompt):
	try:
		response = requests.post(url=config["SDURL"], json={
			"prompt": config["SDPositivePrompt"] + (prompt if isinstance(prompt, str) else " ".join(prompt)),
			"steps": config["SDSteps"],
			"width": config["SDWidth"],
			"height": config["SDHeight"],
			"negative_prompt": config["SDNegativePrompt"],
			"sampler_index": config["SDSampler"]
		})
		response.raise_for_status()
		logging.info("Stable Diffusion API call successful.")
		r = response.json()
		image_data = base64.b64decode(r['images'][0])
		image = Image.open(io.BytesIO(image_data))
		logging.info("Image generated successfully.")
		return image
	except requests.exceptions.RequestException as e:
		logging.error("An error occurred during the Stable Diffusion API call: " + str(e))
		return None
	
# Check if image is generated
def image_generated(image):
	try:
		return image.size[0] > 0 and image.size[1] > 0
	except Exception as e:
		logging.error("An error occurred during image generation: " + str(e))
		return False

# Handle message processing
async def handle_message_processing(config, message, user_message_histories, history_file_name):
	try:
		add_message_to_history('user', message.author.id, message.author.display_name, message.content, user_message_histories, history_file_name)
		async with message.channel.typing():
			response = generate_response(config, message.author.id, user_message_histories)
		
		# Ensure response is a string
		if isinstance(response, list):
			response = ' '.join(response)
		
		# Send the response in chunks respecting word boundaries
		if response:
			chunks = split_into_chunks(response)
			for chunk in chunks:
				if isinstance(message.channel, discord.DMChannel):
					await message.author.send(chunk)
				else:
					await message.channel.send(chunk)
				
				# Add a 100ms delay
				await asyncio.sleep(0.05)
	except Exception as e:
		logging.error("An error occurred during message processing: " + str(e))

import textwrap

# Function to split the response into chunks respecting sentence boundaries and new lines
def split_into_chunks(response):
	max_length = 1900
	chunks = []
	lines = response.split('\n')
	for line in lines:
		# Wrap the line into chunks of approximately 1900 characters
		wrapped_lines = textwrap.wrap(line, width=max_length, break_long_words=False)
		for wrapped_line in wrapped_lines:
			chunks.append(wrapped_line.strip())
	return chunks


# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = TimedRotatingFileHandler('app.log', when='midnight', interval=1, backupCount=7, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(file_handler)


async def run_bot(config):
	try:
		history_file_name = config["Name"] + "_message_histories.json"
		user_message_histories = load_message_histories(history_file_name)
		bucket = TokenBucket(capacity=3, refill_rate=0.5)
		intents = discord.Intents.all()
		intents.message_content = True
		bot = discord.Client(intents=intents)

		@bot.event
		async def on_ready():
			logging.info(f'Logged in as {bot.user}')

		@bot.event
		async def on_message(message):
			try:
				import re
				if message.author == bot.user or message.author.id in config["IgnoredUsers"]:
					return
				if message.channel.id in config["AllowedChannels"] or isinstance(message.channel, discord.DMChannel):
					if config["OnlyWhenCalled"]:
						if config["Name"].lower() in message.content.lower() or isinstance(message.channel, discord.DMChannel) or bot.user in message.mentions:
							if not any(word in message.content.lower() for word in config["IgnoredWords"]) and re.search(r"\b(draw|selfie|send)\b", message.content.lower()):
								await message.channel.send("Hang on while I get that for you...")
								await handle_image_generation(config, message, bucket, user_message_histories)
							else:
								message.content = message.content.replace(config["Name"].lower(), "")
								await handle_message_processing(config, message, user_message_histories, history_file_name)
						else:
							return
					else:
						if not any(word in message.content.lower() for word in config["IgnoredWords"]) and re.search(r"\b(draw|selfie|send)\b", message.content.lower()):
							await message.channel.send("Hang on while I get that for you...")
							await handle_image_generation(config, message, bucket, user_message_histories)
						else:
							if not any(word in message.content.lower() for word in config["IgnoredWords"]):
								await handle_message_processing(config, message, user_message_histories, history_file_name)
			except Exception as e:
				logging.error("An error occurred during message handling: " + str(e))

		await bot.start(config["DiscordToken"])

	except KeyboardInterrupt:
		logging.info("Bot shutting down gracefully due to keyboard interrupt.")
		await bot.close()

	except Exception as e:
		logging.error("An error occurred during bot execution: " + str(e))

	finally:
		logging.info("Closing bot...")
		await bot.close()


if __name__ == "__main__":
	config_path = "config.json"
	config = load_config(config_path)
	if config:
		asyncio.run(run_bot(config))
	else:
		logging.error("Configuration loading failed. Exiting.")