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
	except Exception as e:
		logging.error("An error occurred while writing to the JSON file:", str(e))



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
		logging.error("An error occurred while loading config:", str(e))
	return None

# Load configuration
config_path = "Config.json"
config = load_config(config_path)

# Check if configuration loaded successfully
if config:
	# Use config here
	if config["OpenAPIKey"] is None:
		client = OpenAI(
			base_url=config["OpenAPIEndpoint"],
		)
	else:
		client = OpenAI(
			base_url=config["OpenAPIEndpoint"],
			api_key=config["OpenAPIKey"],
		)

	# Continue with the rest of your code
else:
	logging.error("Failed to load configuration. Bot cannot start.")


# Update OpenAI client initialization
# Consolidate OpenAI client initialization
if config["OpenAPIKey"] is None:
	client = OpenAI(
		base_url=config["OpenAPIEndpoint"]
	)
else:
	client = OpenAI(
		base_url=config["OpenAPIEndpoint"],
		api_key=config["OpenAPIKey"]
	)
# Load existing message histories from file
def load_message_histories(history_file_name):
	try:
		with open(history_file_name, 'r') as file:
			message_histories = json.load(file)
			return message_histories if message_histories is not None else {}
	except FileNotFoundError:
		logging.info("Message history file not found. Creating new one.")
		return {}
	except Exception as e:
		logging.error("An error occurred while loading message histories:", str(e))
		return {}

# Save message histories to file
def save_message_histories(history_file_name, user_message_histories):
	try:
		with open(history_file_name, 'w') as file:
			json.dump(user_message_histories, file)
	except Exception as e:
		logging.error("An error occurred while saving message histories:", str(e))

# Generate image using Comfy API
# def comfy_generate_image(comfy_prompt):
#     try:
#         logging.info("Sending request to Comfy API...")
#         image_data = workflow_api.main(comfy_prompt)
#         image = Image.open(io.BytesIO(image_data))
#         logging.info("Image generated successfully.")
#         return image
#     except Exception as e:
#         logging.error("An error occurred while generating image with Comfy API:", str(e))
#         return None


# Handle image generation
async def handle_image_generation(config, message, bucket):
	try:
		if bucket.consume(1):
			logging.info(f"Enough bucket tokens exist, running image generation")
			prompt = message.content.replace("draw", "")
			if "selfie" in prompt:
				logging.info("The prompt contains Selfie so we are appending Appearance to the prompt.")
				prompt = config["Appearance"] + prompt
				image = stable_diffusion_generate_image(config, prompt)
			elif "--upscale" in prompt:
				prompt = message.content.replace("--upscale", "")
				# Handle upscaling logic
				image = stable_diffusion_generate_image(config, prompt)
			else:
				image = stable_diffusion_generate_image(config, prompt)


			while not image_generated(image):
				async with message.channel.typing():
					await asyncio.sleep(1)

			image_bytes = io.BytesIO()
			image.save(image_bytes, format='PNG')
			image_bytes.seek(0)
			file = discord.File(image_bytes, filename='output.png')

			if isinstance(message.channel, discord.DMChannel):
				await message.author.send(file=file)
			else:
				await message.channel.send(file=file)
		else:
			await message.channel.send("Im busy sketching for you, please wait until I finish this one before asking for another.")
			logging.info("Image drawing throttled. Skipping draw request")
	except Exception as e:
		logging.error("An error occurred during image generation:", str(e))
	pass

# Generate image using Stable Diffusion API
def stable_diffusion_generate_image(config, prompt):
	try:
		response = requests.post(url=config["SDURL"], json={
			"prompt": config["SDPositivePrompt"] + prompt,
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
		logging.error("An error occurred during the Stable Diffusion API call:", str(e))
		return None


# Handle message processing
# Handle message processing
# Handle message processing
async def handle_message_processing(config, message, user_message_histories, history_file_name):
	try:
		add_message_to_history('user', message.author.id, message.author.display_name, message.content, user_message_histories, history_file_name)
		async with message.channel.typing():
			response = generate_response(config, message.author.id, user_message_histories)
		
		# Send the response in chunks if it exceeds 2000 characters
		if response:
			for chunk in response:
				if isinstance(message.channel, discord.DMChannel):
					await message.author.send(chunk)
				else:
					await message.channel.send(chunk)
				
				# Add a 100ms delay
				await asyncio.sleep(0.1)
	except Exception as e:
		logging.error("An error occurred during message processing:", str(e))
	pass

# Generate response using OpenAI API
def generate_response(config, user_id, user_message_histories):
	try:
		# Initialize OpenAI client with host and API key
		client = OpenAI(
			base_url=config["OpenAPIEndpoint"],
			api_key=config["OpenAPIKey"]
		)
		messages = [{'role': 'system', 'content': config["Personality"]}]
		for msg in list(user_message_histories[user_id]):
			if 'name' in msg:
				messages.append({'role': 'user', 'name': msg['name'], 'content': msg['content']})
			else:
				messages.append({'role': msg['role'], 'content': msg['content']})
		logging.info(f"Sending data to OpenAI: {messages[1:]}")
		response = client.chat.completions.create(
			messages=messages,
			model=config["OpenaiModel"]
		)
		logging.info("API response received.")
		
		# Check the length of the response
		response_text = response.choices[0].message.content
		if len(response_text) > 2000:
			# Split the response into chunks of 2000 characters
			chunks = [response_text[i:i+2000] for i in range(0, len(response_text), 2000)]
			return chunks
		else:
			return [response_text]
		
	except Exception as e:
		logging.error("An error occurred during OpenAI API call:", str(e))
		return None



# Check if image is generated
def image_generated(image):
	try:
		return image.size[0] > 0 and image.size[1] > 0
	except Exception as e:
		logging.error("An error occurred during image generation:", str(e))
		return False

# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the desired logging level for console output

# Create a formatter for console output
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Set the formatter for the console handler
console_handler.setFormatter(console_formatter)

# Add the console handler to the root logger
logging.getLogger().addHandler(console_handler)
file_handler = TimedRotatingFileHandler('app.log', when='midnight', interval=1, backupCount=7)
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
							if not any(word in message.content.lower() for word in config["IgnoredWords"]) and re.search(r"\b(draw|selfie)\b", message.content.lower()):
								await message.channel.send("Hang on while I get that for you...")
								await handle_image_generation(config, message, bucket)
							else:
								message.content = message.content.replace(config["Name"].lower(), "")
								await handle_message_processing(config, message, user_message_histories, history_file_name)
						else:
							return
					else:
						if not any(word in message.content.lower() for word in config["IgnoredWords"]) and re.search(r"\b(draw|selfie)\b", message.content.lower()):
							await message.channel.send("Hang on while I get that for you...")
							await handle_image_generation(config, message, bucket)
						else:
							if not any(word in message.content.lower() for word in config["IgnoredWords"]):
								await handle_message_processing(config, message, user_message_histories, history_file_name)
			except Exception as e:
				logging.error("An error occurred during message handling:", str(e))

		await bot.start(config["DiscordToken"])

	except KeyboardInterrupt:
		logging.info("Bot shutting down gracefully due to keyboard interrupt.")
		await bot.close()

	except Exception as e:
		logging.error("An error occurred during bot execution:", str(e))

	finally:
		logging.info("Closing bot...")
		await bot.close()


if __name__ == "__main__":
	config_path = "Config.json"
	config = load_config(config_path)
	if config:
		asyncio.run(run_bot(config))
	else:
		logging.error("Failed to load configuration. Bot cannot start.")