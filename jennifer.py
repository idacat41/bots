from pyexpat.errors import messages
from typing import Self
import nextcord
from math import log
from openai import OpenAI
from collections import deque
import json
import aiohttp
import io
import base64
import time
import logging
import asyncio
from logging.handlers import TimedRotatingFileHandler
from PIL import Image
import re
import os
import sys
import threading
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from nextcord import Interaction, TextChannel, Message

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

# Add interaction to interaction history
def add_interaction_to_history(role, user_id, user_name, interaction_content, user_interaction_histories, history_file_name):
	try:
		if user_id not in user_interaction_histories:
			user_interaction_histories[user_id] = []
		# Add the interaction to the history
		user_interaction_histories[user_id].append({'role': role, 'name': user_name, 'content': interaction_content})
		# Fix the calculation of total_character_count to handle NoneType
		total_character_count = sum(len(entry['content']) for entry in user_interaction_histories.get(user_id, []) if entry is not None)
		while total_character_count > 6000:
			oldest_entry = user_interaction_histories[user_id].pop(0)
			total_character_count -= len(oldest_entry['content'])
		with open(history_file_name, 'w') as file:
			json.dump(user_interaction_histories, file)
		logging.debug("Message added to history and saved to file successfully.")
	except Exception as e:
		logging.error("An error occurred while writing to the JSON file in add_interaction_to_history: " + str(e))


# Define a class to handle file system events
class ConfigFileHandler(FileSystemEventHandler):
	def __init__(self, config_path, on_change):
		super().__init__()
		self.config_path = config_path
		self.on_change = on_change

	def on_modified(self, event):
		if event.src_path == self.config_path:
			logging.info("Config file has been modified. Reloading config...")
			self.on_change()
		else:
			logging.debug("Detected modification in a file, but it is not the config file.")


# Define a function to reload the config
def reload_config():
	global config
	config = load_config(config_path)

# Update the load_config function to not call the reload_config function
def load_config(config_path):
	try:
		with open(config_path, "r") as file:
			config = json.load(file)
			# logging.debug(f"Config file successfully loaded with content: {config}")
		return config
	except FileNotFoundError:
		logging.error("Config file not found. Please check the file path.")
	except PermissionError:
		logging.error("Permission denied. Unable to open Config file.")
	except Exception as e:
		logging.error("An error occurred while loading config: " + str(e))
	return None

async def start_typing(interaction):
    # Define the "typing" text and the delay between each character
    typing_text = "Typing..."
    typing_delay = 0.1  # Adjust as needed
    
    channel = interaction.channel   # get the Channel object from interaction
        
    for _ in typing_text:
        await channel.trigger_typing()
        await asyncio.sleep(typing_delay)   # Pause before the next character

# Adjusted load_config function with threaded file monitoring
def load_config_with_observer(config_path):
	def observe_config_changes():
		try:
			observer = PollingObserver()
			event_handler = ConfigFileHandler(config_path, reload_config)
			observer.schedule(event_handler, path=os.path.dirname(config_path), recursive=False)
			observer.start()
			logging.info("Observer thread started successfully.")
			while True:
				time.sleep(1)  # Add a sleep to keep the thread alive
		except Exception as e:
			logging.error("An error occurred while starting the observer thread: " + str(e))

	# Start file system observer in a separate thread
	observer_thread = threading.Thread(target=observe_config_changes, daemon=True)
	observer_thread.start()

	# Initial load of config
	config = load_config(config_path)
	return config


# Load existing interaction histories from file
def load_interaction_histories(history_file_name):
	try:
		if os.path.exists(history_file_name):
			with open(history_file_name, 'r') as file:
				interaction_histories = json.load(file)
				return interaction_histories if interaction_histories is not None else {}
		else:
			logging.info("Message history file not found in load_interaction_histories. Creating a new one.")
			with open(history_file_name, 'w') as file:
				json.dump({}, file)  # Create a new empty interaction history file
			return {}
	except Exception as e:
		logging.error("An error occurred while loading interaction histories in load_interaction_histories: " + str(e))
		return {}

# Save interaction histories to file
def save_interaction_histories(history_file_name, user_interaction_histories):
	try:
		with open(history_file_name, 'w') as file:
			json.dump(user_interaction_histories, file)
		logging.debug("Message histories saved to file successfully in save_interaction_histories.")
	except Exception as e:
		logging.error("An error occurred while saving interaction histories in save_interaction_histories: " + str(e))

async def generate_response(config, user_id, user_interaction_histories, interaction=None, additional_instructions=None, prompt=None, include_personality=True, upscale=False):
	try:
		client = OpenAI(base_url=config["OpenAPIEndpoint"], api_key=config["OpenAPIKey"])
		return await try_generate_response(client, config, user_id, user_interaction_histories, interaction, additional_instructions, prompt, include_personality)
	except Exception as e:
		logging.error("An error occurred during OpenAI API call: " + str(e))
		try:
			# If primary client fails, try with secondary client
			if config["SecondaryOpenAPIEndpoint"] and config["SecondaryOpenAPIKey"]:
				client = OpenAI(base_url=config["SecondaryOpenAPIEndpoint"], api_key=config["SecondaryOpenAPIKey"])
				return await try_generate_response(client, config, user_id, user_interaction_histories, interaction, additional_instructions, prompt, include_personality)
		except Exception as e:
			logging.error(f"An error occurred during second OpenAI API call: {str(e)}")
			if interaction:	
				await interaction.channel.send("I can't talk right now please try back later.")		
			raise  # Re-raise the exception to propagate it back to the caller

async def try_generate_response(client, config, user_id, user_interaction_histories, interaction, additional_instructions, prompt, include_personality):
	try:
			# Determine recent interaction content
			recent_interaction_content = ""
			if user_id in user_interaction_histories and user_interaction_histories[user_id]:
				recent_interaction_content = user_interaction_histories[user_id][-2]['content'] if len(user_interaction_histories[user_id]) > 1 else ""
			logging.debug(f"Recent interaction content: {recent_interaction_content}")

			# Construct interactions
			interactions = []

			# Include additional instructions
			if additional_instructions:
				interactions.extend(additional_instructions)
				logging.info("Included additional instructions.")

			# Include personality interaction
			if include_personality:
				personality_command = f"You are {config['Personality']}."
				interactions.append({'role': 'system', 'content': personality_command})
				logging.info("Included personality interaction.")

			# Include recent interaction content if prompt is None
			if recent_interaction_content and (interaction is None or recent_interaction_content != interaction.content) and prompt is None:
				interactions.append({'role': 'user', 'content': recent_interaction_content})
				logging.info("Included recent user interaction.")

			# Include interaction content if provided and if prompt is None
			if interaction and isinstance(interaction.content, str) and prompt is None:
				interactions.append({'role': 'user', 'content': interaction.content})
				logging.info("Included interaction content from provided interaction.")

			# Include prompt if provided and not None
			if isinstance(prompt, str):
				logging.info("Using provided prompt:")
				logging.debug(prompt)
				interactions.append({'role': 'system', 'content': prompt})

			logging.debug(f"Sending data to OpenAI: {interactions}")

			# Send request to OpenAI
			response = client.chat.completions.create(
				messages=interactions,
				model=config["OpenaiModel"]
			)

			logging.info("API response received.")
			logging.debug(response)

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
		raise  # Re-raise the exception to propagate it back to the caller



async def handle_image_generation(config, interaction, bucket, user_interaction_histories,bot):
	try:
		await start_typing(interaction)	
		if bucket.consume(1):
			logging.info(f"Enough bucket tokens exist, running image generation for interaction: {interaction.content}")
			bot_name = config["Name"]
			prompt, upscale, additional_instructions = parse_prompt(interaction.content, bot_name, config)

			# Log the parsed prompt
			logging.info("Parsed prompt in handle_image_generation:")
			logging.debug(prompt)

			# Check and load the current model
			sd_model_checkpoint = await check_current_model(config,interaction)

			# Proceed only if the model is successfully loaded
			if sd_model_checkpoint:
				# Generate response using OpenAI API
				openai_response = await generate_openai_response(config, interaction.author.id, user_interaction_histories, interaction, prompt, upscale, additional_instructions)
				logging.debug(openai_response)

				# Generate image
				image = await generate_image(config, prompt, upscale)

				# Send image response
				await send_image_response(interaction, image, openai_response, prompt=prompt)
			else:
				await interaction.channel.send("Failed to fetch the model. Please try again later.")
				logging.error("Failed to fetch the model. Skipping image generation.")
		else:
			await interaction.channel.send("I'm busy sketching for you. Please wait until I finish this one before asking for another.")
			logging.info("Image drawing throttled. Skipping draw request")
	except Exception as e:
		logging.error("An error occurred during image generation: " + str(e))
		pass

def parse_prompt(prompt, bot_name, config):
	try:
		# Log the original prompt
		logging.debug("Original Prompt:")
		logging.debug(prompt)
		
		# Remove "send" and/or "draw" from the prompt
		prompt = prompt.replace("send", "").replace("draw", "").strip()

		# Check if "--upscale" is present
		upscale = "--upscale" in prompt

		# Remove "--upscale" from the prompt
		prompt = prompt.replace("--upscale", "").strip()

		additional_instructions = [
			{
				'role': 'system',
				'content': {
					'SDOpenAI': config.get("SDOpenAI", ""),
					'steps': config.get("steps", {}),
					'additional_steps': config.get("additional_steps", {})
				}
			}
		]

		appearance_info = []

		if "selfie" in prompt or "you" in prompt:
			prompt = prompt.replace(bot_name, "").strip()
			logging.info("The prompt contains Selfie so we are appending Appearance to the prompt.")
			appearance_info = [{'role': 'system', 'content': config.get("Appearance", prompt)}]

		else:
			# Remove bot's name from the prompt
			prompt = re.sub(r'\b{}\b'.format(re.escape(bot_name)), '', prompt, flags=re.IGNORECASE).strip()
			# Log the prompt after removing bot's name
			logging.debug("Prompt after removing bot's name:")
			logging.debug(prompt)

		logging.info("Prompt parsed successfully.")
		
		# Construct prompt with additional_instructions, appearance_info, and prompt
		prompt = json.dumps(additional_instructions + appearance_info) + " " + prompt
		
		return prompt, upscale, additional_instructions
	except Exception as e:
		logging.error("An error occurred during prompt parsing: " + str(e))
		return "", upscale, []  # Return an empty list for additional_instructions


async def generate_openai_response(config, user_id, user_interaction_histories, interaction, prompt, upscale, additional_instructions):
	try:
		
		# Generate response using OpenAI API with additional instructions
		response = await generate_response(config, user_id, user_interaction_histories, interaction, prompt=prompt, include_personality=False, upscale=upscale, additional_instructions=additional_instructions)
		logging.info("OpenAI response generated successfully in generate_openai_response.")
		logging.debug(response)
		
		return response
	except Exception as e:
		logging.error("An error occurred during OpenAI response generation in generate_openai_response: " + str(e))
		if interaction:
			await interaction.channel.send("Your image request has been sent without additional processing.")
		return None


async def generate_image(config, prompt, upscale):
	try:
		image = await stable_diffusion_generate_image(config, prompt, upscale=upscale)
		if image is None:
			raise ValueError("Failed to generate image in generate_image.")
		logging.info("Image generated successfully in generate_image.")
		return image
	except Exception as e:
		logging.error("An error occurred during image generation in generate_image: " + str(e))
		raise  # Re-raise the exception to propagate it back to the caller


async def send_image_response(interaction, image, openai_response, prompt=""):
	try:
		if openai_response:
			prompt += " " + " ".join(openai_response)
			logging.debug("OpenAi Prompt in send_image_response: %s", prompt)

		if image is None:
			raise ValueError("Image object is NoneType in send_image_response.")

		image_bytes = io.BytesIO()
		image.save(image_bytes, format='PNG')
		image_bytes.seek(0)
		file = nextcord.File(image_bytes, filename='output.png')

		if openai_response:
			if isinstance(interaction.channel, nextcord.DMChannel):
				await interaction.author.send(file=file)
				await interaction.author.send(openai_response[0])
			else:
				await interaction.channel.send(file=file)
				await interaction.channel.send(openai_response[0])
		else:
			if isinstance(interaction.channel, nextcord.DMChannel):
				await interaction.author.send(file=file)
			else:
				await interaction.channel.send(file=file)
		logging.info("Image response sent successfully in send_image_response.")
	except Exception as e:
		logging.error("An error occurred during sending image response in send_image_response: " + str(e))
		raise  # Re-raise the exception to propagate it back

# Generate image using Stable Diffusion API
async def stable_diffusion_generate_image(config, prompt, upscale):
	try:
		json_payload = prepare_json_payload(config, prompt, upscale)
		logging.debug(json.dumps(json_payload))
		if json_payload:
			response = await make_api_call(config, json_payload)
			if response:
				return process_response(response)
		# If any step fails, raise an exception to indicate image generation failure
		raise ValueError("Failed to generate image in stable_diffusion_generate_image.")
	except Exception as e:
		logging.error("An error occurred during image generation in stable_diffusion_generate_image: " + str(e))
		raise  # Re-raise the exception to propagate it back to the caller

async def fetch_options(config):
	try:
		async with aiohttp.ClientSession() as session:
			url = config["SDURL"] + "/sdapi/v1/options"
			async with session.get(url) as response:
				response.raise_for_status()
				options = await response.json()
				logging.info("Options fetched successfully in fetch_options.")
				return options
	except aiohttp.ClientError as e:
		# Log the error and raise it
		logging.error(f"HTTP error occurred: {e}, url={url}")
		raise
	except Exception as e:
		# Log the error and raise it
		logging.error(f"An error occurred while fetching options from the Stable Diffusion API in fetch_options: {e}")
		raise

async def check_current_model(config, interaction):
	try:
		options = await fetch_options(config)
		# logging.debug("Received options from the API in check_current_model: %s", options)

		# Check if the 'sd_model_checkpoint' key exists in the options
		if 'sd_model_checkpoint' in options:
			sd_model_checkpoint = options['sd_model_checkpoint']
			configured_model_checkpoint = config["SDModel"]
			if sd_model_checkpoint != configured_model_checkpoint:
				logging.warning("Loaded model does not match configured model in check_current_model.")
				if interaction:
					logging.warning("Sending interaction to notify about model mismatch in check_current_model...")
					if isinstance(interaction.channel, nextcord.DMChannel):
						await interaction.author.send("Please wait, switching models...")
					else:
						await interaction.channel.send("It may take me a bit of time to draw that picture. Please be patient")
				# Construct the JSON payload for the POST request
				json_payload = {
					"sd_model_checkpoint": configured_model_checkpoint
				}
				# Send the POST request to load the model
				async with aiohttp.ClientSession() as session:
					async with session.post(url=config["SDURL"] + "/sdapi/v1/options", json=json_payload) as response:
						response.raise_for_status()
						logging.info("Model loaded successfully in check_current_model.")
			else:
				logging.info("Loaded model matches configured model in check_current_model.")
			return sd_model_checkpoint, configured_model_checkpoint
		else:
			error_interaction = "No 'sd_model_checkpoint' key found in API options."
			logging.error(error_interaction)
			raise RuntimeError(error_interaction)
	except Exception as e:
		logging.error(f"An error occurred in check_current_model: {e}")
		raise e

async def check_models(config):
	async def fetch_models():
		async with aiohttp.ClientSession() as session:
			url = config["SDURL"] + "/sdapi/v1/sd-models"
			async with session.get(url) as response:
				response.raise_for_status()
				available_models_data = await response.json()
				logging.info("Models fetched successfully in fetch_models.")
				return available_models_data

	try:
		available_models_data = await fetch_models()
		available_models = [model['model_name'] for model in available_models_data]
		logging.info("Options fetched successfully in fetch_options.")
		return available_models
	except Exception as e:
		error_interaction = f"An error occurred while checking available models: {str(e)}"
		logging.error(error_interaction)
		raise RuntimeError(error_interaction) from e

async def print_available_models(config,interaction):
	available_models = await check_models(config)
	if available_models:
		models_available = "\n".join(available_models)
		await interaction.channel.send(f"Here are the available models:\n{models_available}")
	else:
		await interaction.channel.send("Sorry there are no models available.")

def prepare_json_payload(config, prompt, upscale):
	try:
		json_payload = {
			"prompt": config.get("SDPositivePrompt") + (prompt if isinstance(prompt, str) else " ".join(prompt)),
			"steps": config.get("SDSteps"),
			"width": config.get("SDWidth"),
			"height": config.get("SDHeight"),
			"negative_prompt": config.get("SDNegativePrompt"),
			"sampler_index": config.get("SDSampler"),
			"cfg_scale": config.get("SDConfig"),
			"self_attention": "yes",
			"enable_hr": upscale,
			"hr_upscaler": "R-ESRGAN 4x+",
			"hr_prompt": config.get("SDPositivePrompt") + (prompt if isinstance(prompt, str) else " ".join(prompt)),
			"hr_negative_prompt": config.get("SDNegativePrompt"),
			"denoising_strength": 0.5,
			"override_settings_restore_afterwards": False,
			"override_settings": {
				"sd_model_checkpoint": config.get("SDModel"),
				"CLIP_stop_at_last_layers": config.get("SDClipSkip")
			}
		}
		logging.info("JSON payload prepared successfully in prepare_json_payload.")
		logging.debug(f"{json_payload}")
		return json_payload
	except Exception as e:
		error_interaction = f"An error occurred while preparing JSON payload: {str(e)}"
		logging.error(error_interaction)
		raise RuntimeError(error_interaction) from e

async def make_api_call(config, json_payload):
	try:
		async with aiohttp.ClientSession() as session:
			async with session.post(url=config["SDURL"] + "/sdapi/v1/txt2img", json=json_payload) as response:
				response.raise_for_status()  # Raise an exception for HTTP errors (status codes 4xx and 5xx)
				return await response.json()
	except aiohttp.ClientError as e:
		# Log the error or handle it appropriately
		logging.error(f"HTTP error occurred: {e}")
		return None
	except Exception as e:
		# Handle other exceptions
		logging.error(f"An error occurred: {e}")
		return None

def process_response(response):
	try:
		image_data = base64.b64decode(response['images'][0])
		image = Image.open(io.BytesIO(image_data))
		logging.debug("Image generated successfully in process_response.")
		return image
	except Exception as e:
		logging.error("An error occurred while processing API response in process_response: " + str(e))
		return None

# Check if image is generated
def image_generated(image):
	try:
		return image.size[0] > 0 and image.size[1] > 0
	except Exception as e:
		logging.error("An error occurred during image generation in image_generated: " + str(e))
		return False

async def send_interaction_in_thread(thread, content):
	try:
		await thread.send(content)
	except Exception as e:
		logging.error("An error occurred while sending interaction in thread: " + str(e))

# Function to handle interaction processing, modified to send responses in the thread
async def handle_interaction_processing(config, interaction, user_interaction_histories, history_file_name,bot):
	try:
		# Add user's interaction to history
		add_interaction_to_history('user', interaction.author.id, interaction.author.display_name, interaction.content, user_interaction_histories, history_file_name)
		
		# Start typing in another coroutine
		await start_typing(interaction)

		logging.debug(f"include_personality parameter in handle_interaction_processing: {True}")

		# Pass include_personality=True when calling generate_response
		openairesponse = await generate_response(config, interaction.author.id, user_interaction_histories, interaction, include_personality=True)

		# typing_event.set()  # Stop the typing indicator when we get a response
		
		# Ensure response is a string
		if isinstance(openairesponse, list):
			openairesponse = ' '.join(openairesponse)

		# Send the response to the user in the thread
		if openairesponse:
			# logging.info(f"response from interaction processing",response)
			chunks = split_into_chunks(openairesponse)
			for chunk in chunks:
				if isinstance(interaction.channel, nextcord.Thread):
					await interaction.channel.send(chunk)
				else:
					await interaction.channel.send(chunk)

				# Add a short delay between interactions
				await asyncio.sleep(0.5)  # Adjust delay as needed
	except Exception as e:
		logging.error("An error occurred during message processing: " + str(e))

# Function to split the response into chunks respecting sentence boundaries and new lines
import textwrap
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

# Task queue for handling image generation
async def image_generation_queue(config, interaction, bucket, user_interaction_histories,bot):
	try:
		await handle_image_generation(config, interaction, bucket, user_interaction_histories,bot)
	except Exception as e:
		logging.error("An error occurred in image generation queue: " + str(e))

# Task queue for handling interaction processing
async def interaction_processing_queue(config, interaction, user_interaction_histories, history_file_name,bot):
	try:
		await handle_interaction_processing(config, interaction, user_interaction_histories, history_file_name,bot)
	except Exception as e:
		logging.error("An error occurred in interaction processing queue: " + str(e))

def log_setup(config):
	# Assuming "LogLevel" is a key in the config dictionary
	log_level = config.get("LogLevel", "INFO")  # Default to INFO if LogLevel is not present
	
	logging.basicConfig(
		level=log_level,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			TimedRotatingFileHandler("app.log", when="midnight", backupCount=7, encoding='utf-8'),
			logging.StreamHandler(sys.stdout)
		]
	)

# Get the current working directory
cwd = os.getcwd()

# Define the filename
filename = "config.json"

# Construct the full path to the config file
config_path = os.path.join(cwd, "config", filename)

# Now you can pass config_path to your load_config function
config = load_config(config_path)

# Define a function to start the sharded bot tasks
async def start_bot():
	try:
		log_setup(config)
		history_file_name = config["Name"] + "_interaction_histories.json"
		user_interaction_histories = load_interaction_histories(history_file_name)
		bucket = TokenBucket(capacity=3, refill_rate=0.5)
		intents = nextcord.Intents.all()
		bot = nextcord.AutoShardedClient(intents=intents)

		@bot.event
		async def on_ready():
			logging.info(f'Logged in as {bot.user}')

		@bot.event
		async def on_message(interaction):
			try:
				reload_config()
				if interaction.author == bot.user:
					return

				# Check if the interaction starts with '!'
				if interaction.content.startswith('!'):
					logging.debug("Message starts with '!'. Ignoring interaction.")
					return
				# Logging additional information about the interaction
				logging.info(f"Received interaction from {interaction.author} in channel {interaction.channel}")
				logging.debug(f"Interaction content: {interaction.content}")

				# Check if the interaction is in an allowed channel
				if not isinstance(interaction.channel, (nextcord.Thread, nextcord.DMChannel)):
					if interaction.channel.id not in config["AllowedChannels"]:
						logging.info("Message not in an allowed channel. Ignoring interaction.")
						return
				logging.info(f"Received interaction: {interaction.content}")

				# Check if the interaction is from a DM

				if interaction.channel.type == nextcord.ChannelType.private and (not interaction.author.id in config["AllowedDMUsers" or config["AllowDMResponses"]]):				
					await interaction.channel.send("I Cannot talk here. Please try a regular Channel.")
					logging.info("Message is from a DM.")
					# Process the interaction here
					return
					# Check if the interaction author is ignored or if any ignored words are present in the interaction
				ignored_users = config.get("IgnoredUsers", [])
				ignored_words = config.get("IgnoredWords", [])

				if interaction.author.id in config["bot_user"]:
					logging.info(f"Message author is ignored. Ignoring interaction.")
					return

				# Check if the interaction author is ignored
				if interaction.author.id in ignored_users and not config["bot_user"]:
					logging.info("Message is from an ignored user. Ignoring interaction.")
					await interaction.channel.send("I'm sorry, I cannot talk to you.")
					return

				# Check if any ignored words are present in the interaction
				if any(word.lower() in interaction.content.lower() for word in ignored_words):
					logging.info("Message contains ignored words. Ignoring interaction.")
					return

				# Check if the bot's name is mentioned
				bot_name = config["Name"]
				if not interaction.channel.type == nextcord.ChannelType.private:
					if config.get("OnlyWhenCalled") and bot_name.lower() not in interaction.content.lower():
						logging.info("Message does not contain bot name. Ignoring interaction.")
						return
				else:
					logging.info("Message contains bot name or bot is configured to respond without mention.")
						# Process the interaction here

				logging.debug(f"Received interaction: {interaction.content}")
				

				# Process interactions without further checks
				if "draw" in interaction.content.lower() or "send" in interaction.content.lower():
					await interaction.channel.send("Hang on while I get that for you...")
					await image_generation_queue(config, interaction, bucket, user_interaction_histories,bot)
				elif "check models" in interaction.content.lower():
					if interaction.channel.type == nextcord.ChannelType.private:
						await print_available_models(config,interaction)
					else: 
						await interaction.channel.send("I'm sorry, I cannot run that here.")
						return
				elif "load model" in interaction.content.lower():
					if interaction.channel.type == nextcord.ChannelType.private:
						return
					else:
						await interaction.channel.send("I'm sorry, I cannot run that here.")
				elif "join_vc" in interaction.content.lower():
					if interaction.interaction.author != None:
						return
					else:
						return 'You need to be in a voice channel to use this command'
				else:
					await interaction_processing_queue(config, interaction, user_interaction_histories, history_file_name,bot)
			except Exception as e:
				logging.error("An error occurred during interaction handling: " + str(e))

		await bot.start(config["DiscordToken"])
	except Exception as e:
		logging.error("An error occurred during bot startup: " + str(e))

# Run the bot
asyncio.run(start_bot())