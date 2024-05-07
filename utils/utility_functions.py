from functools import cache
from pyexpat import model
from openai import AsyncOpenAI, completions
import logging
import asyncio
from io import BytesIO
import sys
import discord
import json
from threading import Thread
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from logging.handlers import TimedRotatingFileHandler
import aiohttp
from discord.ext.voice_recv.extras import *

discord.opus._load_default()
# Define a function to load the config
def load_config(config_path):
	try:
		with open(config_path, "r") as file:
			config = json.load(file)
			return config
	except FileNotFoundError:
		logging.error("Config file not found. Please check the file path.")
	except PermissionError:
		logging.error("Permission denied. Unable to open Config file.")
	except Exception as e:
		logging.error("An error occurred while loading config: " + str(e))
	return None
# Define a class to handle file system events
class ConfigFileHandler(FileSystemEventHandler):
	def __init__(self, config_path, on_change):
		super().__init__()
		self.config_path = config_path
		self.on_change = on_change

	def on_modified(self, event):
		if event.src_path == self.config_path:
			logging.info("Config file has been modified. Reloading config...")
			reload_config()
			log_setup(config)
		else:
			logging.debug("Detected modification in a file, but it is not the config file.")

# Define a function to observe config changes
def observe_config_changes():
	# Create a FileSystemEventHandler to watch the config file
	handler = ConfigFileHandler(config_path, reload_config)
	# Start observing the config file
	observer = PollingObserver()
	observer.schedule(handler, config_path, recursive=False)
	observer.start()

	try:
		while True:
			observer.join()
	except  KeyboardInterrupt:
		observer.stop()
		observer.join()


# Define a function to reload the config
def reload_config():
	global config
	config = load_config(config_path)
	logging.info(f"Config file successfully loaded with content: {config}")

# Define the config path
config_path = "./config/config.json"

# Load the config initially
config = load_config(config_path)

# Start file system observer in a separate thread
observer_thread = Thread(target=observe_config_changes, daemon=True)
observer_thread.start()

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

async def generate_response(config, user_id, user_message_histories, message=None, additional_instructions=None, prompt=None, include_personality=True, upscale=False):
	try:
		client = AsyncOpenAI(base_url=config["OpenAPIEndpoint"], api_key=config["OpenAPIKey"])
		return await try_generate_response(client, config, user_id, user_message_histories, message, additional_instructions, prompt, include_personality)
	except Exception as e:
		logging.error("An error occurred during OpenAI API call: " + str(e))
		try:
			# If primary client fails, try with secondary client
			if config["SecondaryOpenAPIEndpoint"] and config["SecondaryOpenAPIKey"]:
				client = AsyncOpenAI(base_url=config["SecondaryOpenAPIEndpoint"], api_key=config["SecondaryOpenAPIKey"])
				return await try_generate_response(client, config, user_id, user_message_histories, message, additional_instructions, prompt, include_personality)
		except Exception as e:
			logging.error(f"An error occurred during second OpenAI API call: {str(e)}")
			if message:	
				await message.channel.send("I can't talk right now please try back later.")		
			raise  # Re-raise the exception to propagate it back to the caller
async def try_generate_response(client, config, user_id, user_message_histories, message, additional_instructions, prompt, include_personality):
	try:
			# Determine recent message content
			recent_message_content = ""
			if user_id in user_message_histories and user_message_histories[user_id]:
				recent_message_content = user_message_histories[user_id][-2]['content'] if len(user_message_histories[user_id]) > 1 else ""
			logging.debug(f"Recent message content: {recent_message_content}")

			# Construct messages
			messages = []

			# Include additional instructions
			if additional_instructions:
				messages.extend(additional_instructions)
				logging.info("Included additional instructions.")

			# Include personality message
			if include_personality:
				personality_command = f"You are {config['Personality']}."
				messages.append({'role': 'system', 'content': personality_command})
				logging.info("Included personality message.")

			# Include recent message content if prompt is None
			if recent_message_content and (message is None or recent_message_content != message.content) and prompt is None:
				messages.append({'role': 'user', 'content': recent_message_content})
				logging.info("Included recent user message.")

			# Include message content if provided and if prompt is None
			if message and isinstance(message.content, str) and prompt is None:
				messages.append({'role': 'user', 'content': message.content})
				logging.info("Included message content from provided message.")
			logging.info(f"Sending data to OpenAI: {messages}")

			# Include prompt if provided and not None
			if isinstance(prompt, str):
				logging.info("Using provided prompt:")
				logging.debug(prompt)
				messages.append({'role': 'system', 'content': prompt})

			# Send request to OpenAI
			response = await client.chat.completions.create(
				messages=messages,
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

async def load_model(config, message): 
	"""Loads a Stable Diffusion model and updates the config file with the new model name."""
	async with message.channel.typing():
		sd_model = message.content
		# Send another POST request to the specified URL with the specified JSON payload
		async with aiohttp.ClientSession() as session1:
		# Refresh checkpoints
			url1 = config["SDURL"] + "/sdapi/v1/refresh-checkpoints"
			async with session1.post(url1) as response1:
				response1.raise_for_status()
				status1 = await response1.text()
				logging.info("Refreshed checkpoints")

		# Construct the JSON payload for the POST request
		json_payload = {
			"sd_model_checkpoint": sd_model.replace("load model ", "")
		}
		# Send the POST request to load the model
		async with aiohttp.ClientSession()  as session:
			async with session.post(url=config["SDURL"] + "/sdapi/v1/options", json=json_payload) as response:
				response.raise_for_status()

		# Update the config file with the new model name
		with open("config/config.json", "r") as f:
			config = json.load(f)

		config["SDmodel"] = sd_model.replace("load model ", "")

		with open("config/config.json", "w") as f:
			json.dump(config, f, indent=4)
			f.flush()
		logging.info ("Model loaded.")