from openai import AsyncOpenAI
import logging
import asyncio
from io import BytesIO
import sys
import io
import tracemalloc
import traceback
import numpy as np
from logmmse import logmmse
import torch
import torchaudio
import discord
from discord.ext import voice_recv
import json
import watchdog
from threading import Thread
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from logging.handlers import TimedRotatingFileHandler
import torchaudio
import os
import logging
import textwrap
import re

discord.opus._load_default()

# Define a function to load the config
def load_config(config_path):
	try:
		with open(config_path, "r") as file:
			config = json.load(file)
			# logging.debug(f"Config file  successfully loaded with content: {config}")
		return config
	except FileNotFoundError:
		logging.error("Config file not found. Please check the file path.")
	except PermissionError:
		logging.error("Permission denied. Unable to open Config file.")
	except Exception as e:
		logging.error("An error occurred while loading config: " + str(e))
	return None
from silero.tts_utils import apply_tts
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

			# Include prompt if provided and not None
			if isinstance(prompt, str):
				logging.info("Using provided prompt:")
				logging.debug(prompt)
				messages.append({'role': 'system', 'content': prompt})

			logging.debug(f"Sending data to OpenAI: {messages}")

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



import torch
import torchaudio
import io
from silero import silero_tts

class VoiceRecvClient(voice_recv.VoiceRecvClient):
	# Initialize the TTS model as a class attribute
	language = 'en'
	model_id = 'v3_en'
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
							  model='silero_tts',
							  language=language,
							  speaker=model_id)
	model.to(device)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# Capture the current event loop at initialization for use in callbacks
		self.loop = asyncio.get_event_loop()

	def handle_audio_processing(self, ctx):
		def on_speech_recognized(user, text):
			logging.info(f"Speech recognized : {text}")
			# Use the captured loop to ensure the correct event loop is used
			asyncio.run_coroutine_threadsafe(self.process_recognized_speech(text, ctx), self.loop)
		sink = voice_recv.extras.SpeechRecognitionSink(default_recognizer="silero", text_cb=on_speech_recognized)
		self.listen(sink)

	@classmethod
	async def process_recognized_speech(cls, text:  str, message):
		try:
			global config
			user = None
			response_chunks = await cls.generate_response_text( text)
			if response_chunks:
				concatenated_response = ''
				for chunk in response_chunks:
					audio_bytes_list = cls.text_to_speech(chunk)
					logging.info(f"Response audio: {type(audio_bytes_list)} ")
					if audio_bytes_list:
						member = message.author
						if member.voice:
							voice_client = member.voice.channel.guild.voice_client
							if voice_client:
								for audio_bytes in audio_bytes_list:
									audio_source = discord.FFmpegPCMAudio(io.BytesIO(audio_bytes), pipe=True)
									voice_client.play(audio_source)
									logging.info("Playing response in voice channel...")
									while voice_client.is_playing():
										await asyncio. sleep(0.1)
									concatenated_response += chunk  # Concatenate the chunk to the response
									# Split concatenated response into chunks of at most 2000 characters
									# Respect sentence end and newlines inside the concatenated_response
									sentences = re.findall(r'[^.?!\ n]+[.?!\n]+', concatenated_response)
									chunks = []
									start = 0
									for sentence in sentences:
										if len(sentence) + start > 1950:
											chunks.append(concatenated_response[start:start+1950])
											start += 1950
									if start < len(concatenated_response):
										chunks.append(concatenated_response[start:])
									# Send the chunks as separate messages
									for chunk in chunks:
										await message.channel.send(chunk)

			else:
				logging.error("No response text generated.")
		except Exception as e:
			await message.channel.send("Sorry, I'm having trouble processing your request.")
			logging.error(e)

	@staticmethod
	async def generate_response_text(text: str)  ->   str or None:
		try:
			api_key = config["OpenAPIKey"]
			endpoint = config["OpenAPIEndpoint"]
			model = config["OpenaiModel"]
			messages = []
			personality_command = f"You are {config['Personality']} ." 
			messages.append({'role': 'system', 'content': personality_command})
			messages.append({'role': 'user', 'content': text}) 

			client = AsyncOpenAI(base_url=endpoint, api_key=api_key)
			response = await client.chat.completions.create(
				messages=messages,
				model=model
			)

			logging.info("API response received.")
			logging.debug(response)

			# Extract response text from the API response
			response_text = response.choices[0].message.content 

			# Split response text into chunks respecting sentence endings and newlines
			chunks = []
			start = 0
			for sentence in re.findall(r'[^.?!\ n]+[.?!\n]+', response_text):
				if len(sentence) + start > 1000:
					chunks.append(response_text[start:start+1000])
					start += 1000
			if start < len(response_text):
				chunks.append(response_text[start:])

			return chunks

		except Exception as e:
			logging.error(f"Error in generate_response_text: {e}")
			return None

	@classmethod
	def text_to_speech(cls, split_text):
		try:
			sample_rate = 48000
			speaker = config['SileroSpeaker']
			put_accent = True
			put_yo = True

			# Split the text into chunks of 1000 characters
			chunks = [split_text[i:i + 1000] for i in range(0, len(split_text), 1000)]

			# Convert each chunk to speech
			audio_bytes_list = []
			for chunk in chunks:
				audio = cls.model.apply_tts(text=chunk, speaker=speaker, sample_rate=sample_rate,
											put_accent=put_accent, put_yo=put_yo)

				# Assuming audio is a list of tensors, handling single text input
				audio_tensor = audio.unsqueeze(0)

				# Convert the generated audio tensor to bytes
				audio_bytes_io = io.BytesIO()
				torchaudio.save(audio_bytes_io, src=audio_tensor, sample_rate=sample_rate, format="wav")
				audio_bytes = audio_bytes_io.getvalue()

				audio_bytes_list.append(audio_bytes)

			# Return the list of audio bytes
			return audio_bytes_list

		except Exception as e:
			logging.error(f"Error in text_to_speech: {e}")
			return None