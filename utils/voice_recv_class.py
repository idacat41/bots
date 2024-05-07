import logging
import asyncio
import torch
import torchaudio
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import discord
from discord.ext import voice_recv
import io
from openai import AsyncOpenAI
import textwrap
import re
from silero import *
from discord.ext.voice_recv.extras import *
import sys
sys.path.insert(0, './utils/')  # Add this path to system paths for Python to be able to find modules in it.
from utils import *
from utils.utility_functions import config

class VoiceRecvClient(voice_recv.VoiceRecvClient):
	# Initialize the TTS model as a class attribute
	from silero.tts_utils import apply_tts
	language = 'en'
	model_id = 'v3_en'
	device = torch.device ("cpu")   #("cuda" if torch.cuda.is_available() else "cpu")
	silero_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
							  model='silero_tts',
							  language=language,
							  speaker=model_id
							  )
	silero_model.to(device)
	tts_queue = asyncio.Queue()
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Capture the current event loop at initialization for use in callbacks
		self.loop = asyncio.get_event_loop()
		
	def handle_audio_processing(self, ctx):
		def on_speech_recognized(user, text):
			bot_name = config["Name"].lower()
			logging.info(f"Speech recognized : {text}")
			if bot_name in text.lower():
				# Use the captured loop to ensure the correct event loop is used
				asyncio.run_coroutine_threadsafe(self.process_recognized_speech(text, ctx), self.loop)

		sink = SpeechRecognitionSink(default_recognizer="whisper", text_cb=on_speech_recognized)
		self.listen(sink)	

	@classmethod
	async def process_recognized_speech(cls, text:  str, message):
		try:
			global config
			user = None
			response_chunks = await cls.generate_response_text(text)
			if response_chunks:
				concatenated_response = ''
				for chunk in response_chunks:
					cls.text_to_speech(chunk)
					audio_bytes_list = await cls.tts_queue.get()
					logging.info(f"Response audio: {type(audio_bytes_list)} ")
					if audio_bytes_list:
						member = message.author
						if member.voice:
							voice_client = member.voice.channel.guild.voice_client
							if voice_client:
								for audio_bytes in audio_bytes_list:
									audio_source = discord.FFmpegPCMAudio(io.BytesIO(audio_bytes), pipe=True)
									voice_client.play(audio_source, after=lambda e: print('done', e))
									logging.info("Playing response in voice channel...")
									
									# Wait until the audio is finished playing
									while voice_client.is_playing():
										await asyncio.sleep(0.1)  # You can adjust the sleep duration if needed
									
									concatenated_response += chunk  # Concatenate the chunk to the response
									
									# Split concatenated response into chunks of at most 2000 characters, respecting sentence end and newlines
									max_length = 1900
									chunks = []
									lines = concatenated_response.split('\n')
									for line in lines:
										# Wrap the line into chunks of approximately 1900 characters
										wrapped_lines = textwrap.wrap(line, width=max_length, break_long_words=False)
										for wrapped_line in wrapped_lines:
											chunks.append(wrapped_line.strip())
									
									# Send the chunks as separate messages
									for chunk in chunks:
										await message.channel.send(chunk)
										
					else:
						logging.error("No response text generated.")

		except Exception as e:
			await message.channel.send("Sorry, I'm having trouble processing your request.")
			logging.error(e)
		finally:
			cls.tts_queue.task_done()

	@staticmethod
	async def generate_response_text(text: str)  ->   str | None:
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

			try:
				# If primary client fails, try with secondary client
				if config["SecondaryOpenAPIEndpoint"] and config["SecondaryOpenAPIKey"]:
					client = AsyncOpenAI(base_url=config["SecondaryOpenAPIEndpoint"], api_key=config["SecondaryOpenAPIKey"])
					response = await client.chat.completions.create(
						messages=messages,
						model=model
					)

					logging.info("API response received from secondary client.")
					logging.debug(response)

					# Extract response text from the API response
					response_text = response.choices[0].message.content 

					# Split response text into chunks respecting sentence endings and newlines
					chunks = []
					start = 0
					for sentence in re.findall(r'[^.?!\ n]+[.?!\n]+', response_text):
						if len(sentence) + start > 990:
							chunks.append(response_text[start:start+990])
							start += 990
					if start < len(response_text):
						chunks.append(response_text[start:])

					return chunks

			except Exception as e:
				logging.error(f"An error occurred during second OpenAI API call: {str(e)}")
				return None

	@classmethod
	def text_to_speech(cls, text):
		try:
			sample_rate = 48000
			speaker = config.get('SileroSpeaker')  # Using get to handle missing key
			if not speaker:
				raise ValueError("SileroSpeaker configuration not found")

			put_accent = True
			put_yo = True

			# Split text into sentences
			sentences = nltk.sent_tokenize(text)

			# Initialize variables 
			max_chunk_length = 1000
			chunks = []
			current_chunk = ''

			# Iterate over each sentence
			for sentence in sentences:
				# Split sentence into words
				words = nltk.word_tokenize(sentence)

				# Iterate over each word in the sentence
				for word in words:
					# Check if adding the current word to the current chunk would exceed the maximum length
					if len(current_chunk) + len(word) <= max_chunk_length:
						# Add the current word to the current chunk if it won't exceed the maximum  length
						current_chunk += word + ' '
					else:
						# If adding the current word would exceed the maximum length, split the chunk at the last space
						last_space_index = current_chunk.rfind(' ')
						chunks.append(current_chunk[:last_space_index].strip())
						current_chunk = word + ' '

			# Append the last chunk
			if current_chunk:
				chunks.append(current_chunk.strip())


			# Convert each chunk to speech
			audio_bytes_list = []
			for chunk in chunks:
				audio = cls.silero_model.apply_tts(text=chunk, speaker=speaker, sample_rate=sample_rate,
													put_accent=put_accent, put_yo=put_yo)

				# Assuming audio is a list of tensors, handling single text input
				audio_tensor = audio.unsqueeze(0)

				# Convert the generated audio tensor to bytes
				audio_bytes_io = io.BytesIO()
				torchaudio.save(audio_bytes_io, src=audio_tensor, sample_rate=sample_rate, format="wav")
				audio_bytes = audio_bytes_io.getvalue()

				audio_bytes_list.append(audio_bytes)
			
			# Return the list of audio bytes
			cls.tts_queue.put_nowait(audio_bytes_list)
			logging.info("TTS Conversion Completed")
		except ValueError as ve:
			logging.error(f"ValueError in text_to_speech: {ve}")
		except Exception as e:
			logging.error(f"Error in text_to_speech: {e}")
			# Optionally, raise the exception to propagate it further
			raise
		return None