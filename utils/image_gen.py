import asyncio
import logging
import re
import json
import io
import nextcord
import aiohttp
import base64
from utility_functions import *
from PIL import Image
import time
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


async def handle_image_generation(config, message, user_message_histories, bot, draw_msg):
	try:
		
		bucket = TokenBucket(capacity=3, refill_rate=0.5)
		await start_typing(message)	
		if bucket.consume(1):
			logging.info(f"Enough bucket tokens exist, running image generation for message: {message.content}")
			bot_name = config["Name"]
			prompt, upscale, additional_instructions = parse_prompt(message.content, bot_name, config)

			# Log the parsed prompt
			logging.info("Parsed prompt in handle_image_generation:")
			logging.debug(prompt)

			# Check and load the current model
			sd_model_checkpoint = await check_current_model(config, message,draw_msg)

			# Proceed only if the model is successfully loaded
			if sd_model_checkpoint:
				# Generate response using OpenAI API
				openai_response = await generate_openai_response(config, message.author.id, user_message_histories, message, prompt, upscale, additional_instructions)
				logging.debug(openai_response)

				# Generate image
				image = await generate_image(config, prompt, upscale)

				# Send image response
				await send_image_response(message, image, openai_response, draw_msg, prompt=prompt)
			else:
				await draw_msg.edit("Failed to fetch the model. Please try again later.")
				logging.error("Failed to fetch the model. Skipping image generation.")
		else:
			await draw_msg.edit("I'm busy sketching for you. Please wait until I finish this one before asking for another.")
			logging.info("Image drawing throttled. Skipping draw request")
	except Exception as e:
		logging.error("An error occurred during image generation: " + str(e))
		await message.channel.send("I'm sorry, I was not able to do that for you. Please try again")
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


async def generate_openai_response(config, user_id, user_message_histories, message, prompt, upscale, additional_instructions):
	try:
		
		# Generate response using OpenAI API with additional instructions
		response = await generate_response(config, user_id, user_message_histories, message, prompt=prompt, include_personality=False, upscale=upscale, additional_instructions=additional_instructions)
		logging.info("OpenAI response generated successfully in generate_openai_response.")
		logging.debug(response)
		
		return response
	except Exception as e:
		logging.error("An error occurred during OpenAI response generation in generate_openai_response: " + str(e))
		if message:
			await message.channel.send("Your image request has been sent without additional processing.")
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


async def send_image_response(message, image, openai_response, draw_msg, prompt=""):
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
			if isinstance(message.channel, nextcord.DMChannel):
				await draw_msg.edit(file=file,content=openai_response[0])
				# await message.author.send(openai_response[0])
			else:
				await draw_msg.edit(file=file,content=openai_response[0])
				# await message.channel.send(openai_response[0])
		else:
			if isinstance(message.channel, nextcord.DMChannel):
				await draw_msg.delete()
				await message.channel.send(file=file,content=openai_response)
			else:
				await draw_msg.delete()
				await message.channel.send(file=file)
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
			logging.info(f"JSON Payload complete, Making API Call.")
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

async def check_current_model(config, message,draw_msg):
	try:
		options = await fetch_options(config)
		# logging.debug("Received options from the API in check_current_model: %s", options)

		# Check if the 'sd_model_checkpoint' key exists in the options
		if 'sd_model_checkpoint' in options:
			sd_model_checkpoint = options['sd_model_checkpoint']
			configured_model_checkpoint = config["SDModel"]
			if sd_model_checkpoint != configured_model_checkpoint:
				logging.warning("Loaded model does not match configured model in check_current_model.")
				if message:
					logging.warning("Sending message to notify about model mismatch in check_current_model...")
					if isinstance(message.channel, nextcord.DMChannel):
						await draw_msg.edit(content=draw_msg.content + "\n" + "Please wait, switching models...")
					else:
						await draw_msg.edit(content=draw_msg.content + "\n" + "It may take me a bit of time to do that. Please be patient")
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
			error_message = "No 'sd_model_checkpoint' key found in API options."
			logging.error(error_message)
			raise RuntimeError(error_message)
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
		error_message = f"An error occurred while checking available models: {str(e)}"
		logging.error(error_message)
		raise RuntimeError(error_message) from e

async def print_available_models(config,message):
	available_models = await check_models(config)
	if available_models:
		models_available = "\n".join(available_models)
		await message.channel.send(f"Here are the available models:\n{models_available}")
	else:
		await message.channel.send("Sorry there are no models available.")

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
		error_message = f"An error occurred while preparing JSON payload: {str(e)}"
		logging.error(error_message)
		raise RuntimeError(error_message) from e

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

async def send_message_in_thread(thread, content):
	try:
		await thread.send(content)
	except Exception as e:
		logging.error("An error occurred while sending message in thread: " + str(e))
