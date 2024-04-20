from openai import AsyncOpenAI
import logging
import asyncio

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


async def start_typing(message):
    # Define the "typing" text and the delay between each character
    typing_text = "Typing..."
    typing_delay = 0.1  # Adjust as needed
    
    channel = message.channel   # get the Channel object from message
        
    for _ in typing_text:
        await channel.trigger_typing()
        await asyncio.sleep(typing_delay)   # Pause before the next character
