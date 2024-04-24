import logging
import json
from datetime import datetime
from utility_functions import *
import discord

# Add message to message history
def add_message_to_history(role, user_id, user_name, message_content, user_message_histories, history_file_name):
	try:
		if user_id not in user_message_histories:
			user_message_histories[user_id] = []
		# Add the message to the history
		user_message_histories[user_id].append({'role': role, 'name': user_name, 'content': message_content})
		# Fix the calculation of total_character_count to handle NoneType
		total_character_count = sum(len(entry['content']) for entry in user_message_histories.get(user_id, []) if entry is not None)
		while total_character_count > 6000:
			oldest_entry = user_message_histories[user_id].pop(0)
			total_character_count -= len(oldest_entry['content'])
		with open(history_file_name, 'w') as file:
			json.dump(user_message_histories, file)
		logging.debug("Message added to history and saved to file successfully.")
	except Exception as e:
		logging.error("An error occurred while writing to the JSON file in add_message_to_history: " + str(e))

# Save message histories to file
def save_message_histories(history_file_name, user_message_histories):
	try:
		with open(history_file_name, 'w') as file:
			json.dump(user_message_histories, file)
		logging.debug("Message histories saved to file successfully in save_message_histories.")
	except Exception as e:
		logging.error("An error occurred while saving message histories in save_message_histories: " + str(e))

# Function to handle message processing, modified to send responses in the thread
async def handle_message_processing(config, message, user_message_histories, history_file_name,bot):
	try:
		# Add user's message to history
		add_message_to_history('user', message.author.id, message.author.display_name, message.content, user_message_histories, history_file_name)
		channel = message.channel
		# Start typing in another coroutine
		logging.debug(f"include_personality parameter in handle_message_processing: {True}")

		# Pass include_personality=True when calling generate_response
		async with channel.typing():
			openairesponse = await generate_response(config, message.author.id, user_message_histories, message, include_personality=True)
		
		# Ensure response is a string
		if isinstance(openairesponse, list):
			openairesponse = ' '.join(openairesponse)

		# Send the response to the user in the thread
		if openairesponse:
			# logging.info(f"response from message processing",response)
			chunks = split_into_chunks(openairesponse)
			for chunk in chunks:
				if isinstance(message.channel, discord.Thread):
					await message.channel.send(chunk)  # Use message.channel instead of message.followup
				else:
					await message.channel.send(chunk)  # Use message.channel instead of message.followup
				# Add a short delay between messages
				await asyncio.sleep(0.5)  # Adjust delay as needed
	except Exception as e:
		logging.error("An error occurred during message processing in handle_message_processing: " + str(e))

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