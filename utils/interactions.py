import logging
import json
from datetime import datetime
from utility_functions import *
import nextcord

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

# Save interaction histories to file
def save_interaction_histories(history_file_name, user_interaction_histories):
	try:
		with open(history_file_name, 'w') as file:
			json.dump(user_interaction_histories, file)
		logging.debug("Message histories saved to file successfully in save_interaction_histories.")
	except Exception as e:
		logging.error("An error occurred while saving interaction histories in save_interaction_histories: " + str(e))

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
					await interaction.channel.send(chunk)  # Use interaction.channel instead of interaction.followup
				else:
					await interaction.channel.send(chunk)  # Use interaction.channel instead of interaction.followup
				# Add a short delay between interactions
				await asyncio.sleep(0.5)  # Adjust delay as needed
	except Exception as e:
		logging.error("An error occurred during message processing in handle_interaction_processing: " + str(e))

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