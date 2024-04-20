import nextcord
from math import log
import json
import time
import logging
import asyncio
from logging.handlers import TimedRotatingFileHandler
import os
import sys
import threading
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from nextcord import Interaction, TextChannel, Message
import asyncio
from collections import namedtuple
# Import the utils package from a specific path '/utils/'
sys.path.insert(0, './utils/')  # Add this path to system paths for Python to be able to find modules in it.
from utils import *
from utils.image_gen import handle_image_generation
from utils.image_gen import print_available_models
from utils.interactions import handle_interaction_processing
from utils import utility_functions
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

# define a Task named tuple to hold the configuration, interaction, bucket, user_interaction_histories and bot objects.
Task = namedtuple('Task','config, interaction, user_interaction_histories, bot, history_file_name')
Draw_Task = namedtuple('Draw_Task','config, interaction, user_interaction_histories, bot, draw_msg')

async def worker(image_queue):
	while True:  # an infinite loop that runs until the program is stopped
		config, interaction, user_interaction_histories, bot, draw_msg = await image_queue.get()
		try:
			await handle_image_generation(config, interaction, user_interaction_histories, bot, draw_msg)
		except Exception as e:
			logging.error("An error occurred in image generation queue: " + str(e))
		finally:
			image_queue.task_done()  # mark the task as done once it's finished no matter if there was an exception or not

async def worker2(interaction_queue):
	while True:
		config, interaction, user_interaction_histories, history_file_name, bot = await interaction_queue.get()
		try:
			await handle_interaction_processing(config, interaction, user_interaction_histories, bot, history_file_name)
		except Exception as e:
			logging.error("An error occurred in interaction processing queue: " + str(e))
		finally:
			interaction_queue.task_done()

# initialize the queues
image_queue = asyncio.Queue()
interaction_queue = asyncio.Queue()

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
import logging

async def start_bot():
	try:
		# Set up logging
		config = load_config(config_path)
		log_setup(config)
		history_file_name = config["Name"] + "_interaction_histories.json"
		user_interaction_histories = load_interaction_histories(history_file_name)
		intents = nextcord.Intents.all()
		intents.message_content = True
		bot = nextcord.AutoShardedClient(intents=intents)
		
		# Initialize queues
		# image_queue = asyncio.Queue()
		# interaction_queue = asyncio.Queue()
		
		# Load configuration
		reload_config()  # Reload the config after setting up logging, this will help catch any errors that occur during config loading
		
		# Start worker tasks
		asyncio.create_task(worker(image_queue))
		asyncio.create_task(worker2(interaction_queue))
		
		@bot.event
		async def on_ready():
			logging.info('Logged in as {0.user}'.format(bot))
			
		@bot.event
		async def on_message(interaction):
			logging.info('Message from {0.author}: {0.content}'.format(Message))
   
			try:
				if interaction.author.id == bot.user.id:
					return
				# Handle interactions here
				await handle_interactions(config, interaction, user_interaction_histories, history_file_name, bot, interaction_queue,image_queue)
				
			except Exception as e:
				logging.error("An error occurred during message handling: %s", str(e))
		await bot.start(config["DiscordToken"])
	except Exception as e:
		logging.critical('Failed to start bot due to exception: %s', str(e))

		logging.error(f"An error occurred during bot startup: {str(e)}")
async def handle_interactions(config, interaction, user_interaction_histories, history_file_name, bot,interaction_queue,image_queue):
	# Check if the interaction author is ignored or if any ignored words are present in the interaction
	ignored_users = config.get("IgnoredUsers", [])
	ignored_words = config.get("IgnoredWords", [])

	try:
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
		if interaction.channel.type == nextcord.ChannelType.private and (not interaction.author.id in config["AllowedDMUsers"] or not config["AllowDMResponses"]):                
			await interaction.channel.send("I Cannot talk here. Please try a regular Channel.")
			logging.info("Message is from a DM.")
			return
		
		if interaction.author.id in config["bot_user"]:
			logging.info(f"Message author is ignored. Ignoring interaction.")
			return	

		# Check if the interaction author is ignored
		if interaction.author.id in ignored_users and not bot["bot_user"]:
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
			draw_msg = await interaction.channel.send("Hang on while I get that for you...")
			image_queue.put_nowait(Draw_Task(config, interaction, user_interaction_histories, bot, draw_msg))
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
			interaction_queue.put_nowait(Task(config, interaction, user_interaction_histories, bot, history_file_name))  # replace config etc with actual values
	except Exception as e:
		logging.error("An error occurred during bot startup: " + str(e))

# Run the bot
asyncio.run(start_bot())
