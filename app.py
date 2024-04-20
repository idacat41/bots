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
from nextcord import message, TextChannel, Message
import asyncio
from collections import namedtuple
# Import the utils package from a specific path '/utils/'
sys.path.insert(0, './utils/')  # Add this path to system paths for Python to be able to find modules in it.
from utils import *
from utils.image_gen import handle_image_generation
from utils.image_gen import print_available_models
from utils.interactions import handle_message_processing
from utils import utility_functions
from threading import Thread

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


# Load existing message histories from file
def load_message_histories(history_file_name):
	try:
		if os.path.exists(history_file_name):
			with open(history_file_name, 'r') as file:
				message_histories = json.load(file)
				return message_histories if message_histories is not None else {}
		else:
			logging.info("Message history file not found in load_message_histories. Creating a new one.")
			with open(history_file_name, 'w') as file:
				json.dump({}, file)  # Create a new empty message history file
			return {}
	except Exception as e:
		logging.error("An error occurred while loading message histories in load_message_histories: " + str(e))
		return {}

# define a Task named tuple to hold the configuration, message, bucket, user_message_histories and bot objects.
Task = namedtuple('Task','config, message, user_message_histories, bot, history_file_name')
Draw_Task = namedtuple('Draw_Task','config, message, user_message_histories, bot, draw_msg')

async def worker(image_queue):
	while True:  # an infinite loop that runs until the program is stopped
		config, message, user_message_histories, bot, draw_msg = await image_queue.get()
		try:
			await handle_image_generation(config, message, user_message_histories, bot, draw_msg)
		except Exception as e:
			logging.error("An error occurred in image generation queue: " + str(e))
		finally:
			image_queue.task_done()  # mark the task as done once it's finished no matter if there was an exception or not

async def worker2(message_queue):
	while True:
		config, message, user_message_histories, history_file_name, bot = await message_queue.get()
		try:
			await handle_message_processing(config, message, user_message_histories, bot, history_file_name)
		except Exception as e:
			logging.error("An error occurred in message processing queue: " + str(e))
		finally:
			message_queue.task_done()

# initialize the queues
image_queue = asyncio.Queue()
message_queue = asyncio.Queue()

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
		history_file_name = config["Name"] + "_message_histories.json"
		user_message_histories = load_message_histories(history_file_name)
		intents = nextcord.Intents.all()
		bot = nextcord.AutoShardedClient(intents=intents)
			
		# Load configuration
		reload_config()  # Reload the config after setting up logging, this will help catch any errors that occur during config loading
		
		# Start worker tasks
		asyncio.create_task(worker(image_queue))
		asyncio.create_task(worker2(message_queue))
		
		@bot.event
		async def on_ready():
			logging.info('Logged in as {0.user}'.format(bot))
			
		@bot.event
		async def on_message(message):
			try:
				if message.author.id == bot.user.id:
					return
				# Handle messages here
				await handle_messages(config, message, user_message_histories, history_file_name, bot, message_queue,image_queue)
				
			except Exception as e:
				logging.error("An error occurred during message handling: %s", str(e))
		await bot.start(config["DiscordToken"])
	except Exception as e:
		logging.critical('Failed to start bot due to exception: %s', str(e))

		logging.error(f"An error occurred during bot startup: {str(e)}")

async def handle_messages(config, message, user_message_histories, history_file_name, bot,message_queue,image_queue):
	# Check if the message author is ignored or if any ignored words are present in the message
	ignored_users = config.get("IgnoredUsers", [])
	ignored_words = config.get("IgnoredWords", [])

	try:
		# Check if the message starts with '!'
		if message.content.startswith('!'):
			logging.debug("Message starts with '!'. Ignoring message.")
			return
		
		# Logging additional information about the message
		logging.info(f"Received message from {message.author} in channel {message.channel}")
		logging.debug(f"message content: {message.content}")

		# Check if the message is in an allowed channel
		if not isinstance(message.channel, (nextcord.Thread, nextcord.DMChannel)):
			if message.channel.id not in config["AllowedChannels"]:
				logging.info("Message not in an allowed channel. Ignoring message.")
				return
		
		logging.info(f"Received message: {message.content}")

		# Check if the message is from a DM
		if message.channel.type == nextcord.ChannelType.private and (message.author.id in config["AllowedDMUsers"] or not config["AllowDMResponses"]):
			await message.channel.send("I Cannot talk here. Please try a regular Channel.")
			logging.info("User: %s (%s)", message.author.name, message.author.id)
			logging.info("Message is from an unauthorized DM.")
			return

		if message.author.id in config["bot_user"]:
			logging.info(f"Message author is ignored. Ignoring message.")
			return	

		# Check if the message author is ignored
		if message.author.id in ignored_users and not bot["bot_user"]:
			logging.info("Message is from an ignored user. Ignoring message.")
			await message.channel.send("I'm sorry, I cannot talk to you.")
			return
		
		# Check if any ignored words are present in the message
		if any(word.lower() in message.content.lower() for word in ignored_words):
			logging.info("Message contains ignored words. Ignoring message.")
			return
		
		# Check if the bot's name is mentioned
		bot_name = config["Name"]
		if not message.channel.type == nextcord.ChannelType.private:
			if config.get("OnlyWhenCalled") and bot_name.lower() not in message.content.lower():
				logging.info("Message does not contain bot name. Ignoring message.")
				return
		else:
			logging.info("Message contains bot name or bot is configured to respond without mention.")

		logging.debug(f"Received message: {message.content}")
		

		# Process messages without further checks
		if "draw" in message.content.lower() or "send" in message.content.lower():
			draw_msg = await message.channel.send("Hang on while I get that for you...")
			image_queue.put_nowait(Draw_Task(config, message, user_message_histories, bot, draw_msg))
		elif "check models" in message.content.lower():
			if message.channel.type == nextcord.ChannelType.private:
				await print_available_models(config,message)
			else: 
				await message.channel.send("I'm sorry, I cannot run that here.")
				return
		elif "load model" in message.content.lower():
			if message.channel.type == nextcord.ChannelType.private:
				return
			else:
				await message.channel.send("I'm sorry, I cannot run that here.")
		elif "join_vc" in message.content.lower():
			if message.message.author != None:
				return
			else:
				return 'You need to be in a voice channel to use this command'
		else:
			message_queue.put_nowait(Task(config, message, user_message_histories, bot, history_file_name))
	except Exception as e:
		logging.error("An error occurred during bot startup: " + str(e))

# Run the bot
asyncio.run(start_bot())
