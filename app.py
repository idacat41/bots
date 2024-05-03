import discord
from discord import VoiceState
from openai import AsyncOpenAI
from openai import OpenAI
from math import log
import json
import time
import logging
import asyncio
import os
import sys
import aiohttp
from discord.errors import *
# from discord import message, TextChannel, Message
from collections import namedtuple
# Import the utils package from a specific path '/utils/'
sys.path.insert(0, './utils/')  # Add this path to system paths for Python to be able to find modules in it.
from utils import *
from utils.image_gen import handle_image_generation
from utils.image_gen import print_available_models
from utils.interactions import handle_message_processing
from utils.voice_recv_class import *
from utils.utility_functions import *
import tracemalloc
tracemalloc.start()

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

async def join_voice_channel(ctx, voice_channel_id: int):
	try:
		channel = ctx.guild.get_channel(voice_channel_id)
		vc = await channel.connect(cls=VoiceRecvClient)
		logging.info(f"Joined voice channel: {channel}")
		is_listening = vc.is_listening()
		logging.info(f"Initial listening state: {is_listening}")
		vc.handle_audio_processing(ctx)
	except discord.errors.ClientException as e:
		logging.error(f"Error joining voice channel: {e}")
		await  ctx.channel.send("I do not have permission to join that voice channel. Please check my permissions.")
	except discord.errors.InvalidData as e:
		logging.error(f"Error joining voice channel: {e}")
		await ctx.channel.send("Invalid voice channel ID. Please provide a valid ID.")
	except Exception as e:
		logging.error(f"Error joining voice channel: {e}")
		await ctx.channel.send("Unable to join the voice channel. Check my permissions.")
		raise e


# Get the current working directory
cwd = os.getcwd()

# Define the filename
filename = "config.json"

# Construct the full path to the config file
config_path = os.path.join(cwd, "config", filename)

# Now you can pass config_path to your load_config function
config = load_config(config_path)

async def check_and_leave_voice_channel (guild):
	voice_client = guild.voice_client
	if voice_client and voice_client.is_connected():
		if len(voice_client.channel.members) == 1:
			await voice_client.disconnect()

# Define a function to start the sharded bot tasks
import logging

async def start_bot():
	try:
		# Set up logging
		config = load_config(config_path)
		log_setup(config)
		# logging.info(f"Config file successfully loaded with content: {config}")
		history_file_name = config["Name"] + "_message_histories.json"
		user_message_histories = load_message_histories(history_file_name)
		intents = discord.Intents.all()
		bot = discord.AutoShardedClient(intents=intents)
				
		# Start worker tasks
		asyncio.create_task(worker(image_queue))
		asyncio.create_task(worker2(message_queue))
		
		@bot.event
		async def on_ready():
			logging.info('Logged in as {0.user}'.format(bot))
			
		@bot.event
		async def on_message(message):
			try:
				# Ignore messages from all bot users
				if message.author.bot:
					logging.info("Message is from a Bot.")
					return
				# Handle messages here
				await handle_messages(config, message, user_message_histories, history_file_name, bot, message_queue,image_queue)
				
			except Exception as e:
				logging.error("An error occurred during message handling: %s", str(e))

		@bot.event
		async def on_voice_state_update(member, before, after):
			voice_client = member.guild.voice_client
			if voice_client and voice_client.is_connected():
				if before.channel is None and after.channel is not None:
					# User joined the voice channel
					print(f"{member.name} joined the voice channel")
				elif before.channel is not None and after.channel is None:
					# User left the voice channel
					print(f"{member.name} left the voice channel")
					await check_and_leave_voice_channel(member.guild)

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
		if not isinstance(message.channel, (discord.Thread, discord.DMChannel)):
			if message.channel.id not in config["AllowedChannels"]:
				logging.info("Message not in an allowed channel. Ignoring message.")
				return
		
		logging.info(f"Received message: {message.content}")

		# Check if the message is from  a DM
		if message.channel.type == discord.ChannelType.private:
			if message.author.id in config["AllowedDMUsers"]:
				# Allow the message from an allowed DM user
				pass
			elif not config["AllowDMResponses"]:
				# Ignore the message if  DM responses are not allowed
				await message.channel.send("I Cannot talk here. Please try a regular Channel.")
				logging.info("Message is from a DM.")
				return

		if message.author.id in config["bot_user"]:
			logging.info(f"Message author is ignored. Ignoring message.")
			return	

		# Check if the message author is ignored
		if message.author.id in ignored_users and not config["bot_user"]:
			logging.info("Message is from an ignored user. Ignoring message.")
			await message.channel.send("I'm sorry, I cannot talk to you.")
			return
		
		# Check if any ignored words are present in the message
		if any(word.lower() in message.content.lower() for word in ignored_words):
			logging.info("Message contains ignored words. Ignoring message.")
			return
		
		# Check if the bot's name is mentioned
		bot_name = config["Name"]
		if not message.channel.type == discord.ChannelType.private:
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
			if message.channel.type == discord.ChannelType.private:
				await print_available_models(config,message)
			else: 
				await message.channel.send("I'm sorry, I cannot run that here.")
				return
		elif "load model" in message.content.lower():
			if message.channel.type == discord.ChannelType.private:
				try:
					load_msg = await message.channel.send(f"Loading model {message.content[10:]}.")
					await load_model(config,message)
					await load_msg.edit(content=load_msg.content + "\n" + "Model loaded successfully.")
					return
				except:
					await load_msg.edit(content= "I am sorry there is a problem with my camera.")
			else:
				await message.channel.send("I'm sorry, I cannot run that here.")
		elif "join_vc" in message.content.lower() or "join-vc" in message.content.lower():
			if message.author.voice:
				try:
					await join_voice_channel(message, message.author.voice.channel.id)
				except Exception as e:
					logging.error(f"Error occurred while joining voice channel: {e}")
					await message.channel.send("I was unable to join voice channel.")
					await message.channel.send("Please ensure that I have proper permissions to join voice channels.")
					return
				await message.add_reaction("👍")
				await message.channel.send("Joined voice channel")
			else:
				await message.channel.send('You need to be in a voice channel to use this command')
				return
		elif "leave_vc" in message.content.lower() or "leave-vc" in message.content.lower():
			if message.author.voice:
				voice_client = message.guild.voice_client
				if voice_client and voice_client.is_connected():
					await voice_client.disconnect()
					await message.add_reaction("👋")
					await message.channel.send("Left voice channel")
			else:			
				await message.channel.send('You need to be in a voice channel to use this command')
				return		
		elif message.guild:   # Check for guild and voice channel presence before sending to message_queue
			# Check for voice channel presence and bot and user connection
			voice_channel = None
			for channel in message.guild.channels:
				if channel.type == discord.ChannelType.voice:
					voice_channel = channel
					break
			if voice_channel and message.author.voice:
				async with channel.typing():
					await VoiceRecvClient.process_recognized_speech(message.content, message)
			else:
				# Check if the message was sent in a DM channel
				if message.channel.type == discord.ChannelType.private:
					message_queue.put_nowait(Task(config, message, user_message_histories, bot, history_file_name))
				else:
					message_queue.put_nowait(Task(config, message, user_message_histories, bot, history_file_name))
		else:
			message_queue.put_nowait(Task(config, message, user_message_histories, bot, history_file_name))
	except Exception as e:
		logging.error("An error occurred during bot startup: " + str(e))

# Run the bot
asyncio.run(start_bot())
