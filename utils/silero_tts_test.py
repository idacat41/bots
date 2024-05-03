import torch
from pprint import pprint
from omegaconf import OmegaConf
from IPython.display import Audio, display
import sounddevice as sd
import discord
from discord.ext import commands
import asyncio
import os
from io import BytesIO
import io
torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
							   'latest_silero_models.yml',
							   progress=False)
models = OmegaConf.load('latest_silero_models.yml')

# see latest avaiable models
available_languages = list(models.tts_models.keys())
print(f'Available languages {available_languages}')

for lang in available_languages:
	_models = list(models.tts_models.get(lang).keys())
	print(f'Available models for {lang}: {_models}')

import torch

language = 'en'
model_id = 'v3_en'
device = torch.device('cpu')

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
									 model='silero_tts',
									 language=language,
									 speaker=model_id)
model.to(device)

# sample_rate = 48000
# speaker = '"en_en_12"'
# put_accent=False
# put_yo=False
# example_text = 'This is a test'

# import num2words

# speakers =  model.speakers

# for speaker in speakers:
# 	speaker_id = speaker.split('_')
# 	if speaker_id[1].isdigit():
# 		speaker_id[1] = num2words.num2words(int(speaker_id[1]))
# 	example_text = f' This is speaker {speaker_id[0]} {speaker_id[1]}. Hello, How are you today?'
# 	audio = model.apply_tts(text=example_text,
# 							speaker=speaker,
# 							sample_rate=sample_rate,
# 							put_accent=put_accent,
# 							put_yo=put_yo)
# Initialize the Discord bot
intents=discord.Intents.all()
bot = commands.Bot(command_prefix='!',intents=intents)

# Define the command to generate and play audio
# Define the command to generate and play audio
@bot.command("audio")
async def tts(ctx):
	# Get the speaker from the list of available speakers
	speaker = model.speakers[0]  # Select the first speaker by default
	sample_rate = 48000
	speaker = "en_12"
	put_accent=False
	put_yo=False
	example_text  = 'This is a test'

	import num2words
	import torchaudio

	# speakers = model.speakers
	speakers = ["en_24","en_18", "en_4"]

	# Play the audio
	voice_client = await ctx.author.voice.channel.connect()
	for speaker in speakers:
		speaker_id = speaker.split('_')
		if speaker_id[1].isdigit():
			speaker_id[1] = num2words.num2words(int(speaker_id[1]))
		example_text = f' This is speaker {speaker_id[0]} {speaker_id[1]}. The five statements are: 1. A  healthy body is truly a guest-house for the soul. 2. One of the greatest diseases is to be nobody to anybody. 3. Your future  depends on many things, but mostly on you. 4. Neither a lofty degree of intelligence nor imagination nor both together go to the making of genius. Love, love, love, that is the soul of genius. 5. Long time no see!'
		audio = model.apply_tts(text=example_text,
								speaker=speaker,
								sample_rate=sample_rate,
								put_accent=put_accent,
								put_yo=put_yo)
		audio_tensor = audio.unsqueeze(0)
		audio_bytes_io = io.BytesIO()
		torchaudio.save(audio_bytes_io, src=audio_tensor, sample_rate=sample_rate, format="wav")
		audio_bytes = audio_bytes_io.getvalue() 
		audio_source = discord.FFmpegPCMAudio(io.BytesIO (audio_bytes), pipe=True)
		voice_client.play(audio_source)
		while voice_client.is_playing():
			await asyncio.sleep(1)
	await voice_client.disconnect()

# Run the bot
bot.run(os.environ["DiscordToken"])
