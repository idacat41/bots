import asyncio
import discord
import logging
import pydub
from discord import opus
import sys
sys.path.insert(0, './utils/')
from discord.ext.voice_recv import *

class SileroSTTSink:
    def __init__(self, text_cb):
        self.text_cb = text_cb
        self.buffer = b''
        self.opus_decoder = opus.Decoder()

    async def write(self, data):
        # Extract Opus audio data from the RTP packet using rtp module
        opus_data = reader.PacketDecryptor(data)
        if opus_data:
            # Decode Opus audio to PCM
            pcm_data = self.opus_decoder.decode(opus_data)
            # Perform STT
            stt_result = await self.perform_stt(pcm_data)
            if stt_result:
                await self.text_cb(stt_result)

    async def perform_stt(self, pcm_data):
        try:
            # Call Silero STT subprocess
            process = await asyncio.create_subprocess_exec(
                'silero_stt',
                '-i', 'pipe:0',  # Read from stdin
                '-t', 'wav',     # Input format
                '-o', 'pipe:1',  # Write to stdout
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE  # Capture stderr for error logging
            )

            # Write PCM data to Silero STT process stdin
            process.stdin.write(pcm_data)
            await process.stdin.drain()
            process.stdin.close()

            # Read STT result from stdout
            stt_result = await process.stdout.read()
            await process.wait()

            if process.returncode != 0:
                error_output = await process.stderr.read()
                logging.error(f"Error performing STT: {error_output.decode().strip()}")

            return stt_result.decode().strip()  # Decode bytes to string and strip whitespace
        except asyncio.CancelledError:
            raise  # Re-raise CancelledError to propagate cancellation
        except Exception as e:
            logging.error("Error performing STT:", exc_info=e)
            return None

    def decode_opus(self, chunk):
        # Convert Opus audio bytes to Pydub AudioSegment
        return pydub.AudioSegment(chunk, frame_rate=48000, sample_width=2, channels=2)


class SileroSTTClient(discord.VoiceClient):
    def __init__(self, client, channel, *args, **kwargs):
        super().__init__(client, channel, *args, **kwargs)
        self._recv_client = SileroSTTSink(self.on_stt_result)

    async def on_stt_result(self, text):
        # This method is called whenever STT result is received
        logging.info(f"Speech recognized: {text}")

    async def _do_run(self):
        # Override this method if you need custom behavior
        await super()._do_run()

    async def connect_to(self, *args, **kwargs):
        # Override this method to customize connection behavior if needed
        await super().connect(*args, **kwargs)

async def connect_to_voice_channel(client, channel):
    # Use SileroSTTClient when connecting to the voice channel
    try:
        voice_client = SileroSTTClient(client, channel)
        await voice_client.connect_to(channel)
        logging.info(f"Connected to voice channel: {channel}")
        return voice_client
    except Exception as e:
        logging.error(f"Error connecting to voice channel: {e}")
        return None