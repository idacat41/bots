import discord
from discord.ext import commands
import wave
import os
import pyopus
import asyncio
import array
import speech_recognition as sr
from __future__ import annotations
import audioop
import logging
from collections import defaultdict
from rtp import SilencePacket
from concurrent.futures import Future as CFuture
from typing import Literal, Callable, Optional, Any, Final, Protocol, Awaitable, TypeVar
from discord import Member
from typing import TYPE_CHECKING, TypedDict
from opus import VoiceData
from types import MemberOrUser as User
import time
from sinks import AudioSink
from silero import silero_stt
class SRStopper(Protocol):
    def __call__(self, wait: bool = True, /) -> None:
        ...

SRProcessDataCB = Callable[[sr.Recognizer, sr.AudioData, User], Optional[str]]
SRTextCB = Callable[[User, str], Any]

log = logging.getLogger(__name__)
__all__ = [
    'SpeechRecognitionSink',
]
class _StreamData(TypedDict):
        stopper: Optional[SRStopper]
        recognizer: sr.Recognizer
        buffer: array.array[int]

class SpeechToTextHandler(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.model = silero_stt.SileroSTT(lang="en")
        self.opus_decoder = pyopus.Decoder(16000, 1)
        self._stream_data = defaultdict(lambda: _StreamData(stopper=None, recognizer=None, buffer=array.array('B')))

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        if not after.channel:
            return

        if member.guild.id not in self.bot.voice_states:
            self.bot.voice_states[member.guild.id] = {}

        self.bot.voice_states[member.guild.id][member.id] = after.channel.id

    @commands.command()
    async def join(self, ctx, channel: discord.VoiceChannel):
        if ctx.author.voice is None:
            return await ctx.send("You are not connected to a voice channel.")

        if ctx.voice_client is not None:
            return await ctx.voice_client.move_to(channel)

        await channel.connect()

    @commands.command()
    async def leave(self, ctx):
        if ctx.voice_client is None:
            return await ctx.send("I am not connected to a voice channel.")

        await ctx.voice_client.disconnect()

    @commands.command()
    async def start_listening(self, ctx):
        if ctx.voice_client is None:
            return await ctx.send("I am not connected to a voice channel.")

        def audio_callback(data, user):
            decoded_data = self.opus_decoder.decode(data, len(data))
            self._stream_data[user.id]['buffer'].extend(decoded_data)

            if not self._stream_data[user.id]['stopper']:
                self._stream_data[user.id]['stopper'] = self._stream_data[user.id]['recognizer'].listen_in_background(
                    (self._stream_data[user.id]['buffer']), self.background_listener(user), self.phrase_time_limit
                )

        ctx.voice_client.start_receiving_audio(audio_callback)

    @commands.command()
    async def stop_listening(self, ctx):
        if ctx.voice_client is None:
            return await ctx.send("I am not connected to a voice channel.")

        ctx.voice_client.stop_receiving_audio()

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return

        if message.guild.id not in self.bot.voice_states:
            return

        if message.author.id not in self.bot.voice_states[message.guild.id]:
            return

        channel = self.bot.get_channel(self.bot.voice_states[message.guild.id][message.author.id])

        if channel is None:
            return

        with wave.open("audio.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframesraw(self.recognizer.final_result())

        with open("audio.wav", "rb") as f:
            data = f.read()

        result = self.recognizer.recognize(data)

        if result is not None:
            await message.channel.send(result)

    def background_listener(self, user):
        def callback(_recognizer: sr.Recognizer, _audio: sr.AudioData):
            output = self.process_cb(_recognizer, _audio, user)
            if output is not None:
                self.text_cb(user, output)

        return callback

    def get_default_process_callback(self) -> SRProcessDataCB:
        def cb(recognizer: sr.Recognizer, audio: sr.AudioData, user: Optional[User]) -> Optional[str]:
            log.debug("Got %s, %s, %s", audio, audio.sample_rate, audio.sample_width)
            text: Optional[str] = None
            try:
                text = self.model.transcribe(audio.get_wav_data())
            except Exception as e:
                log.exception("Error transcribing audio: %s", e)

            return text

        return cb

    def get_default_text_callback(self) -> SRTextCB:
        def cb(user: Optional[User], text: Optional[str]) -> Any:
            log.info("%s said: %s", user.display_name if user else 'Someone', text)

        return cb

    @AudioSink.listener()
    def on_voice_member_disconnect(self, member: Member, ssrc: Optional[int]) -> None:
        self._drop(member.id)

    def cleanup(self) -> None:
        for user_id in tuple(self._stream_data.keys()):
            self._drop(user_id)

    def _drop(self, user_id: int) -> None:
        data = self._stream_data.pop(user_id)

        stopper = data.get('stopper')
        if stopper:
            stopper()

        buffer = data.get('buffer')
        if buffer:
            # arrays don't have a clear function
            del buffer[:]

    def _debug_audio_chunk(self, audio: sr.AudioData, filename: str = 'sound.wav') -> None:
        import io, wave, discord

        with io.BytesIO() as b:
            with wave.open(b, 'wb') as writer:
                writer.setframerate(48000)
                writer.setsampwidth(2)
                writer.setnchannels(2)
                writer.writeframes(audio.get_wav_data())

            b.seek(0)
            f = discord.File(b, filename)
            self._await(self.voice_client.channel.send(file=f))  # type: ignore

    class DiscordSRAudioSource(sr.AudioSource):
        little_endian: Final[bool] = True
        SAMPLE_RATE: Final[int] = 48_000
        SAMPLE_WIDTH: Final[int] = 2
        CHANNELS: Final[int] = 2
        CHUNK: Final[int] = 960

        def __init__(self, buffer: array.array[int]):
            self.buffer = buffer
            self._entered: bool = False

        @property
        def stream(self):
            return self

        def __enter__(self):
            if self._entered:
                log.warning('Already entered sr audio source')
            self._entered = True
            return self

        def __exit__(self, *exc) -> None:
            self._entered = False
            if any(exc):
                log.exception('Error closing sr audio source')

        def read(self, size: int) -> bytes:
            # TODO: make this timeout configurable
            for _ in range(10):
                if len(self.buffer) < size * self.CHANNELS:
                    time.sleep(0.1)
                else:
                    break
            else:
                if len(self.buffer) == 0:
                    return b''

            chunksize = size * self.CHANNELS
            audiochunk = self.buffer[:chunksize].tobytes()
            del self.buffer[: min(chunksize, len(audiochunk))]
            audiochunk = audioop.tomono(audiochunk, 2, 1, 1)
            return audiochunk

        def close(self) -> None:
            self.buffer.clear()
            
        