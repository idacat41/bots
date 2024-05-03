# Initial package import statement
import sys
sys.path.insert(0, './utils/')
sys.path.insert(0, './utils/discord-ext-voice-recv')
from utils import *
from discord.ext.voice_recv.voice_client import *
from discord.ext.voice_recv.reader import *
from discord.ext.voice_recv.sinks import *
from discord.ext.voice_recv.video import *
from discord.ext.voice_recv.opus import *
from discord.ext.voice_recv.rtp import *

from discord.ext.voice_recv import (
    rtp as rtp,
    extras as extras,
)