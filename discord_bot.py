import discord
from discord.ext import commands
from discord.ext.voice_recv import VoiceRecvClient  # Import VoiceRecvClient
import asyncio
import os
import logging
import queue
import time
from pathlib import Path
import subprocess
import speech_recognition as sr
import numpy as np
import wave
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from cfg import DISCORD_TOKEN
os.environ['DISCORD_TOKEN'] = DISCORD_TOKEN

class DiscordBot:
    def __init__(self, gemini_queue, last_responses, tts_completed):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        self.gemini_queue = gemini_queue
        self.last_responses = last_responses
        self.tts_completed = tts_completed
        self.voice_clients = {}
        self.recognizer = sr.Recognizer()
        
        self.setup_commands()
        
        @self.bot.event
        async def on_ready():
            logger.info(f'Bot connected as {self.bot.user}')

        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return

            await self.bot.process_commands(message)
            
            if self.bot.user.mentioned_in(message):
                content = message.content.replace(f'<@{self.bot.user.id}>', '').strip()
                if content:
                    await self.handle_mention(message, content)

    def setup_commands(self):
        @self.bot.command()
        async def join(ctx):
            if not ctx.author.voice:
                await ctx.send("You need to be in a voice channel!")
                return
                
            channel = ctx.author.voice.channel
            try:
                # Connect to the voice channel using VoiceRecvClient
                voice_client = await channel.connect(cls=VoiceRecvClient)
                self.voice_clients[ctx.guild.id] = voice_client
                await ctx.send("Connected to voice channel!")
                
                # Create and attach the VoiceSink
                sink = VoiceSink(self, ctx)
                voice_client.listen(sink)
                await ctx.send("🎙️ Ready to chat!")
            except Exception as e:
                logger.error(f"Error joining voice: {e}")
                await ctx.send("Failed to join voice channel.")

        @self.bot.command()
        async def leave(ctx):
            if ctx.guild.id in self.voice_clients:
                await self.voice_clients[ctx.guild.id].disconnect()
                del self.voice_clients[ctx.guild.id]
                await ctx.send("Left voice channel!")

    async def handle_mention(self, message, content):
        try:
            current_length = len(self.last_responses)
            self.gemini_queue.put(content)
            
            timeout = 120
            start_time = asyncio.get_event_loop().time()
            
            while True:
                if len(self.last_responses) > current_length:
                    response = self.last_responses[-1]
                    
                    while not self.tts_completed.is_set():
                        if asyncio.get_event_loop().time() - start_time > timeout:
                            await message.channel.send("TTS generation timed out.")
                            return
                        await asyncio.sleep(0.1)

                    # Find the latest inferred audio file
                    audio_files = glob.glob(os.path.join("audio", "out", "out_*.wav"))
                    if audio_files:
                        # Sort by modification time and get the most recent file
                        latest_audio_file = max(audio_files, key=os.path.getmtime)
                        
                        # Send the text response and the inferred audio file
                        await message.channel.send(
                            response,
                            file=discord.File(latest_audio_file)
                        )

                        # Play the audio in the voice channel
                        if message.guild.id in self.voice_clients:
                            voice_client = self.voice_clients[message.guild.id]
                            if not voice_client.is_playing():
                                voice_client.play(
                                    discord.FFmpegPCMAudio(latest_audio_file),
                                    after=lambda e: print(f'Player error: {e}' if e else 'Finished playing')
                                )
                    else:
                        await message.channel.send(response)
                    
                    self.tts_completed.clear()
                    break
                
                if asyncio.get_event_loop().time() - start_time > timeout:
                    await message.channel.send("Response timed out. Please try again.")
                    break
                    
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await message.channel.send("An error occurred while processing your request.")

    async def start(self):
        await self.bot.start(os.getenv('DISCORD_TOKEN'))

    async def close(self):
        if not self.bot.is_closed():
            await self.bot.close()


class VoiceSink(discord.ext.voice_recv.AudioSink):
    def __init__(self, bot, ctx):
        super().__init__()
        self.bot = bot
        self.ctx = ctx
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # Adjust sensitivity
        self.audio_buffer = []
        self.recording = False
        self.current_speaker = None
        self.output_dir = Path('recordings')
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('VoiceSink')
        self.wake_words = ["2b", "two bee", "hey 2b", "hey two bee", "toby", "hey toby", "darling"]  # Multiple wake words
        self.is_awake = False  # Track if the wake word has been detected

    def wants_opus(self) -> bool:
        return False

    def cleanup(self):
        """Clean up resources when the sink is being destroyed."""
        self.logger.info("Cleaning up VoiceSink")
        self.audio_buffer = []
        self.recording = False
        self.current_speaker = None
        self.is_awake = False  # Reset wake state

    def write(self, user, data: discord.ext.voice_recv.VoiceData):
        try:
            if user is None or data.pcm is None or not self.recording:
                return
            
            if user != self.current_speaker:
                return
                
            try:
                audio_np = np.frombuffer(data.pcm, dtype=np.int16)
                audio_stereo = audio_np.reshape(-1, 2)
                audio_mono = audio_stereo.mean(axis=1).astype(np.int16)
                
                self.audio_buffer.append(audio_mono.tobytes())
                
                max_amplitude = np.max(np.abs(audio_mono))
                if max_amplitude > 100:
                    self.logger.debug(f"Audio levels - Max amplitude: {max_amplitude}")
                    
            except Exception as e:
                self.logger.error(f"Error processing audio: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            self.logger.error(f"Error in write: {e}")
            import traceback
            traceback.print_exc()

    def save_and_process_audio(self):
        if not self.audio_buffer:
            return

        try:
            timestamp = int(time.time() * 1000)
            filename = self.output_dir / f"{self.current_speaker}_{timestamp}.wav"
            
            audio_data = b''.join(self.audio_buffer)
            
            with wave.open(str(filename), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(48000)
                wav_file.writeframes(audio_data)
            
            self.logger.info(f"Saved audio: {filename}")
            
            try:
                audio_data = sr.AudioData(audio_data, 48000, 2)
                text = self.recognizer.recognize_google(audio_data).lower()  # Convert to lowercase for case-insensitive comparison
                self.logger.info(f"Recognized: {text}")

                # Replace "toby" with "2B" in the recognized text
                text = text.replace("toby", "2B")

                # Get the username of the current speaker
                username = self.current_speaker.name if self.current_speaker else "Unknown User"

                # Check for any of the wake words (case-insensitive)
                if any(wake_word.lower() in text.lower() for wake_word in self.wake_words):
                    self.is_awake = True
                    self.logger.info(f"Wake word detected: {text}")
                    asyncio.run_coroutine_threadsafe(
                        self.ctx.send(f"🎤 **{username} said:** {text}"),
                        self.bot.bot.loop
                    )
                    # Process the input immediately if the wake word is detected
                    asyncio.run_coroutine_threadsafe(
                        self.bot.handle_mention(self.ctx.message, text),
                        self.bot.bot.loop
                    )
                    self.is_awake = False  # Reset wake state after processing
                    return  # Skip further processing

                # Only process audio if the wake word has been detected
                if self.is_awake:
                    asyncio.run_coroutine_threadsafe(
                        self.ctx.send(f"🎤 **{username} said:** {text}"),
                        self.bot.bot.loop
                    )
                    asyncio.run_coroutine_threadsafe(
                        self.bot.handle_mention(self.ctx.message, text),
                        self.bot.bot.loop
                    )
                    self.is_awake = False  # Reset wake state after processing
            except sr.UnknownValueError:
                self.logger.info("No speech recognized")
            except sr.RequestError as e:
                self.logger.error(f"Recognition error: {e}")
                
        except Exception as e:
            self.logger.error(f"Error saving/processing audio: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.audio_buffer = []

    @discord.ext.voice_recv.AudioSink.listener()
    def on_voice_member_speaking_start(self, member):
        self.logger.info(f"🗣️ {member} started speaking")
        if not self.recording or self.current_speaker != member:
            self.recording = True
            self.current_speaker = member
            self.audio_buffer = []

    @discord.ext.voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member):
        self.logger.info(f"🤐 {member} stopped speaking")
        if member == self.current_speaker:
            self.recording = False
            self.save_and_process_audio()
            self.current_speaker = None