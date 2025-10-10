"""
Voice channel handler for the Communication Manager.

This handler provides bidirectional voice communication for Raspberry Pi
with wake word detection, speech recognition, and text-to-speech.
"""

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from common.communication_manager import ChannelHandler, ChannelType, MessageFormat

logger = logging.getLogger(__name__)


class VoiceChannelHandler(ChannelHandler):
    """
    Channel handler for voice communication on Raspberry Pi.

    Supports:
    - Wake word detection
    - Speech recognition (speech-to-text)
    - Text-to-speech output
    - Bidirectional voice interaction
    - Audio device management
    """

    channel_type = ChannelType.VOICE

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize the Voice channel handler."""
        super().__init__(config)

        # Configuration
        self.wake_word = self.config.get("wake_word", "hey assistant")
        self.language = self.config.get("language", "en-US")
        self.mic_device_index = self.config.get("mic_device_index")
        self.speaker_device_index = self.config.get("speaker_device_index")
        self.sensitivity = self.config.get("sensitivity", 0.5)
        self.timeout = self.config.get("timeout", 5.0)
        self.phrase_timeout = self.config.get("phrase_timeout", 1.0)

        # Voice engines
        self.stt_engine = self.config.get(
            "stt_engine", "google"
        )  # google, whisper, vosk
        self.tts_engine = self.config.get(
            "tts_engine", "pyttsx3"
        )  # pyttsx3, espeak, festival
        self.wake_word_engine = self.config.get(
            "wake_word_engine", "porcupine"
        )  # porcupine, snowboy

        # Runtime state
        self.recognizer = None
        self.microphone = None
        self.tts_engine_instance = None
        self.wake_word_detector = None
        self.listening_active = False
        self.pending_questions = {}
        self.audio_modules = {}

    def _validate_requirements(self) -> bool:
        """Validate voice processing requirements."""
        try:
            # Check for speech recognition
            import speech_recognition as sr

            self.audio_modules["sr"] = sr

            # Check for audio devices
            try:
                import pyaudio

                self.audio_modules["pyaudio"] = pyaudio
            except ImportError:
                logger.error("pyaudio required for voice support: pip install pyaudio")
                return False

            # Check for TTS engine
            if self.tts_engine == "pyttsx3":
                try:
                    import pyttsx3

                    self.audio_modules["pyttsx3"] = pyttsx3
                except ImportError:
                    logger.error("pyttsx3 required for TTS: pip install pyttsx3")
                    return False

            # Check for wake word detection (optional)
            if self.wake_word_engine == "porcupine":
                try:
                    import pvporcupine

                    self.audio_modules["pvporcupine"] = pvporcupine
                except ImportError:
                    logger.warning(
                        "pvporcupine not available, wake word detection disabled"
                    )

            return True

        except ImportError:
            logger.error(
                "speech_recognition required for voice support: pip install SpeechRecognition"
            )
            return False
        except Exception as e:
            logger.error(f"Voice requirements validation failed: {e}")
            return False

    async def start_session(self):
        """Initialize voice processing components."""
        try:
            # Initialize speech recognition
            sr = self.audio_modules["sr"]
            self.recognizer = sr.Recognizer()

            # Configure microphone
            if self.mic_device_index is not None:
                self.microphone = sr.Microphone(device_index=self.mic_device_index)
            else:
                self.microphone = sr.Microphone()

            # Adjust for ambient noise
            logger.info("Adjusting for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            # Initialize TTS engine
            if self.tts_engine == "pyttsx3":
                pyttsx3 = self.audio_modules["pyttsx3"]
                self.tts_engine_instance = pyttsx3.init()

                # Configure TTS settings
                voices = self.tts_engine_instance.getProperty("voices")
                if voices:
                    # Try to find a voice matching the language
                    for voice in voices:
                        if self.language.lower() in voice.id.lower():
                            self.tts_engine_instance.setProperty("voice", voice.id)
                            break

                # Set speech rate and volume
                self.tts_engine_instance.setProperty("rate", 150)  # Words per minute
                self.tts_engine_instance.setProperty("volume", 0.8)

            # Initialize wake word detection if available
            if "pvporcupine" in self.audio_modules:
                await self._initialize_wake_word_detection()

            self.session_active = True
            logger.info("Voice session initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize voice session: {e}")
            self.session_active = False

    async def _initialize_wake_word_detection(self):
        """Initialize Porcupine wake word detection."""
        try:
            pvporcupine = self.audio_modules["pvporcupine"]

            # Create Porcupine instance with built-in wake words
            # Note: In production, you'd use custom wake word models
            keywords = ["picovoice", "alexa", "hey google"]  # Built-in keywords

            self.wake_word_detector = pvporcupine.create(
                keywords=keywords, sensitivities=[self.sensitivity] * len(keywords)
            )

            logger.info(f"Wake word detection initialized with keywords: {keywords}")

        except Exception as e:
            logger.warning(f"Wake word detection initialization failed: {e}")
            self.wake_word_detector = None

    def get_capabilities(self) -> dict[str, Any]:
        """Voice channel capabilities."""
        return {
            "supports_rich_text": False,  # Voice is audio only
            "supports_buttons": False,
            "supports_audio": True,
            "supports_images": False,
            "bidirectional": True,  # Full bidirectional voice
            "requires_session": True,  # Need audio device setup
            "max_message_length": 500,  # Reasonable speech length
        }

    async def _background_session_loop(self):
        """Run voice processing in background thread."""
        if not self.session_active:
            return

        logger.info("Starting voice processing loop...")

        try:
            while self.session_active:
                if self.wake_word_detector:
                    # Listen for wake word
                    if await self._listen_for_wake_word():
                        # Wake word detected, listen for command
                        await self._process_voice_command()
                else:
                    # No wake word detection, continuous listening
                    await self._process_voice_command()

                await asyncio.sleep(0.1)  # Small delay to prevent busy loop

        except Exception as e:
            logger.error(f"Voice processing loop error: {e}")

    async def _listen_for_wake_word(self) -> bool:
        """Listen for wake word using Porcupine."""
        try:
            # This is a simplified implementation
            # In production, you'd run this in a separate thread with proper audio streaming
            loop = asyncio.get_event_loop()

            def _detect_wake_word():
                # Simulate wake word detection
                # In real implementation, this would process audio frames
                time.sleep(1)  # Simulate processing time
                return False  # No wake word detected in this simulation

            return await loop.run_in_executor(None, _detect_wake_word)

        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return False

    async def _process_voice_command(self):
        """Process incoming voice command."""
        try:
            # Listen for speech
            audio_text = await self._speech_to_text()

            if audio_text and audio_text.strip():
                logger.info(f"Voice command received: {audio_text}")

                # Send to main thread via queue
                if self.message_queue:
                    await self.message_queue.put(
                        {
                            "type": "incoming_message",
                            "user_id": "voice_user",
                            "channel_id": "voice",
                            "text": audio_text.strip(),
                            "timestamp": time.time(),
                        }
                    )

        except Exception as e:
            logger.error(f"Voice command processing error: {e}")

    async def _speech_to_text(self) -> Optional[str]:
        """Convert speech to text using configured STT engine."""
        try:
            loop = asyncio.get_event_loop()

            def _recognize_speech():
                try:
                    with self.microphone as source:
                        # Listen for audio with timeout
                        audio = self.recognizer.listen(
                            source,
                            timeout=self.timeout,
                            phrase_time_limit=self.phrase_timeout,
                        )

                    # Recognize speech using configured engine
                    if self.stt_engine == "google":
                        return self.recognizer.recognize_google(
                            audio, language=self.language
                        )
                    elif self.stt_engine == "whisper":
                        return self.recognizer.recognize_whisper(
                            audio, language=self.language
                        )
                    else:
                        return self.recognizer.recognize_google(
                            audio, language=self.language
                        )

                except Exception as e:
                    logger.debug(f"Speech recognition error: {e}")
                    return None

            return await loop.run_in_executor(None, _recognize_speech)

        except Exception as e:
            logger.error(f"Speech-to-text conversion failed: {e}")
            return None

    async def _send(
        self,
        message: str,
        recipient: Optional[str],
        message_format: MessageFormat,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send audio message via text-to-speech.

        Args:
            message: The text message to speak
            recipient: Ignored for voice output
            message_format: Ignored for voice output
            metadata: Additional options:
                - rate: Speech rate (words per minute)
                - volume: Volume level (0.0-1.0)
                - voice: Voice selection

        Returns:
            Dict with status information
        """
        if not self.session_active or not self.tts_engine_instance:
            return {"success": False, "error": "Voice session not active"}

        try:
            # Configure TTS settings from metadata
            rate = metadata.get("rate", 150)
            volume = metadata.get("volume", 0.8)
            voice = metadata.get("voice")

            # Apply settings
            self.tts_engine_instance.setProperty("rate", rate)
            self.tts_engine_instance.setProperty("volume", volume)

            if voice:
                self.tts_engine_instance.setProperty("voice", voice)

            # Speak the message
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._speak_text, message)

            return {"success": True, "message_length": len(message)}

        except Exception as e:
            logger.error(f"Voice output failed: {e}")
            return {"success": False, "error": str(e)}

    def _speak_text(self, text: str):
        """Speak text using TTS engine (synchronous)."""
        try:
            self.tts_engine_instance.say(text)
            self.tts_engine_instance.runAndWait()
        except Exception as e:
            logger.error(f"TTS speech failed: {e}")

    async def _ask_question_impl(
        self, question: str, options: list[str], timeout: int
    ) -> str:
        """Ask question via voice and wait for spoken response."""
        if not self.get_capabilities().get("bidirectional", False):
            raise NotImplementedError("Voice bidirectional communication not available")

        try:
            # Speak the question
            await self._send(question, None, MessageFormat.PLAIN_TEXT, {})

            # If options provided, speak them
            if options:
                options_text = "Your options are: " + ", ".join(options)
                await self._send(options_text, None, MessageFormat.PLAIN_TEXT, {})

            # Listen for response with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                response = await self._speech_to_text()
                if response and response.strip():
                    # If options provided, try to match response to options
                    if options:
                        response_lower = response.lower().strip()
                        for option in options:
                            if (
                                option.lower() in response_lower
                                or response_lower in option.lower()
                            ):
                                return option

                    return response.strip()

                await asyncio.sleep(0.5)  # Brief pause before trying again

            return "timeout"

        except Exception as e:
            logger.error(f"Voice question failed: {e}")
            raise

    async def stop_session(self):
        """Stop voice processing session."""
        self.session_active = False

        if self.wake_word_detector:
            try:
                self.wake_word_detector.delete()
            except:
                pass

        if self.tts_engine_instance:
            try:
                self.tts_engine_instance.stop()
            except:
                pass

        logger.info("Voice session stopped")
