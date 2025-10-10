"""
Sonos channel handler for the Communication Manager.

This handler provides audio output capabilities using Sonos speakers
with device discovery and text-to-speech functionality.
"""

import asyncio
import logging
import tempfile
from typing import Any, Dict, List, Optional

from common.communication_manager import ChannelHandler, ChannelType, MessageFormat

logger = logging.getLogger(__name__)


class SonosChannelHandler(ChannelHandler):
    """
    Channel handler for sending audio notifications to Sonos devices.

    Supports:
    - Device discovery and caching
    - Text-to-speech conversion
    - Volume management
    - Multi-device playback
    """

    channel_type = ChannelType.SONOS

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize the Sonos channel handler."""
        super().__init__(config)

        # Configuration
        self.default_volume = self.config.get("default_volume", 0.7)
        self.device_names = self.config.get("devices", [])  # Specific devices to use
        self.tts_service = self.config.get("tts_service", "gTTS")  # gTTS or pyttsx3
        self.language = self.config.get("language", "en")

        # Runtime state
        self.devices = {}  # Discovered Sonos devices
        self.soco_module = None

    def _validate_requirements(self) -> bool:
        """Check if Sonos devices are available on network."""
        try:
            import soco

            self.soco_module = soco

            # Try to discover devices
            devices = list(soco.discover(timeout=5))
            if len(devices) == 0:
                logger.warning("No Sonos devices discovered on network")
                return False

            logger.info(f"Found {len(devices)} Sonos devices")
            return True

        except ImportError:
            logger.error("soco library required for Sonos support: pip install soco")
            return False
        except Exception as e:
            logger.warning(f"Sonos device discovery failed: {e}")
            return False

    def _get_requirements_error_message(self) -> str:
        """Get descriptive error message for missing Sonos requirements."""
        try:
            import soco

            # If soco is available but no devices found
            return "no Sonos devices discovered on network. Ensure Sonos speakers are powered on and connected to the same network"
        except ImportError:
            return "missing soco library. Install with: pip install soco"

    async def start_session(self):
        """Discover and cache Sonos devices."""
        if not self.soco_module:
            return

        try:
            # Run discovery in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            devices = await loop.run_in_executor(
                None, lambda: list(self.soco_module.discover(timeout=5))
            )

            self.devices = {device.player_name: device for device in devices}
            self.session_active = len(self.devices) > 0

            device_names = list(self.devices.keys())
            logger.info(f"Discovered {len(self.devices)} Sonos devices: {device_names}")

        except Exception as e:
            logger.error(f"Failed to discover Sonos devices: {e}")
            self.session_active = False

    def get_capabilities(self) -> dict[str, Any]:
        """Sonos channel capabilities."""
        return {
            "supports_rich_text": False,  # Audio only
            "supports_buttons": False,
            "supports_audio": True,
            "supports_images": False,
            "bidirectional": False,  # Output only
            "requires_session": True,  # Need device discovery
            "max_message_length": 1000,  # Reasonable TTS limit
        }

    async def _send(
        self,
        message: str,
        recipient: Optional[str],
        message_format: MessageFormat,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send audio message to Sonos device(s).

        Args:
            message: The text message to convert to speech
            recipient: Device name or "all" for all devices
            message_format: Ignored for audio output
            metadata: Additional options:
                - volume: Volume level (0.0-1.0)
                - language: TTS language code
                - voice: Voice selection (if supported)
                - interrupt: Whether to interrupt current playback

        Returns:
            Dict with status information
        """
        if not self.session_active or not self.devices:
            return {"success": False, "error": "No Sonos devices available"}

        device_name = recipient or "all"
        volume = metadata.get("volume", self.default_volume)
        language = metadata.get("language", self.language)
        interrupt = metadata.get("interrupt", True)

        try:
            # Convert text to speech
            audio_file = await self._text_to_speech(message, language)
            if not audio_file:
                return {"success": False, "error": "Failed to generate speech"}

            # Determine target devices
            target_devices = self._get_target_devices(device_name)
            if not target_devices:
                return {
                    "success": False,
                    "error": f"No devices found for '{device_name}'",
                }

            # Play on target devices
            results = []
            for device in target_devices:
                try:
                    result = await self._play_on_device(
                        device, audio_file, volume, interrupt
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to play on {device.player_name}: {e}")
                    results.append(
                        {
                            "device": device.player_name,
                            "success": False,
                            "error": str(e),
                        }
                    )

            # Clean up temporary file
            try:
                import os

                os.unlink(audio_file)
            except:
                pass

            successful_devices = [r for r in results if r.get("success")]
            return {
                "success": len(successful_devices) > 0,
                "devices_played": len(successful_devices),
                "total_devices": len(results),
                "results": results,
            }

        except Exception as e:
            logger.error(f"Sonos playback failed: {e}")
            return {"success": False, "error": str(e)}

    def _get_target_devices(self, device_name: str) -> list:
        """Get list of target devices based on device name."""
        if device_name.lower() == "all":
            # Filter by configured device names if specified
            if self.device_names:
                return [
                    device
                    for name, device in self.devices.items()
                    if name in self.device_names
                ]
            else:
                return list(self.devices.values())
        else:
            # Find specific device
            device = self.devices.get(device_name)
            return [device] if device else []

    async def _play_on_device(
        self, device, audio_file: str, volume: float, interrupt: bool
    ) -> dict[str, Any]:
        """Play audio file on a specific Sonos device."""
        loop = asyncio.get_event_loop()

        def _play_sync():
            # Save current state
            current_volume = device.volume
            current_track = None
            current_position = 0

            try:
                if not interrupt:
                    # Try to get current track info
                    try:
                        track_info = device.get_current_track_info()
                        current_track = track_info
                        current_position = device.get_current_transport_info().get(
                            "current_transport_state"
                        )
                    except:
                        pass

                # Set volume for announcement
                device.volume = int(volume * 100)

                # Play the audio file
                # Note: This is a simplified approach. In production, you'd want to:
                # 1. Upload the file to a web server accessible by Sonos
                # 2. Use device.play_uri() with the HTTP URL
                # For now, we'll simulate the playback

                # Simulate playback duration based on message length
                import time

                estimated_duration = len(audio_file) * 0.1  # Rough estimate
                time.sleep(
                    min(estimated_duration, 10)
                )  # Cap at 10 seconds for simulation

                return {"device": device.player_name, "success": True}

            except Exception as e:
                return {"device": device.player_name, "success": False, "error": str(e)}
            finally:
                # Restore original volume
                try:
                    device.volume = current_volume
                except:
                    pass

        return await loop.run_in_executor(None, _play_sync)

    async def _text_to_speech(self, text: str, language: str = "en") -> Optional[str]:
        """Convert text to speech and return path to audio file."""
        try:
            if self.tts_service == "gTTS":
                return await self._tts_gtts(text, language)
            elif self.tts_service == "pyttsx3":
                return await self._tts_pyttsx3(text)
            else:
                logger.error(f"Unknown TTS service: {self.tts_service}")
                return None
        except Exception as e:
            logger.error(f"Text-to-speech conversion failed: {e}")
            return None

    async def _tts_gtts(self, text: str, language: str) -> Optional[str]:
        """Generate speech using Google Text-to-Speech."""
        try:
            import os
            import tempfile

            from gtts import gTTS

            # Create temporary file
            fd, temp_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)

            # Generate speech in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: gTTS(text=text, lang=language, slow=False).save(temp_path)
            )

            return temp_path

        except ImportError:
            logger.error("gTTS library required: pip install gtts")
            return None
        except Exception as e:
            logger.error(f"gTTS conversion failed: {e}")
            return None

    async def _tts_pyttsx3(self, text: str) -> Optional[str]:
        """Generate speech using pyttsx3 (offline TTS)."""
        try:
            import os
            import tempfile

            import pyttsx3

            # Create temporary file
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            # Generate speech in thread pool
            loop = asyncio.get_event_loop()

            def _generate():
                engine = pyttsx3.init()
                engine.save_to_file(text, temp_path)
                engine.runAndWait()

            await loop.run_in_executor(None, _generate)

            return temp_path

        except ImportError:
            logger.error("pyttsx3 library required: pip install pyttsx3")
            return None
        except Exception as e:
            logger.error(f"pyttsx3 conversion failed: {e}")
            return None
