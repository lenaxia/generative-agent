"""
Unit tests for the SonosChannelHandler.

Tests the Sonos audio output functionality including device discovery,
text-to-speech conversion, and audio playback.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from common.channel_handlers.sonos_handler import SonosChannelHandler
from common.communication_manager import ChannelType, MessageFormat


class TestSonosChannelHandler:
    """Test cases for the SonosChannelHandler."""

    def test_sonos_handler_initialization(self):
        """Test Sonos handler initialization with configuration."""
        config = {
            "default_volume": 0.8,
            "devices": ["Kitchen", "Living Room"],
            "tts_service": "gTTS",
            "language": "en-US",
        }

        handler = SonosChannelHandler(config)

        assert handler.channel_type == ChannelType.SONOS
        assert handler.default_volume == 0.8
        assert handler.device_names == ["Kitchen", "Living Room"]
        assert handler.tts_service == "gTTS"
        assert handler.language == "en-US"

    def test_sonos_handler_capabilities(self):
        """Test Sonos handler capabilities."""
        handler = SonosChannelHandler()
        capabilities = handler.get_capabilities()

        assert capabilities["supports_audio"] is True
        assert capabilities["bidirectional"] is False
        assert capabilities["requires_session"] is True
        assert capabilities["supports_rich_text"] is False
        assert capabilities["supports_buttons"] is False

    def test_validate_requirements_no_soco(self):
        """Test validation when soco library is not available."""
        handler = SonosChannelHandler()

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'soco'")
        ):
            assert handler._validate_requirements() is False

    def test_validate_requirements_no_devices(self):
        """Test validation when no Sonos devices are found."""
        handler = SonosChannelHandler()

        mock_soco = MagicMock()
        mock_soco.discover.return_value = []  # No devices found

        with patch("builtins.__import__", return_value=mock_soco):
            with patch.dict("sys.modules", {"soco": mock_soco}):
                assert handler._validate_requirements() is False

    def test_validate_requirements_success(self):
        """Test successful validation with devices found."""
        handler = SonosChannelHandler()

        # Mock Sonos device
        mock_device = MagicMock()
        mock_device.player_name = "Kitchen"

        mock_soco = MagicMock()
        mock_soco.discover.return_value = [mock_device]

        with patch("builtins.__import__", return_value=mock_soco):
            with patch.dict("sys.modules", {"soco": mock_soco}):
                assert handler._validate_requirements() is True

    @pytest.mark.asyncio
    async def test_start_session_success(self):
        """Test successful session start with device discovery."""
        handler = SonosChannelHandler()

        # Mock Sonos devices
        mock_device1 = MagicMock()
        mock_device1.player_name = "Kitchen"
        mock_device2 = MagicMock()
        mock_device2.player_name = "Living Room"

        mock_soco = MagicMock()
        mock_soco.discover.return_value = [mock_device1, mock_device2]
        handler.soco_module = mock_soco

        await handler.start_session()

        assert handler.session_active is True
        assert len(handler.devices) == 2
        assert "Kitchen" in handler.devices
        assert "Living Room" in handler.devices

    @pytest.mark.asyncio
    async def test_start_session_failure(self):
        """Test session start failure."""
        handler = SonosChannelHandler()

        mock_soco = MagicMock()
        mock_soco.discover.side_effect = Exception("Network error")
        handler.soco_module = mock_soco

        await handler.start_session()

        assert handler.session_active is False
        assert len(handler.devices) == 0

    def test_get_target_devices_all(self):
        """Test getting all devices."""
        handler = SonosChannelHandler()

        mock_device1 = MagicMock()
        mock_device1.player_name = "Kitchen"
        mock_device2 = MagicMock()
        mock_device2.player_name = "Living Room"

        handler.devices = {"Kitchen": mock_device1, "Living Room": mock_device2}

        devices = handler._get_target_devices("all")
        assert len(devices) == 2

    def test_get_target_devices_filtered(self):
        """Test getting filtered devices based on configuration."""
        handler = SonosChannelHandler({"devices": ["Kitchen"]})

        mock_device1 = MagicMock()
        mock_device1.player_name = "Kitchen"
        mock_device2 = MagicMock()
        mock_device2.player_name = "Living Room"

        handler.devices = {"Kitchen": mock_device1, "Living Room": mock_device2}

        devices = handler._get_target_devices("all")
        assert len(devices) == 1
        assert devices[0] == mock_device1

    def test_get_target_devices_specific(self):
        """Test getting specific device."""
        handler = SonosChannelHandler()

        mock_device = MagicMock()
        mock_device.player_name = "Kitchen"

        handler.devices = {"Kitchen": mock_device}

        devices = handler._get_target_devices("Kitchen")
        assert len(devices) == 1
        assert devices[0] == mock_device

    def test_get_target_devices_not_found(self):
        """Test getting device that doesn't exist."""
        handler = SonosChannelHandler()
        handler.devices = {}

        devices = handler._get_target_devices("NonExistent")
        assert len(devices) == 0

    @pytest.mark.asyncio
    async def test_tts_gtts_success(self):
        """Test successful gTTS text-to-speech conversion."""
        handler = SonosChannelHandler()

        mock_gtts = MagicMock()
        mock_gtts_instance = MagicMock()
        mock_gtts.return_value = mock_gtts_instance

        with patch("tempfile.mkstemp", return_value=(1, "/tmp/test.mp3")):
            with patch("os.close"):
                with patch.dict("sys.modules", {"gtts": MagicMock(gTTS=mock_gtts)}):
                    result = await handler._tts_gtts("Hello world", "en")

                    assert result == "/tmp/test.mp3"
                    mock_gtts.assert_called_once_with(
                        text="Hello world", lang="en", slow=False
                    )
                    mock_gtts_instance.save.assert_called_once_with("/tmp/test.mp3")

    @pytest.mark.asyncio
    async def test_tts_gtts_import_error(self):
        """Test gTTS when library is not available."""
        handler = SonosChannelHandler()

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'gtts'")
        ):
            result = await handler._tts_gtts("Hello world", "en")
            assert result is None

    @pytest.mark.asyncio
    async def test_tts_pyttsx3_success(self):
        """Test successful pyttsx3 text-to-speech conversion."""
        handler = SonosChannelHandler()

        mock_engine = MagicMock()
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        with patch("tempfile.mkstemp", return_value=(1, "/tmp/test.wav")):
            with patch("os.close"):
                with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
                    result = await handler._tts_pyttsx3("Hello world")

                    assert result == "/tmp/test.wav"
                    mock_pyttsx3.init.assert_called_once()
                    mock_engine.save_to_file.assert_called_once_with(
                        "Hello world", "/tmp/test.wav"
                    )
                    mock_engine.runAndWait.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_on_device_success(self):
        """Test successful audio playback on device."""
        handler = SonosChannelHandler()

        mock_device = MagicMock()
        mock_device.player_name = "Kitchen"
        mock_device.volume = 50

        result = await handler._play_on_device(mock_device, "/tmp/test.mp3", 0.7, True)

        assert result["success"] is True
        assert result["device"] == "Kitchen"

    @pytest.mark.asyncio
    async def test_send_no_devices(self):
        """Test sending when no devices are available."""
        handler = SonosChannelHandler()
        handler.session_active = False

        result = await handler._send("Hello world", None, MessageFormat.PLAIN_TEXT, {})

        assert result["success"] is False
        assert "No Sonos devices available" in result["error"]

    @pytest.mark.asyncio
    async def test_send_success(self):
        """Test successful message sending."""
        handler = SonosChannelHandler()
        handler.session_active = True

        # Mock device
        mock_device = MagicMock()
        mock_device.player_name = "Kitchen"
        handler.devices = {"Kitchen": mock_device}

        # Mock TTS and playback
        handler._text_to_speech = AsyncMock(return_value="/tmp/test.mp3")
        handler._play_on_device = AsyncMock(
            return_value={"device": "Kitchen", "success": True}
        )

        with patch("os.unlink"):  # Mock file cleanup
            result = await handler._send(
                "Hello world", "Kitchen", MessageFormat.PLAIN_TEXT, {"volume": 0.8}
            )

        assert result["success"] is True
        assert result["devices_played"] == 1
        assert result["total_devices"] == 1

        handler._text_to_speech.assert_called_once_with("Hello world", "en")
        handler._play_on_device.assert_called_once_with(
            mock_device, "/tmp/test.mp3", 0.8, True
        )

    @pytest.mark.asyncio
    async def test_send_tts_failure(self):
        """Test sending when TTS conversion fails."""
        handler = SonosChannelHandler()
        handler.session_active = True
        handler.devices = {"Kitchen": MagicMock()}

        # Mock TTS failure
        handler._text_to_speech = AsyncMock(return_value=None)

        result = await handler._send(
            "Hello world", "Kitchen", MessageFormat.PLAIN_TEXT, {}
        )

        assert result["success"] is False
        assert "Failed to generate speech" in result["error"]

    @pytest.mark.asyncio
    async def test_send_no_target_devices(self):
        """Test sending when no target devices are found."""
        handler = SonosChannelHandler()
        handler.session_active = True
        handler.devices = {"Kitchen": MagicMock()}

        # Mock TTS success but no target devices
        handler._text_to_speech = AsyncMock(return_value="/tmp/test.mp3")

        with patch("os.unlink"):
            result = await handler._send(
                "Hello world", "NonExistent", MessageFormat.PLAIN_TEXT, {}
            )

        assert result["success"] is False
        assert "No devices found for 'NonExistent'" in result["error"]

    @pytest.mark.asyncio
    async def test_send_multiple_devices(self):
        """Test sending to multiple devices."""
        handler = SonosChannelHandler()
        handler.session_active = True

        # Mock devices
        mock_device1 = MagicMock()
        mock_device1.player_name = "Kitchen"
        mock_device2 = MagicMock()
        mock_device2.player_name = "Living Room"

        handler.devices = {"Kitchen": mock_device1, "Living Room": mock_device2}

        # Mock TTS and playback
        handler._text_to_speech = AsyncMock(return_value="/tmp/test.mp3")
        handler._play_on_device = AsyncMock(
            return_value={"device": "test", "success": True}
        )

        with patch("os.unlink"):
            result = await handler._send(
                "Hello world", "all", MessageFormat.PLAIN_TEXT, {}
            )

        assert result["success"] is True
        assert result["devices_played"] == 2
        assert result["total_devices"] == 2

        # Verify TTS called once
        handler._text_to_speech.assert_called_once()

        # Verify playback called for each device
        assert handler._play_on_device.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])
