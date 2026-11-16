"""Unit tests for broker factory selection"""

from pathlib import Path
from unittest.mock import patch

from aegra_api.services.broker import BrokerManager, _create_broker_manager
from aegra_api.services.redis_broker import RedisBrokerManager
from aegra_api.settings import EventStreamingSettings


class TestBrokerFactory:
    """Test _create_broker_manager factory function"""

    def test_returns_in_memory_broker_when_redis_disabled(self) -> None:
        with patch("aegra_api.services.broker.settings") as mock_settings:
            mock_settings.redis.REDIS_BROKER_ENABLED = False

            manager = _create_broker_manager()

            assert isinstance(manager, BrokerManager)

    def test_returns_redis_broker_when_redis_enabled(self) -> None:
        with patch("aegra_api.services.broker.settings") as mock_settings:
            mock_settings.redis.REDIS_BROKER_ENABLED = True

            manager = _create_broker_manager()

            assert isinstance(manager, RedisBrokerManager)


class TestReplayRetentionIsDocumented:
    """Both brokers read one setting; pin the documented default to it."""

    def test_documented_window_matches_the_setting(self) -> None:
        doc = Path(__file__).parents[5] / "docs" / "guides" / "streaming.mdx"
        default = EventStreamingSettings.model_fields["RESUMABLE_STREAM_TTL_SECONDS"].default
        assert f"`RESUMABLE_STREAM_TTL_SECONDS`, default {default} s" in doc.read_text(encoding="utf-8")
