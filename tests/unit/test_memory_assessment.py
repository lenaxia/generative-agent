"""Tests for MemoryAssessment Pydantic model."""

import pytest
from pydantic import ValidationError

from common.memory_assessment import MemoryAssessment


class TestMemoryAssessmentValidation:
    """Test MemoryAssessment validation rules."""

    def test_valid_memory_assessment(self):
        """Test creating a valid MemoryAssessment."""
        assessment = MemoryAssessment(
            importance=0.8,
            summary="User scheduled team meeting for tomorrow at 2pm",
            tags=["meeting", "work", "team"],
            topics=["Team Meetings", "Schedule Management"],
            reasoning="Important work event with specific time and participants",
        )

        assert assessment.importance == 0.8
        assert assessment.summary == "User scheduled team meeting for tomorrow at 2pm"
        assert len(assessment.tags) == 3
        assert len(assessment.topics) == 2
        assert "Important work event" in assessment.reasoning

    def test_minimal_memory_assessment(self):
        """Test assessment with minimal required fields."""
        assessment = MemoryAssessment(
            importance=0.5,
            summary="Brief conversation",
            tags=["chat"],
            topics=[],
            reasoning="Simple exchange",
        )

        assert assessment.importance == 0.5
        assert len(assessment.tags) == 1
        assert len(assessment.topics) == 0

    def test_importance_bounds_validation(self):
        """Test importance must be between 0.0 and 1.0."""
        # Valid bounds
        MemoryAssessment(
            importance=0.0,
            summary="Test summary",
            tags=["test"],
            topics=[],
            reasoning="Test reasoning",
        )
        MemoryAssessment(
            importance=1.0,
            summary="Test summary",
            tags=["test"],
            topics=[],
            reasoning="Test reasoning",
        )

        # Invalid: too low
        with pytest.raises(ValidationError):
            MemoryAssessment(
                importance=-0.1,
                summary="Test",
                tags=["test"],
                topics=[],
                reasoning="Test",
            )

        # Invalid: too high
        with pytest.raises(ValidationError):
            MemoryAssessment(
                importance=1.1,
                summary="Test",
                tags=["test"],
                topics=[],
                reasoning="Test",
            )

    def test_summary_required(self):
        """Test summary field is required."""
        with pytest.raises(ValidationError):
            MemoryAssessment(importance=0.5, tags=["test"], topics=[], reasoning="Test")

    def test_summary_min_length(self):
        """Test summary must be at least 10 characters."""
        # Valid
        MemoryAssessment(
            importance=0.5,
            summary="Ten chars!",
            tags=["test"],
            topics=[],
            reasoning="Test",
        )

        # Invalid: too short
        with pytest.raises(ValidationError):
            MemoryAssessment(
                importance=0.5,
                summary="Short",
                tags=["test"],
                topics=[],
                reasoning="Test",
            )

    def test_tags_validation(self):
        """Test tags must have 1-10 items."""
        # Valid: 1 tag
        MemoryAssessment(
            importance=0.5,
            summary="Test summary",
            tags=["one"],
            topics=[],
            reasoning="Test",
        )

        # Valid: 10 tags
        MemoryAssessment(
            importance=0.5,
            summary="Test summary",
            tags=[f"tag{i}" for i in range(10)],
            topics=[],
            reasoning="Test",
        )

        # Invalid: no tags
        with pytest.raises(ValidationError):
            MemoryAssessment(
                importance=0.5,
                summary="Test summary",
                tags=[],
                topics=[],
                reasoning="Test",
            )

        # Invalid: too many tags
        with pytest.raises(ValidationError):
            MemoryAssessment(
                importance=0.5,
                summary="Test summary",
                tags=[f"tag{i}" for i in range(11)],
                topics=[],
                reasoning="Test",
            )

    def test_topics_optional(self):
        """Test topics field is optional."""
        # No topics
        assessment = MemoryAssessment(
            importance=0.5, summary="Test summary", tags=["test"], reasoning="Test"
        )
        assert assessment.topics == []

        # With topics
        assessment = MemoryAssessment(
            importance=0.5,
            summary="Test summary",
            tags=["test"],
            topics=["Topic 1"],
            reasoning="Test",
        )
        assert len(assessment.topics) == 1

    def test_topics_max_items(self):
        """Test topics limited to 5 items."""
        # Valid: 5 topics
        MemoryAssessment(
            importance=0.5,
            summary="Test summary",
            tags=["test"],
            topics=[f"Topic {i}" for i in range(5)],
            reasoning="Test",
        )

        # Invalid: too many topics
        with pytest.raises(ValidationError):
            MemoryAssessment(
                importance=0.5,
                summary="Test summary",
                tags=["test"],
                topics=[f"Topic {i}" for i in range(6)],
                reasoning="Test",
            )

    def test_reasoning_required(self):
        """Test reasoning field is required."""
        with pytest.raises(ValidationError):
            MemoryAssessment(
                importance=0.5, summary="Test summary", tags=["test"], topics=[]
            )

    def test_reasoning_max_length(self):
        """Test reasoning limited to 500 characters."""
        # Valid
        MemoryAssessment(
            importance=0.5,
            summary="Test summary",
            tags=["test"],
            topics=[],
            reasoning="A" * 500,
        )

        # Invalid: too long
        with pytest.raises(ValidationError):
            MemoryAssessment(
                importance=0.5,
                summary="Test summary",
                tags=["test"],
                topics=[],
                reasoning="A" * 501,
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            MemoryAssessment(
                importance=0.5,
                summary="Test",
                tags=["test"],
                topics=[],
                reasoning="Test",
                extra_field="not allowed",
            )
