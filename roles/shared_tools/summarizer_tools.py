"""
Summarizer tools for StrandsAgent - converted from SummarizerAgent.

These tools replace the LangChain-based SummarizerAgent with @tool decorated functions
that can be used by the Universal Agent for text summarization functionality.
"""

import logging
import re
from datetime import datetime
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)


@tool
def summarize_text(
    text: str, max_length: int = 200, summary_type: str = "general"
) -> dict[str, Any]:
    """
    Summarize text content - converted from SummarizerAgent.

    This tool creates concise summaries of text content with configurable
    length and summary type.

    Args:
        text: The text content to summarize
        max_length: Maximum length of summary in words (default: 200)
        summary_type: Type of summary - "general", "bullet_points", "key_facts" (default: "general")

    Returns:
        Dict containing the summary and metadata
    """
    logger.info(
        f"Summarizing text: {len(text)} characters, max_length: {max_length}, type: {summary_type}"
    )

    if not text or not text.strip():
        return {
            "summary": "",
            "original_length": 0,
            "summary_length": 0,
            "compression_ratio": 0,
            "summary_type": summary_type,
            "status": "empty_input",
        }

    # Clean the text
    cleaned_text = re.sub(r"\s+", " ", text.strip())
    words = cleaned_text.split()

    # If text is already short enough, return as-is
    if len(words) <= max_length:
        return {
            "summary": cleaned_text,
            "original_length": len(words),
            "summary_length": len(words),
            "compression_ratio": 1.0,
            "summary_type": summary_type,
            "status": "no_compression_needed",
        }

    # Create summary based on type
    if summary_type == "bullet_points":
        summary = create_bullet_point_summary(cleaned_text, max_length)
    elif summary_type == "key_facts":
        summary = create_key_facts_summary(cleaned_text, max_length)
    else:  # general
        summary = create_general_summary(cleaned_text, max_length)

    summary_words = summary.split()
    compression_ratio = len(summary_words) / len(words)

    result = {
        "summary": summary,
        "original_length": len(words),
        "summary_length": len(summary_words),
        "compression_ratio": compression_ratio,
        "summary_type": summary_type,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
    }

    logger.info(
        f"Text summarized: {len(words)} -> {len(summary_words)} words ({compression_ratio:.2%} compression)"
    )
    return result


def create_general_summary(text: str, max_length: int) -> str:
    """Create a general summary by taking key sentences."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[: max_length * 5]  # Fallback to character truncation

    # Take first and last sentences, then middle ones
    summary_sentences = []
    if len(sentences) > 0:
        summary_sentences.append(sentences[0])
    if len(sentences) > 2:
        summary_sentences.append(sentences[-1])

    # Add middle sentences until we reach max_length
    current_words = len(" ".join(summary_sentences).split())
    for i in range(1, len(sentences) - 1):
        sentence_words = len(sentences[i].split())
        if current_words + sentence_words <= max_length:
            summary_sentences.insert(-1, sentences[i])
            current_words += sentence_words
        else:
            break

    return ". ".join(summary_sentences) + "."


def create_bullet_point_summary(text: str, max_length: int) -> str:
    """Create a bullet point summary."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    bullet_points = []
    current_words = 0

    for sentence in sentences[:5]:  # Max 5 bullet points
        words_in_sentence = len(sentence.split())
        if (
            current_words + words_in_sentence + 2 <= max_length
        ):  # +2 for bullet formatting
            bullet_points.append(f"• {sentence}")
            current_words += words_in_sentence + 2
        else:
            break

    return "\n".join(bullet_points)


def create_key_facts_summary(text: str, max_length: int) -> str:
    """Create a key facts summary."""
    # Look for numbers, dates, names, and important keywords
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    key_sentences = []
    current_words = 0

    # Prioritize sentences with numbers, dates, or key indicators
    priority_patterns = [
        r"\d+%",  # Percentages
        r"\$\d+",  # Money
        r"\d{4}",  # Years
        r"(increased|decreased|grew|fell|rose|dropped)",  # Change indicators
        r"(announced|launched|released|introduced)",  # Action words
    ]

    # Score sentences by importance
    scored_sentences = []
    for sentence in sentences:
        score = 0
        for pattern in priority_patterns:
            score += len(re.findall(pattern, sentence, re.IGNORECASE))
        scored_sentences.append((score, sentence))

    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    for _score, sentence in scored_sentences:
        words_in_sentence = len(sentence.split())
        if current_words + words_in_sentence <= max_length:
            key_sentences.append(sentence)
            current_words += words_in_sentence
        else:
            break

    return ". ".join(key_sentences) + "." if key_sentences else text[: max_length * 5]


@tool
def analyze_text_complexity(text: str) -> dict[str, Any]:
    """
    Analyze text complexity and readability.

    Args:
        text: Text to analyze

    Returns:
        Dict containing complexity metrics
    """
    logger.info(f"Analyzing text complexity: {len(text)} characters")

    if not text or not text.strip():
        return {
            "complexity": "unknown",
            "word_count": 0,
            "sentence_count": 0,
            "avg_words_per_sentence": 0,
            "status": "empty_input",
        }

    # Basic text analysis
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences)
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0

    # Simple complexity scoring
    if avg_words_per_sentence < 10:
        complexity = "simple"
    elif avg_words_per_sentence < 20:
        complexity = "moderate"
    else:
        complexity = "complex"

    # Count long words (>6 characters)
    long_words = [word for word in words if len(word) > 6]
    long_word_ratio = len(long_words) / word_count if word_count > 0 else 0

    result = {
        "complexity": complexity,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": round(avg_words_per_sentence, 2),
        "long_word_ratio": round(long_word_ratio, 3),
        "estimated_reading_time_minutes": round(word_count / 200, 1),  # Assume 200 WPM
        "status": "success",
    }

    logger.info(
        f"Text complexity analysis: {complexity} ({word_count} words, {sentence_count} sentences)"
    )
    return result


@tool
def extract_key_phrases(text: str, max_phrases: int = 10) -> dict[str, Any]:
    """
    Extract key phrases from text.

    Args:
        text: Text to extract phrases from
        max_phrases: Maximum number of phrases to extract

    Returns:
        Dict containing extracted key phrases
    """
    logger.info(f"Extracting key phrases from text: {len(text)} characters")

    if not text or not text.strip():
        return {"key_phrases": [], "phrase_count": 0, "status": "empty_input"}

    # Simple key phrase extraction
    # Remove common stop words and extract meaningful phrases
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "this",
        "that",
        "these",
        "those",
    }

    # Extract potential phrases (2-4 words)
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    phrases = []

    for i in range(len(words) - 1):
        # 2-word phrases
        if words[i] not in stop_words and words[i + 1] not in stop_words:
            phrase = f"{words[i]} {words[i+1]}"
            phrases.append(phrase)

        # 3-word phrases
        if i < len(words) - 2:
            if words[i] not in stop_words and words[i + 2] not in stop_words:
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                phrases.append(phrase)

    # Count phrase frequency
    phrase_counts = {}
    for phrase in phrases:
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    # Sort by frequency and take top phrases
    top_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[
        :max_phrases
    ]
    key_phrases = [phrase for phrase, count in top_phrases]

    result = {
        "key_phrases": key_phrases,
        "phrase_count": len(key_phrases),
        "total_phrases_found": len(phrase_counts),
        "timestamp": datetime.now().isoformat(),
        "status": "success",
    }

    logger.info(
        f"Key phrases extracted: {len(key_phrases)} phrases from {len(phrase_counts)} total"
    )
    return result


@tool
def compare_texts(text1: str, text2: str) -> dict[str, Any]:
    """
    Compare two texts for similarity and differences.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Dict containing comparison results
    """
    logger.info(f"Comparing texts: {len(text1)} vs {len(text2)} characters")

    if not text1 or not text2:
        return {
            "similarity_score": 0.0,
            "common_words": [],
            "unique_to_text1": [],
            "unique_to_text2": [],
            "status": "empty_input",
        }

    # Simple word-based comparison
    words1 = set(re.findall(r"\b[a-zA-Z]+\b", text1.lower()))
    words2 = set(re.findall(r"\b[a-zA-Z]+\b", text2.lower()))

    common_words = words1.intersection(words2)
    unique_to_text1 = words1 - words2
    unique_to_text2 = words2 - words1

    # Calculate similarity score (Jaccard similarity)
    union_words = words1.union(words2)
    similarity_score = len(common_words) / len(union_words) if union_words else 0.0

    result = {
        "similarity_score": round(similarity_score, 3),
        "common_words": sorted(common_words)[:20],  # Top 20 common words
        "unique_to_text1": sorted(unique_to_text1)[:10],  # Top 10 unique words
        "unique_to_text2": sorted(unique_to_text2)[:10],  # Top 10 unique words
        "text1_word_count": len(words1),
        "text2_word_count": len(words2),
        "common_word_count": len(common_words),
        "timestamp": datetime.now().isoformat(),
        "status": "success",
    }

    logger.info(f"Text comparison completed: {similarity_score:.1%} similarity")
    return result


# Tool registry for summarizer tools
SUMMARIZER_TOOLS = {
    "summarize_text": summarize_text,
    "analyze_text_complexity": analyze_text_complexity,
    "extract_key_phrases": extract_key_phrases,
    "compare_texts": compare_texts,
}


def get_summarizer_tools() -> dict[str, Any]:
    """Get all available summarizer tools."""
    return SUMMARIZER_TOOLS


def get_summarizer_tool_descriptions() -> dict[str, str]:
    """Get descriptions of all summarizer tools."""
    return {
        "summarize_text": "Create concise summaries of text content",
        "analyze_text_complexity": "Analyze text complexity and readability metrics",
        "extract_key_phrases": "Extract important phrases from text",
        "compare_texts": "Compare two texts for similarity and differences",
    }
