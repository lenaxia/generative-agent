#!/usr/bin/env python3
"""
Test runner script for the Comprehensive Timer System.

This script runs all tests related to the timer system to validate the implementation.
"""

import argparse
import logging
import os
import sys
import unittest
from pathlib import Path

import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def run_tests(test_type="all", verbose=False):
    """Run timer system tests.

    Args:
        test_type: Type of tests to run ("unit", "integration", or "all")
        verbose: Whether to show verbose output

    Returns:
        True if all tests pass, False otherwise
    """
    # Determine test paths
    project_root = Path(__file__).parent

    if test_type == "unit" or test_type == "all":
        logger.info("Running timer unit tests...")
        unit_test_files = [
            "tests/unit/test_timer_manager.py",
            "tests/unit/test_timer_monitor.py",
            "tests/unit/test_communication_manager.py",
        ]

        # Run unit tests
        unit_args = ["-v"] if verbose else []
        for test_file in unit_test_files:
            test_path = project_root / test_file
            if test_path.exists():
                logger.info(f"Running {test_file}")
                result = pytest.main([str(test_path)] + unit_args)
                if result != 0:
                    logger.error(f"Unit tests failed in {test_file}")
                    return False
            else:
                logger.warning(f"Test file not found: {test_file}")

    if test_type == "integration" or test_type == "all":
        logger.info("Running timer integration tests...")
        integration_test_files = [
            "tests/integration/test_timer_communication_integration.py",
        ]

        # Run integration tests
        integration_args = ["-v"] if verbose else []
        for test_file in integration_test_files:
            test_path = project_root / test_file
            if test_path.exists():
                logger.info(f"Running {test_file}")
                result = pytest.main([str(test_path)] + integration_args)
                if result != 0:
                    logger.error(f"Integration tests failed in {test_file}")
                    return False
            else:
                logger.warning(f"Test file not found: {test_file}")

    logger.info("All timer tests completed successfully!")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run timer system tests")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )

    args = parser.parse_args()

    success = run_tests(args.type, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
