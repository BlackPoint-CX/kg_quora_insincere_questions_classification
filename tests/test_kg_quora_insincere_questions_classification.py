#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `kg_quora_insincere_questions_classification` packages."""


import unittest
from click.testing import CliRunner

from kg_quora_insincere_questions_classification import kg_quora_insincere_questions_classification
from kg_quora_insincere_questions_classification import cli


class TestKg_quora_insincere_questions_classification(unittest.TestCase):
    """Tests for `kg_quora_insincere_questions_classification` packages."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'kg_quora_insincere_questions_classification.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
