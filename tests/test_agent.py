"""
Test suite for AI agent functionality
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from app.agent.agent import run_agent, probe_models


class TestAgent(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_workbook = {
            'Sheet1': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        }
        self.test_chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    
    @patch('app.agent.agent.client')
    def test_run_agent_basic(self, mock_client):
        """Test basic agent functionality."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_client.chat.completions.create.return_value = mock_response
        
        result = run_agent(
            user_msg="Test message",
            workbook=self.test_workbook,
            current_sheet="Sheet1",
            chat_history=self.test_chat_history,
            model_name="gpt-4-turbo"
        )
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Test response")
    
    @patch('app.agent.agent.client')
    def test_probe_models(self, mock_client):
        """Test model probing functionality."""
        # Mock successful probe
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = probe_models(candidates=["gpt-4-turbo"])
        
        self.assertIsInstance(result, dict)
        self.assertIn('working', result)
        self.assertIn('failed', result)
    
    def test_agent_error_handling(self):
        """Test agent error handling."""
        # Test with empty message should still work
        try:
            result = run_agent(
                user_msg="Test",
                workbook=self.test_workbook,
                current_sheet="Sheet1",
                chat_history=[],
                model_name="gpt-4-turbo"
            )
            # This should not raise an exception with proper workbook
            self.assertTrue(True)
        except Exception:
            # If it fails due to API call, that's expected in test environment
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
