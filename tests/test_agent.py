"""
Test suite for AI agent functionality
"""
import unittest
from unittest.mock import patch, MagicMock
from app.agent.agent import run_agent, probe_models


class TestAgent(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_workbook = {
            'Sheet1': MagicMock()
        }
        self.test_chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    
    @patch('app.agent.agent.openai')
    def test_run_agent_basic(self, mock_openai):
        """Test basic agent functionality."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        result = run_agent(
            user_msg="Test message",
            workbook=self.test_workbook,
            current_sheet="Sheet1",
            chat_history=self.test_chat_history,
            model_name="gpt-4-turbo"
        )
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Test response")
    
    @patch('app.agent.agent.openai')
    def test_probe_models(self, mock_openai):
        """Test model probing functionality."""
        # Mock successful probe
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test"
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        result = probe_models()
        
        self.assertIsInstance(result, dict)
        self.assertIn('working', result)
        self.assertIn('failed', result)
    
    def test_agent_error_handling(self):
        """Test agent error handling."""
        # Test with invalid parameters
        with self.assertRaises(Exception):
            run_agent(
                user_msg="",  # Empty message
                workbook=None,  # Invalid workbook
                current_sheet="",
                chat_history=[],
                model_name=""
            )


if __name__ == '__main__':
    unittest.main()
