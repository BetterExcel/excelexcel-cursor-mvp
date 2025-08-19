"""
Test suite for AI agent functionality
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from app.agent.agent import run_agent, probe_models, TOOLS
from app.services.workbook import new_workbook


class TestAgent(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Use proper workbook initialization
        self.test_workbook = new_workbook()
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
    def test_run_agent_with_tool_calls(self, mock_client):
        """Test agent functionality with tool calls."""
        # Mock tool call response
        mock_tool_call = MagicMock()
        mock_tool_call.id = "test_id"
        mock_tool_call.function.name = "set_cell"
        mock_tool_call.function.arguments = '{"sheet":"Sheet1","cell":"A1","value":"test"}'
        
        # First response with tool call
        mock_response_1 = MagicMock()
        mock_response_1.choices[0].message.content = None
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call]
        
        # Second response after tool execution
        mock_response_2 = MagicMock()
        mock_response_2.choices[0].message.content = "Tool executed successfully"
        mock_response_2.choices[0].message.tool_calls = None
        
        mock_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        result = run_agent(
            user_msg="Set A1 to test",
            workbook=self.test_workbook,
            current_sheet="Sheet1",
            chat_history=[],
            model_name="gpt-4-turbo"
        )
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Tool executed successfully")
        
        # Verify tool was actually executed
        df = self.test_workbook["Sheet1"]
        self.assertEqual(df.iloc[0, 0], "test")
    
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
    
    def test_tools_available(self):
        """Test that all expected tools are available."""
        expected_tools = [
            "set_cell", "get_cell", "apply_formula", "sort_sheet", 
            "filter_equals", "add_sheet", "make_chart", "export_sheet",
            "generate_sample_data", "create_csv_file", "save_current_sheet"
        ]
        
        tool_names = [tool["function"]["name"] for tool in TOOLS]
        
        for expected_tool in expected_tools:
            self.assertIn(expected_tool, tool_names)
    
    def test_agent_timeout_configuration(self):
        """Test that agent has timeout configured."""
        from app.agent.agent import client
        
        # Check that client has timeout configured
        self.assertIsNotNone(client.timeout)
        self.assertEqual(client.timeout, 30.0)
    
    def test_agent_error_handling(self):
        """Test agent error handling."""
        # Test with proper workbook should work (or fail gracefully with API error)
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
        except Exception as e:
            # If it fails due to API call, that's expected in test environment
            # But should not be a DataFrame-related error
            error_msg = str(e).lower()
            self.assertNotIn("no defined columns", error_msg)
            self.assertNotIn("empty dataframe", error_msg)
    
    def test_workbook_initialization(self):
        """Test that workbook is properly initialized."""
        workbook = new_workbook()
        
        # Should have Sheet1
        self.assertIn("Sheet1", workbook)
        
        # Sheet1 should have proper columns
        df = workbook["Sheet1"]
        self.assertGreater(len(df.columns), 0)
        self.assertEqual(list(df.columns), ['A', 'B', 'C', 'D', 'E'])
        
        # Should have proper dimensions
        self.assertEqual(len(df), 20)  # 20 rows
        self.assertEqual(len(df.columns), 5)  # 5 columns


if __name__ == '__main__':
    unittest.main()
