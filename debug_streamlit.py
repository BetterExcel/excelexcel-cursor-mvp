#!/usr/bin/env python3
"""
Debug script to test streamlit app functionality
"""
import streamlit as st
import sys
import os
sys.path.insert(0, os.getcwd())

from app.agent.agent import run_agent
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Simple test to verify streamlit is working
st.title("Debug Chat Test")

# Initialize workbook
if "workbook" not in st.session_state:
    st.session_state.workbook = {
        'Sheet1': pd.DataFrame({
            'A': [1, 2, 3],
            'B': [10, 20, 30],
            'C': ['a', 'b', 'c']
        })
    }

if "current_sheet" not in st.session_state:
    st.session_state.current_sheet = 'Sheet1'

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Simple chat input
user_input = st.text_input("Type a message:")
if st.button("Send"):
    if user_input:
        st.write(f"You said: {user_input}")
        
        # Try to get AI response
        with st.spinner("Getting AI response..."):
            try:
                response = run_agent(
                    user_msg=user_input,
                    workbook=st.session_state.workbook,
                    current_sheet=st.session_state.current_sheet,
                    chat_history=st.session_state.chat_history,
                    model_name="gpt-4-turbo-2024-04-09"
                )
                st.success(f"AI replied: {response}")
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# Show chat history
if st.session_state.chat_history:
    st.subheader("Chat History:")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.write(f"👤 **You:** {msg['content']}")
        else:
            st.write(f"🤖 **AI:** {msg['content']}")
