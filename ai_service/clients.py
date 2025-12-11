import pandas as pd
import time
from openai import OpenAI
from google import genai
from google.genai import types
from anthropic import Anthropic
import anthropic

class AiClient():
    def __init__(self):
        try:
            self.openai_api_client = OpenAI(api_key="sk-")
            self.gemini_api_client = genai.Client(api_key="AI")
            self.anthropic_api_client = Anthropic(api_key="sk-")
        except Exception as e:
            print(f"keys: {e}")

    def apenai_call(self, messages, model_name):
        start_time = time.time()
        try:
            response = self.openai_api_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0
            )
            end_time = time.time()
            
            pred = response.choices[0].message.content.strip()
            elapsed_time = end_time - start_time
            
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            return pred, elapsed_time, prompt_tokens, completion_tokens
            
        except Exception as e:
            print(f"API Error: {e}")
            return "error", 0, 0, 0

    def gemini_call(self, messages_list, model_name="gemini-2.0-flash"):
        start_time = time.time()
        
        system_instruction = ""
        user_content = ""
        
        if isinstance(messages_list, list):
            for msg in messages_list:
                role = msg.get("role")
                content = msg.get("content", "")
                
                if role == "system":
                    system_instruction += content + "\n"
                elif role == "user":
                    user_content += content + "\n"
        else:
            user_content = str(messages_list)

        try:
            response = self.gemini_api_client.models.generate_content(
                model=model_name,
                contents=user_content.strip(), 
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction.strip(), 
                    temperature=0.0,
                )
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            pred = response.text.strip() if response.text else "error"
        
            usage = response.usage_metadata
            prompt_tokens = usage.prompt_token_count if usage else 0
            completion_tokens = usage.candidates_token_count if usage else 0
            
            return pred, elapsed_time, prompt_tokens, completion_tokens
            
        except Exception as e:
            print(f"Error: {e}")
            return "error", 0, 0, 0
        
    def anthropic_call(self, messages_list, model_name="claude-3-5-haiku-20241022"):
        system_prompt = anthropic.NOT_GIVEN 
        clean_messages = []

        for m in messages_list:
            role = m.get("role")
            content = m.get("content", "")
            
            if role == "system":
                system_prompt = content
            else:
                clean_messages.append({"role": role, "content": content})

        start_time = time.time()
        
        try:
            response = self.anthropic_api_client.messages.create(
                model=model_name,
                max_tokens=1000,     
                temperature=0.0,
                system=system_prompt,  
                messages=clean_messages  
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            pred = response.content[0].text

            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens

            return pred, elapsed_time, prompt_tokens, completion_tokens

        except Exception as e:
            print(f"Error: {e}")
            return "error", 0, 0, 0
        
