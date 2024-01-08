import os
import base64
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

class NFLCoachAssistant:
    def __init__(self):
        self.instructions = """
        As an "Expert NFL Coach," your role is focused on advising on play strategies related to American football, particularly in the areas of defense and offense. Adhere to these guidelines:

        1. Address inquiries specifically about offensive and defensive play strategies in American football.
        2. If asked to remember any part of the conversation, respond only with the word "Remembered" to acknowledge the request do not provide suggestion or anything else just reply "Remembered".
        3. If asked about topics outside of American football and its associated domains, respond with, "Sorry, I don't have information on that."
        4. Your expertise is exclusively in the realm of American football.
        5. You will be provided with briefs on offense and defense teams. Only address questions related to these teams. Politely decline queries about other teams and steer the conversation back to offense and defense.
        6. When responding, always include relevant player details: jersey number, name, and position.
        7. Utilize additional player information like height, weight, and position to suggest gameplay strategies.

        Your expertise is strictly limited to American football and the NFL.
        """
        self.temperature = 0.5
        self.max_tokens = 1024
        self.frequency_penalty = 0
        self.presence_penalty = 0.6
        self.max_context_questions = 10
        self.previous_questions_and_answers = []
        self.chunk_size = 2048
        self.encoding = tiktoken.encoding_for_model('gpt-4-vision-preview')
        self.image_paths = []
        self.client = None
        self.set_api_key()
        self.gemini_client = None
        self.gemini_model = None
        self.gemini_chat = None
        self.set_gemini_api_key()
        

    def set_api_key(self):
        api_key = os.environ.get('API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError("API key is not set. Please check your .env file.")

             
    def set_gemini_api_key(self):
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            self.gemini_client = genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            self.gemini_chat = self.gemini_model.start_chat()
        else:
            raise ValueError("API key is not set. Please check your .env file.")

    def add_image_path(self, image_path):
        self.image_paths.append(image_path)

    def add_image_paths(self, image_paths_list):
        self.image_paths.extend(image_paths_list)

    def split_into_chunks(self, text):
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start:start+self.chunk_size]
            token_count = len(self.encoding.encode(chunk))
            if token_count > self.max_tokens:
                chunk = text[start:start+self.chunk_size//2]
            elif start + self.chunk_size >= len(text):
                chunk = text[start:]
            chunks.append(chunk)
            start += len(chunk)
        return chunks

    def process_chunk(self, chunk):
        image_messages = []
        for url in self.image_paths:
            image_message = {"type": "image_url", "image_url": {"url": url}}
            image_messages.append(image_message)

        messages = [
            {"role": "system", "content": self.instructions},
        ]

        for question, answer in self.previous_questions_and_answers[-self.max_context_questions:]:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})

        messages.append({"role": "user", "content": [{"type": "text", "text": chunk}, *image_messages]})

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview", 
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        print("Have Used Vison Model")
        self.previous_questions_and_answers.append((chunk, response.choices[0].message.content))
        return response

    def ask_question(self, new_question):
        response = self.process_chunk(new_question)
        generated_text = response.choices[0].message.content
        return generated_text

    def process_large_input(self, large_input):
        chunks = self.split_into_chunks(large_input)
        responses = []

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]
            for future in futures:
                try:
                    response = future.result()
                    if response:
                        responses.append(response)
                except Exception as e:
                    print(f"Error processing chunk: {e}")

        return ' '.join([response.choices[0].message.content for response in responses])

    def process_input(self, input_text):
        if len(input_text) > self.chunk_size:
            return self.process_large_input(input_text)
        else:
            return self.ask_question(input_text)


    def get_normal_response(self, new_question):
        messages = [{"role": "system", "content": """
        You are designated as an "Expert NFL Coach." Your role involves:

        1. Promptly and accurately responding to user queries about American football, NFL, and related topics.
        2. If asked to remember any part of the conversation, respond only with the word "Remembered" to acknowledge the request do not provide suggestion or anything else just reply "Remembered".
        3. Providing helpful advice within your domain of expertise.
        4. Courteously acknowledging gratitude.
        5. Addressing dissatisfaction with understanding and offering additional assistance.
        6. Never disclosing that you are an AI-based model; instead, refer to yourself as an "NFL Vision Coach."
        7. If inquiries are unrelated to American football, NFL, or its related domains, respond with "Sorry, I don't have information on that."
                     
        Your expertise is strictly limited to American football and the NFL.
        """}]
        for question, answer in self.previous_questions_and_answers[-self.max_context_questions:]:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})

        messages.append({"role": "user", "content": new_question})
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106", 
            messages=messages
        )
        new_answer = completion.choices[0].message.content
        self.previous_questions_and_answers.append((new_question, new_answer))
        return new_answer

    def get_answer_from_geminie(self,question):
        response = self.gemini_chat.send_message(question)
        return response.text

