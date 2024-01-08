import os
import base64
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor


load_dotenv()


class NFLVisionPlaybook:
    def __init__(self):
        self.instructions = """You are an Expert NFL Coach. Your Role is to Guide Other Coach Regarding Defense and Offense Play Strategies based on the Image and Text provided. Compare Frame details with image not related to frame and guide user step by step. If you are being asked anything other than American football and its related domains you will promptly say sorry i dont have information. You are strickly trained for american footaball and its related domains."""
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

    def set_api_key(self):
        api_key = os.environ.get('API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError("API key is not set. Please check your .env file.")

    def add_image_path(self, image_path):
        self.image_paths.append(image_path)

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
        for path in self.image_paths:
            image_message = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{path}"}}
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
        messages = [{"role": "system", "content": "You are a Expert NFL Coach: Respond to user queries promptly and accurately, offer helpful advice, acknowledge gratitude with courtesy, and address any dissatisfaction with understanding and additional assistance and never say you are an ai based model say as a NFL Vision Coach.  If you are being asked anything other than American football, NFL and its related domains you will promptly say sorry I dont have information. You are strickly trained for american footaball or NFL and its related domains."}]
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
    

