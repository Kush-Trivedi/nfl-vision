import os
import io
import glob
import boto3
import base64
import requests
from main import *
from models.nfl_vision_playbook import NFLVisionPlaybook
from models.nfl_coach_assistant import NFLCoachAssistant
from flask import Flask, render_template, request, redirect, url_for

class NFLVisionApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = 'nfl_vision'
        self.app.jinja_env.auto_reload = True
        self.app.config["TEMPLATES_AUTO_RELOAD"] = True
        self.app.config['PROPAGATE_EXCEPTIONS'] = True
        self.coach_client = NFLCoachAssistant()
        self.playbook_client = NFLVisionPlaybook()
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'nfl-big-data-bowl-2024'
        self.image_paths = []
        self.text_paths = []
        self.total_cost = 0
        self.max_size = 20 * 1024 * 1024
        self.model_type = None

    @staticmethod
    def get_image_size(url):
        try:
            response = requests.head(url)
            return int(response.headers.get('Content-Length',0))
        except requests.RequestException as e:
            return None

    def read_text_from_s3(self, text_url):
        key = text_url.split('amazonaws.com/')[1] if 'amazonaws.com/' in text_url else text_url
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        return obj['Body'].read().decode('utf-8')

    def _encode_image_from_url(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
        else:
            return None

    def _encode_image(self, image_file):
        in_memory_file = io.BytesIO()
        image_file.save(in_memory_file)
        in_memory_file.seek(0)
        return base64.b64encode(in_memory_file.read()).decode('utf-8')

    def openai_error(self):
        return render_template('error.html')
    
    def index(self):
        self.image_paths.clear()
        self.text_paths.clear()
        try:
            if request.method == 'POST':
                game_id = int(request.form.get('game_id'))
                play_id = int(request.form.get('play_id'))

                matplotlib_visualizer.plot_game_in_matplotlib(game_id, play_id)

                frames_prefix = f"game_plays/{game_id}_{play_id}/"
                frame_urls = []
                gif_urls = []
                txt_urls = []
                mp4_urls = []

                for obj in self.s3_client.list_objects(Bucket=self.bucket_name, Prefix=frames_prefix)['Contents']:
                    if ".png" in obj['Key']:
                        frame_url = f"https://{self.bucket_name}.s3.amazonaws.com/{obj['Key']}"
                        frame_urls.append(frame_url)
                    if ".gif" in obj['Key']:
                        gif_url = f"https://{self.bucket_name}.s3.amazonaws.com/{obj['Key']}"
                        gif_urls.append(gif_url)
                    if ".txt" in obj['Key']:
                        txt_url = f"https://{self.bucket_name}.s3.amazonaws.com/{obj['Key']}"
                        txt_urls.append(txt_url)
                    if ".mp4" in obj['Key']:
                        mp4_url = f"https://{self.bucket_name}.s3.amazonaws.com/{obj['Key']}"
                        mp4_urls.append(mp4_url)

                sorted_frame_urls = sorted(frame_urls)
                sorted_text_url = sorted(txt_urls)
                selected_frames = sorted_frame_urls[::5] 
                selected_texts = sorted_text_url[::5]
                selected_frame_names = [url.split('/')[-1] for url in selected_frames]
                selected_text_names = [url.split('/')[-1] for url in selected_texts]
                frames = selected_frame_names + selected_text_names
                self.image_paths.extend(sorted_frame_urls)
                self.text_paths.extend(sorted_text_url)
                gif_path = gif_urls[0]
                mp4_path = mp4_urls[0]
                return render_template('chat.html', gif_path=gif_path, mp4_path=mp4_path)
            return render_template('index.html')
        except ValueError:
            return render_template('gameplayerror.html'), 400
        except Exception:
            return render_template('gameplayerror.html'), 500

    def openai_coach(self):
        try:
            question = request.form.get('input')
            selected_item = request.form.get('selected_item')
            dropdown_items = ["Frame {}".format(i + 1) for i in range(6, len(self.image_paths) - 5)]
            print(selected_item)
            
            if selected_item == 'undefined' or selected_item == 'null' or selected_item == 'openai':
                generated_text = self.coach_client.get_normal_response(question)
                return {'generated_text': generated_text, 'dropdown_items': dropdown_items}
            else:
                print("Satge here")
                index = int(selected_item.replace('Frame ', '')) - 1
                image_path = self.image_paths[index]
                text_path = self.text_paths[index]
                self.coach_client.add_image_path(image_path)
                text_content = self.read_text_from_s3(text_path)
                combined_text = question + "\n" + text_content if question else text_content
                generated_text = self.coach_client.process_input(combined_text)
            return {'generated_text': generated_text, 'dropdown_items': dropdown_items}
        except Exception:
            return redirect(url_for('openai_error')), 500

    def gemini_coach(self):
        try:
            question = request.form.get('input')
            selected_item = request.form.get('selected_item')
            dropdown_items = ["Frame {}".format(i + 1) for i in range(6, len(self.image_paths) - 5)]
            default_message = "You are a Expert NFL Coach: Respond to user queries promptly and accurately, offer helpful advice, acknowledge gratitude with courtesy, and address any dissatisfaction with understanding and additional assistance and nver say you are an ai based model say as a NFL Vision Coach.  If you are being asked anything other than American football, NFL and its related domains you will promptly say sorry i dont have information. You are strickly trained for american footaball or NFL and its related domains."
            default_response = self.coach_client.get_answer_from_geminie(default_message)

            if selected_item == 'undefined' or selected_item == 'null' or selected_item == 'gemini':
                default_message = "You are a Expert NFL Coach: Respond to user queries promptly and accurately, offer helpful advice, acknowledge gratitude with courtesy, and address any dissatisfaction with understanding and additional assistance and nver say you are an ai based model say as a NFL Vision Coach.  If you are being asked anything other than American football, NFL and its related domains you will promptly say sorry i dont have information. You are strickly trained for american footaball or NFL and its related domains."
                default_response = self.coach_client.get_answer_from_geminie(default_message)
                generated_text = self.coach_client.get_answer_from_geminie(question)
                return {'generated_text': generated_text, 'dropdown_items': dropdown_items}
            else:
                index = int(selected_item.replace('Frame ', '')) - 1  
                image_path = self.image_paths[index]
                text_path = self.text_paths[index]
                text_content = self.read_text_from_s3(text_path)
                combined_text = question + "\n" + text_content if question else text_content
                generated_text = self.coach_client.get_answer_from_geminie(combined_text)
                return {'generated_text': generated_text, 'dropdown_items': dropdown_items}
        except Exception:
            return redirect(url_for('openai_error')), 500

    def playbook(self):
        self.image_paths.clear()
        self.text_paths.clear()
        try:
            if request.method == 'POST':
                game_id = int(request.form.get('game_id'))
                play_id = int(request.form.get('play_id'))

                matplotlib_visualizer.plot_game_in_matplotlib(game_id, play_id)

                frames_prefix = f"game_plays/{game_id}_{play_id}/"
                frame_urls = []
                txt_urls = []
    
                for obj in self.s3_client.list_objects(Bucket=self.bucket_name, Prefix=frames_prefix)['Contents']:
                    if ".png" in obj['Key']:
                        frame_url = f"https://{self.bucket_name}.s3.amazonaws.com/{obj['Key']}"
                        frame_urls.append(frame_url)
                    if ".txt" in obj['Key']:
                        txt_url = f"https://{self.bucket_name}.s3.amazonaws.com/{obj['Key']}"
                        txt_urls.append(txt_url)
        
                sorted_frame_urls = sorted(frame_urls)
                sorted_text_url = sorted(txt_urls)

                selected_frames = sorted_frame_urls[5] 
                selected_texts = sorted_text_url[5]

                frame_name = os.path.basename(selected_frames)
                frames = [frame_name]

                self.image_paths.extend(sorted_frame_urls)
                self.text_paths.extend(sorted_text_url)
                return render_template("playbookchat.html", gif_path=selected_frames, frames=frames)
            return render_template('playbook.html')
        except ValueError:
            return render_template('gameplayerror.html'), 400
        except Exception:
            return render_template('gameplayerror.html'), 500

    def playbook_coach(self):
        question = request.form.get('input')
        uploaded_files = request.files.getlist("files[]") 
        
        if uploaded_files: 
            image_url = self.image_paths[5]
            text_path = self.text_paths[5]

            base64_images = []

            if image_url: 
                base64_string_from_url = self._encode_image_from_url(image_url)
                if base64_string_from_url:
                    base64_images.append(base64_string_from_url)

            for file in uploaded_files:
                if file: 
                    base64_string = self._encode_image(file)
                    base64_images.append(base64_string)

            self.playbook_client.add_image_path(base64_images)
            text_content = self.read_text_from_s3(text_path)
            split_point = text_content.find("Frame 6:")
            if split_point != -1: 
                extracted_text = text_content[:split_point]
            else:
                extracted_text = text_content
            combined_text = question + "\n" + extracted_text if question else extracted_text
            print(combined_text)
            generated_text = self.playbook_client.process_input(combined_text)
            return {'generated_text': generated_text}
        else:
            generated_text = self.playbook_client.get_normal_response(question)
            return {'generated_text': generated_text}


  
    def run(self, host='0.0.0.0', port=5000):
        self.app.route('/', methods=["GET", "POST"])(self.index)
        self.app.route('/openai', methods=["GET", "POST"])(self.openai_coach)
        self.app.route('/gemini', methods=["GET", "POST"])(self.gemini_coach)
        self.app.route('/playbook', methods=["GET", "POST"])(self.playbook)
        self.app.route('/playbookchat', methods=["GET", "POST"])(self.playbook_coach)
        self.app.route('/error')(self.openai_error)
        self.app.run(host=host, port=port)

nfl_vision_app = NFLVisionApp()

if __name__ == '__main__':
    nfl_vision_app.run()


