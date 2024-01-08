# NFL Vision
A multimodal chatbot where a user can chat by providing their desired football query and get a decent suggestion not only but also with the precision of predictive modeling to predict the number of yards gained/lost.

- **Prerequisite**:
  - Python 3.10 or later
  - AWS Configured locally
  - Open AI API
  - Gemini API 

### Step 1: Clone Repository

Open your Terminal and clone the repository  

```bash
git clone https://github.com/Kush-Trivedi/nfl-vision.git
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
```

### Step 3: Install Packages

```bash
pip3 install -r requirements.txt
```

### Step 4: Activate a Virtual Environment

```bash
source venv/bin/activate
```

### Step 5: Define Open AI & Gemini API

```bash
touch .env
```

```bash
vim .env
```

replace API_KEY and GEMINI_API_KEY with your own

```bash
API_KEY = sk-xxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY = AIxxxxxxxxxxxxxxxxxx
```

### Step 6: Interact Locally with NFL Vision

```bash
python3 app.py
```

navigate to http://127.0.0.1:5000/
