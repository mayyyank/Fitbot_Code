from flask import Flask, request, jsonify, render_template
import re
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Initialize Flask app
app = Flask(__name__)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Global variables for model and tokenizer
loaded_model = None
loaded_tokenizer = None

# Function to read and parse exercise routines from the updated text file
def load_exercise_routines(file_path='data/exercise_routines.txt'):
    routines = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regex pattern to match each combination and the associated exercise data
    pattern = re.compile(r'Combination:\s*(.*?)\s*\*\*Exercise Routine for.*?\*\*([\s\S]*?)(?=\nCombination:|$)')
    matches = pattern.findall(content)

    for match in matches:
        combination = match[0].strip()
        exercises_text = match[1].strip()

        exercises = []
        exercise_pattern = re.compile(r'Exercise Name:\s*(.*?)\s*Description:\s*(.*?)\s*How to Perform:\s*(.*?)\n', re.DOTALL)
        exercises_matches = exercise_pattern.findall(exercises_text)

        for exercise in exercises_matches:
            exercise_name = exercise[0].strip().replace('*', '')  # Remove asterisks from exercise name
            description = exercise[1].strip().replace('*', '') if exercise[1] else None  # Remove asterisks from description if available
            how_to = exercise[2].strip().replace('*', '')  # Remove asterisks from how-to instruction
            exercises.append({'exercise_name': exercise_name, 'description': description, 'how_to': how_to})

        routines[combination] = exercises

    return routines


# Load exercise routines at startup
exercise_routines = load_exercise_routines()

# Route for the landing (home) page
@app.route('/')
def home():
    return render_template('home.html')  # Home page

# Route for the chatbot page
@app.route('/chat')
def chat():
    return render_template('chat.html')  # Chatbot page

# Route for generating exercise routines
@app.route('/generate_exercises', methods=['POST'])
def generate_routine():
    data = request.get_json()

    skill_level = data.get('fitnessLevel')
    equipment = data.get('equipmentAccess')
    muscle_group = data.get('bodyPart')

    if not skill_level or not equipment or not muscle_group:
        return jsonify({'error': 'Missing required fields! Please select all options.'}), 400

    combination = f"{skill_level}_{equipment}_{muscle_group}"

    if combination in exercise_routines:
        routine = exercise_routines[combination]
        exercises = []
        for exercise in routine:
            # Only include description if it's not empty
            exercise_details = {
                'name': exercise['exercise_name'],
                'description': exercise['description'] if exercise['description'] else None,
                'how_to': exercise['how_to']
            }
            exercises.append(exercise_details)
        return jsonify({'exercises': exercises})  # Returning structured data
    else:
        return jsonify({'error': 'No exercise found for the given combination.'}), 404


# Route for handling user queries (GPT-2 model)
@app.route('/ask', methods=['POST'])
def ask():
    global loaded_model, loaded_tokenizer

    user_input = request.get_json().get('message')
    fitness_keywords = [
        "fitness", "exercise", "workout", "gym", "training", "strength", "cardio",
        "yoga", "stretching", "aerobics", "endurance", "weightlifting", "weights",
        "calisthenics", "bodybuilding", "running", "jogging", "cycling", "swimming",
        "HIIT", "crossfit", "pilates", "flexibility", "mobility", "balance",
        "nutrition", "diet", "calories", "protein", "carbs", "fats", "hydration",
        "recovery", "rest", "warm-up", "cool-down", "posture", "form", "health",
        "lifestyle", "goals", "fat loss", "weight loss", "muscle gain", "toning",
        "BMI", "metabolism", "core", "abs", "legs", "arms", "back", "chest",
        "shoulders", "glutes", "hamstrings", "quads", "heart rate", "progress",
        "routine", "plan", "program", "coach", "trainer", "self-care", "energy",
        "fitness tracker", "heart rate monitor", "stretch", "pull-up", "push-up",
        "plank", "squat", "deadlift", "bench press", "cardiovascular", "resistance",
        'gain','lean','muscle','mass'
    ]

    if "hello" in user_input and len(user_input)==5:
        response= "Hello! How can I help you?"
        return jsonify({'response': response})
    elif "bye" in user_input and len(user_input)==3:
        response= "Goodbye! Have a great day!"
        return jsonify({'response': response})
    elif not any(keyword in user_input.lower() for keyword in fitness_keywords):
        response = "I can only help with fitness-related questions."
        return jsonify({'response': response})

    if user_input:
        # Load model and tokenizer if not already loaded
        if loaded_model is None or loaded_tokenizer is None:
            model_path ="/home/sunbeam/Desktop/Project_ML/Python_Code/Fitbot_Code/model/chatbot_model_2"
            try:
                loaded_tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
                loaded_model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
                print("Model and tokenizer loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                return jsonify({'error': 'Model loading failed. Please contact support.'}), 500

        def prepare_input(tokenizer, input_text):
            prompt = f"Question: {input_text} Answer: "
            encoded_input = tokenizer(prompt, return_tensors='pt', padding=True,
                                      truncation=True, max_length=512)
            return encoded_input.input_ids.to(device), encoded_input.attention_mask.to(device)

        def generate_text(model, tokenizer, input_ids, attention_mask):
            model.eval()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True
                )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

        input_ids, attention_mask = prepare_input(loaded_tokenizer, user_input)
        generated_text = generate_text(loaded_model, loaded_tokenizer, input_ids, attention_mask)

        # Parsing the model's response to extract only the answer
        if 'Answer:' in generated_text:
            response = generated_text.split("Answer:")[1].strip()
            return jsonify({'response': response})
        else:
            response = "Sorry, I couldn't understand the question."
            return jsonify({'response': response})
    else:
        return jsonify({'error': 'No message received'}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run('0.0.0.0',port=5400,debug=True)
