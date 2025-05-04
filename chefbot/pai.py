import os
from pyngrok import ngrok, conf
from flask import Flask, request, render_template_string
import torch, csv, requests
from diffusers import StableDiffusionPipeline

# Set tokens
os.environ['GROQ_API_KEY'] = "Your_Groq_key"
os.environ['HUGGINGFACE_TOKEN'] = "Your_HF_Token"
conf.get_default().auth_token = "Your_auth_Token"

# Flask app
app = Flask(__name__)

# Load image model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
).to("cuda" if torch.cuda.is_available() else "cpu")

# --- Your functions go here ---
def generate_recipe(ingredients):
    prompt = f"""
    You are a chef assistant. Create a human friendly dish name from: {', '.join(ingredients)}.
    Then write cooking steps.

    Format:
    Recipe Name: <name>
    Instructions:
    Step one...
    """
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    content = res.json()["choices"][0]["message"]["content"]
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    name, steps = "", []
    for l in lines:
        if l.lower().startswith("recipe name:"):
            name = l.split(":", 1)[1].strip()
        elif not l.lower().startswith("instructions:"):
            steps.append(l)
    return name, steps

def generate_image(prompt):
    image = pipe(prompt).images[0]
    path = "static/recipe_image.png"
    os.makedirs("static", exist_ok=True)
    image.save(path)
    return path

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "save" in request.form:
            recipe_name = request.form["recipe_name"]
            ingredients = request.form["ingredients"]
            steps = request.form.getlist("steps")
            with open('recipes.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([recipe_name, ingredients, ' | '.join(steps)])
            return f"<h2 class='text-success'>✅ Recipe Saved Successfully!</h2><a href='/'>Back</a>"

        try:
            ingredients = request.form["ingredients"].split(',')
            ingredients = [i.strip() for i in ingredients]
            recipe_name, steps = generate_recipe(ingredients)
            image_path = generate_image(recipe_name + ' on a plate')
            ingredients_str = ', '.join(ingredients)
        except Exception as e:
            return f"<h3>Error generating recipe: {e}</h3><a href='/'>Try again</a>"

        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ recipe_name }}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body class="bg-light text-center p-5">
            <div class="container">
                <h1 class="text-success">{{ recipe_name }}</h1>
                <img src="/{{ image_path }}" class="img-fluid rounded shadow" width="400"><br><br>
                <div class="text-start mx-auto" style="max-width: 600px;">
                    <h4>Instructions:</h4>
                    <ol>
                        {% for step in steps %}
                        <li>{{ step }}</li>
                        {% endfor %}
                    </ol>
                    <form method="post">
                        <input type="hidden" name="recipe_name" value="{{ recipe_name }}">
                        <input type="hidden" name="ingredients" value="{{ ingredients }}">
                        {% for step in steps %}
                        <input type="hidden" name="steps" value="{{ step }}">
                        {% endfor %}
                        <button name="save" value="1" class="btn btn-primary">💾 Save Recipe</button>
                    </form>
                    <a href="/" class="btn btn-secondary mt-2">🔁 Try Another</a>
                </div>
            </div>
        </body>
        </html>
        """, recipe_name=recipe_name, steps=steps, image_path=image_path, ingredients=ingredients_str)

    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Chef</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light text-center p-5">
        <div class="container">
            <h1 class="mb-4">🍳 AI Recipe Generator</h1>
            <form method="post" class="mx-auto" style="max-width: 500px;">
                <div class="mb-3">
                    <input name="ingredients" class="form-control" placeholder="Enter ingredients (comma separated)" required>
                </div>
                <button type="submit" class="btn btn-success">Generate Recipe</button>
            </form>
        </div>
    </body>
    </html>
    '''

public_url = ngrok.connect(5000)
print("Your app is live at:", public_url)
app.run(port=5000)

