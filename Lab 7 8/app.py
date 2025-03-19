import requests
from flask import Flask, render_template, request

app = Flask(__name__)

API_KEY = "e33d3804f67244eea60b870fc609fc57"  # Replace with your API Key
SEARCH_URL = "https://api.spoonacular.com/recipes/complexSearch"
DETAIL_URL = "https://api.spoonacular.com/recipes/{}/information"

@app.route("/", methods=["GET", "POST"])
def search_recipes():
    recipes = []
    if request.method == "POST":
        query = request.form["recipe_name"]
        params = {
            "titleMatch": query,  
            "number": 5,  
            "apiKey": API_KEY,
            "addRecipeInformation": True  # Ensures we get correct image & details
        }
        response = requests.get(SEARCH_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            for item in data.get("results", []):
                recipe_id = item["id"]
                detail_response = requests.get(DETAIL_URL.format(recipe_id), params={"apiKey": API_KEY})

                if detail_response.status_code == 200:
                    details = detail_response.json()
                    recipes.append({
                        "title": details["title"],
                        "image": details.get("image", "https://via.placeholder.com/200"),  # Correct Image
                        "instructions": details.get("instructions", "No instructions available."),
                    })

    return render_template("index.html", recipes=recipes)

if __name__ == "__main__":
    app.run(debug=True)
