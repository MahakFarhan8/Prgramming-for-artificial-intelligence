import nltk
from nltk.chat.util import Chat, reflections
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('vader_lexicon')

pairs = [
    [r"(?i).*hello.*|.*hi.*|.*hey.*",
     ["Hey there! Welcome to Foodie's Paradise! What would you like today?"]],
    
    [r"(?i).*menu.*",
     ["Our menu is packed with love!We have pizzas, burgers, pasta, biryani, BBQ, desserts, and much more!"]],
    
    [r"(?i).*burger.*|.*burgers.*",
     ["Our burgers are juicy and delicious! Options: Classic Beef, Chicken Supreme, Veggie Delight."]],
    
    [r"(?i).*pizza.*|.*pizzas.*",
     ["Hot and cheesy pizzas await you! Options: Margherita, Pepperoni, BBQ Chicken, Veggie Special."]],
    
    [r"(?i).*pasta.*",
     ["We serve creamy Alfredo, spicy Arrabiata, and classic Bolognese pasta!"]],
    
    [r"(?i).*biryani.*",
     ["Aromatic biryani for you! Options: Chicken Biryani, Mutton Biryani, and Veg Biryani."]],
    
    [r"(?i).*bbq.*",
     ["Smoky and tender BBQ dishes! Options: BBQ Wings, Ribs, and BBQ Platters."]],
    
    [r"(?i).*dessert.*|.*sweet.*",
     ["Dessert time!Options: Chocolate Lava Cake, Cheesecake, Ice Cream Sundae."]],
    
    [r"(?i).*drinks.*|.*beverage.*",
     ["Refreshing drinks available! Options: Lemonade, Mojito, Cold Coffee, Fresh Juices."]],
    
    [r"(?i).*booking.*|.*book.*|.*reservation.*",
     ["Sure! To book a table, please call us at +123-456-7890 or visit our website to reserve online!"]],
    
    [r"(?i).*thanks.*|.*thank you.*",
     ["You're welcome! Enjoy your meal!"]],
    
    [r"(?i).*bye.*|.*goodbye.*|.*see you.*",
     ["Goodbye! Come again!"]],
]

# Initialize chatbot and sentiment analyzer
chatbot = Chat(pairs, reflections)
sia = SentimentIntensityAnalyzer()

def get_chatbot_response(user_input):
    response = chatbot.respond(user_input)
    if response:
        return response
    else:
        return "I'm not sure about that. Try asking about food, booking, or offers!"

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return "Positive Sentence"
    elif sentiment_score['compound'] <= -0.05:
        return "Negative Sentence"
    else:
        return "Neutral Sentence"
