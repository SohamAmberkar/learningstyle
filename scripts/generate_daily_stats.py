import json
import random
from datetime import datetime
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'daily_stats.json')

LEARNING_STYLES = ['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing']
MODELS = ['CatBoost', 'Neural Network', 'Decision Tree', 'Random Forest']

def generate_stats():
    today = datetime.now().strftime('%Y-%m-%d')
    
    stats = {
        "date": today,
        "timestamp": datetime.now().isoformat(),
        "active_users": random.randint(50, 500),
        "top_learning_style": random.choice(LEARNING_STYLES),
        "model_accuracy": round(random.uniform(0.85, 0.98), 4),
        "best_performing_model": random.choice(MODELS),
        "predictions_made": random.randint(100, 1000)
    }

    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Successfully generated stats for {today}: {stats}")
    except Exception as e:
        print(f"Error generating stats: {e}")
        exit(1)

if __name__ == "__main__":
    generate_stats()
