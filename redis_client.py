import redis
import json
import pickle

# Redis connection
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)  # decode_responses=False because we're storing binary data

def get_user_data(user_id):
    key = f"user:{user_id}:data"
    data = r.get(key)
    return json.loads(data) if data else []

def save_user_data(user_id, data):
    key = f"user:{user_id}:data"
    r.set(key, json.dumps(data))

# Optional: Save model if needed
def save_model(user_id, model):
    key = f"user:{user_id}:model"
    r.set(key, pickle.dumps(model))

def load_model(user_id):
    key = f"user:{user_id}:model"
    data = r.get(key)
    return pickle.loads(data) if data else None
