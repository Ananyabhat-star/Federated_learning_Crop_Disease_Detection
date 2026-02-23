import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "super-secret-key-change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///federated_crop.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
