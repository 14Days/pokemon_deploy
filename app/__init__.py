from flask import Flask
from app.controller.color import color_page


def create_flask_app() -> Flask:
    app = Flask(__name__)

    app.register_blueprint(color_page)

    return app
