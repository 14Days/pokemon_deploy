from flask import Blueprint, jsonify, request
from app.model import signal_net

color_page = Blueprint('color', __name__, url_prefix='/color')


@color_page.route('', methods=['POST'])
def color():
    file = request.files['file']
    img_bytes = file.read()
    temp = signal_net.prediction(img_bytes)

    return jsonify({
        'color': temp
    })
