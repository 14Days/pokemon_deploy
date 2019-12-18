from flask import Blueprint, jsonify, request
from app.model import signal_net

color_page = Blueprint('color', __name__, url_prefix='/color')


@color_page.route('', methods=['POST'])
def color():
    file = request.files['file']
    img_bytes = file.read()

    try:
        temp = signal_net.prediction(img_bytes)
        return jsonify({
            'status': 'success',
            'tag_id': temp
        })
    except RuntimeError as e:
        return jsonify({
            'status': 'error',
            'err_msg': e.args[0]
        })
    except:
        return jsonify({
            'status': 'error',
            'err_msg': '未知错误'
        })
