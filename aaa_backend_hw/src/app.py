import logging
import io
from flask import Flask, request, jsonify
from models.plate_reader import PlateReader, InvalidImage
from image_provider_client import ImageProviderClient


app = Flask(__name__)
plate_reader = PlateReader.load_from_file('./model_weights/plate_reader_model.pth')


@app.route('/')
def hello():
    user = request.args.get('user', 'World')
    return f'<h1 style="color:red;"><center>Hello {user}!</center></h1>'

@app.route('/readPlateById/<int:image_id>', methods=['GET'])
def read_plate_by_id(image_id):
    try:
        img_bytes = ImageProviderClient.get_image(image_id)
        res = plate_reader.read_text(io.BytesIO(img_bytes))
        return {'plate_number': res}
    except InvalidImage:
        return {'error': 'Invalid image for recognition'}, 400
    except Exception as e:
        return {'error': str(e)}, 502


@app.route('/readPlatesBatch', methods=['POST'])
def read_plates_batch():
    data = request.get_json()
    if not data or 'ids' not in data or not isinstance(data['ids'], list):
        return {'error': 'Field "ids" (list of integers) is required'}, 400

    result = []
    for image_id in data['ids']:
        try:
            img_bytes = ImageProviderClient.get_image(image_id)
            plate = plate_reader.read_text(io.BytesIO(img_bytes))
            result.append({'id': image_id, 'plate_number': plate})
        except InvalidImage:
            result.append({'id': image_id, 'error': 'Invalid image'})
        except Exception as e:
            result.append({'id': image_id, 'error': str(e)})

    return jsonify(result)

if __name__ == '__main__':
    logging.basicConfig(
        format='[%(levelname)s] [%(asctime)s] %(message)s',
        level=logging.INFO,
    )

    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=8080, debug=True)
