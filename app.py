from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import your_cnn_model  # Import your CNN model here
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/analyze', methods=['POST'])
def analyze_skin():
    if 'image' not in request.files:
        logging.error('No image uploaded')
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']

    if not image.content_type.startswith('image/'):
        logging.error('Uploaded file is not an image')
        return jsonify({'error': 'Uploaded file is not an image'}), 400

    try:
        result = your_cnn_model.analyze(image)  # Process the image with your model
    except Exception as e:
        logging.error(f'Error analyzing skin: {str(e)}')
        return jsonify({'error': str(e)}), 500  # Return error if analysis fails

    if not result['living']:
        return jsonify({'message': 'Non-human/non-living entity detected, no skin analysis available.'}), 200

    skin_metrics = result.get('skinMetrics', None)
    if skin_metrics is None:
        return jsonify({'error': 'Skin metrics missing in analysis result'}), 500

    # Check for required keys in skin_metrics
    required_keys = ['tone', 'acne_level', 'blackheads', 'dark_circles', 'skin_type', 'hair_quality', 'hydration_level']
    for key in required_keys:
        if key not in skin_metrics:
            return jsonify({'error': f'Missing key in skin metrics: {key}'}), 500

    # Return a detailed JSON response with all skin metrics
    return jsonify({
    'skinMetrics': {
        'skinTone': {
            'label': skin_metrics['tone'][0],  # Label
            'percentage': f"{skin_metrics['tone'][1]:.4f}"  # Format as percentage with 4 decimal places
        },
        'acneLevel': {
            'label': skin_metrics['acne_level'][0],  # Label
            'percentage': f"{skin_metrics['acne_level'][1]:.4f}"  # Format as percentage with 4 decimal places
        },
        'blackheads': {
            'label': skin_metrics['blackheads'][0],  # Label
            'percentage': f"{skin_metrics['blackheads'][1]:.4f}"  # Format as percentage with 4 decimal places
        },
        'darkCircles': {
            'label': skin_metrics['dark_circles'][0],  # Label
            'percentage': f"{skin_metrics['dark_circles'][1]:.4f}"  # Format as percentage with 4 decimal places
        },
        'skinType': {
            'label': skin_metrics['skin_type'][0],  # Label
            'percentage': f"{skin_metrics['skin_type'][1]:.4f}"  # Format as percentage with 4 decimal places
        },
        'hairQuality': {
            'label': skin_metrics['hair_quality'][0],  # Label
            'percentage': f"{skin_metrics['hair_quality'][1]:.4f}"  # Format as percentage with 4 decimal places
        },
        'hydrationLevel': {
            'label': skin_metrics['hydration_level'][0],  # Label
            'percentage': f"{skin_metrics['hydration_level'][1]:.4f}"  # Format as percentage with 4 decimal places
        }
    }
})


if __name__ == '__main__':
    app.run(debug=True)
