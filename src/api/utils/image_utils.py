#!/usr/bin/python3
"""Images Upload Utility Module"""
from flask import request, jsonify, make_response
from werkzeug.utils import secure_filename
import secrets
import os
from marshmallow import EXCLUDE


if os.getenv("UPLOAD_FOLDER"):
    upload_folder = os.getenv("UPLOAD_FOLDER")
else:
    upload_folder = 'media'


def upload_image(request=None):
    """helper function to upload images to the server
    Args:
        request: the request object that contains an Image to upload
    """
    if 'image' not in request.files:
        responseObject = {'error': 'No image uploaded'}
        return make_response(jsonify(responseObject), 400)
    file = request.files['image']
    filename = secure_filename(file.filename)

    # Generate a random token for filename
    random_token = secrets.token_hex(8)
    new_filename = f'{random_token}_{filename}'

    # Check allowed image extensions
    allowed_extensions = ['jpeg', 'jpg', 'png', 'gif', 'JPG', 'PNG', 'GIF', 'BMP', 'TIFF']
    if filename.split('.')[-1].lower() not in allowed_extensions:
        responseObject = {'error': 'Unsupported file format'}
        return make_response(jsonify(responseObject), 400)
    
    try:
        # Save the image
        file.save(os.path.join(upload_folder, new_filename))
        return new_filename
    except Exception as e:
        responseObject = {'error': f'An error occured: {str(e)}'}
        return make_response(jsonify(responseObject), 500)

def delete_image(filename):
    """Delete an image resource from the server
    Arg:
        Filename: name of the file to be deleted
    """
    try:
        # Construct the filepath based on upload location
        filepath = os.path.join(upload_folder, filename)

        # Check if file exists
        if not os.path.exists(filepath):
            return make_response(jsonify({'error': 'File not found'}), 404)

        # Delete the image file
        os.remove(filepath)
    except Exception as e:
        responseObject = {'error': f'An error occured: {str(e)}'}
        return make_response(jsonify(responseObject), 500)
