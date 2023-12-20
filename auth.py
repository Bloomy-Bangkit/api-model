import os
import jwt
from dotenv import load_dotenv
from flask import jsonify
from flask_httpauth import HTTPTokenAuth
from http import HTTPStatus

load_dotenv()

auth = HTTPTokenAuth(scheme="Bearer")
SECRET_KEY = os.environ.get('SECRET_KEY')

if not SECRET_KEY:
    raise ValueError("SECRET_KEY is not set in the environment variables.")

@auth.error_handler
def unauthorized():
    return {
        "status": {
            "code": HTTPStatus.UNAUTHORIZED,
            "message": "Unauthorized"
        }
    }, HTTPStatus.UNAUTHORIZED

@auth.verify_token
def verify_token(token):
    try:
        result = jwt.decode(token, SECRET_KEY)
        return jsonify({
            'status': {
                'code': HTTPStatus.OK,
                'message': 'Berhasil request',
                'result': result
            }
        }), HTTPStatus.OK
    except jwt.ExpiredSignatureError:
        return {
            "status": {
                "code": HTTPStatus.UNAUTHORIZED,
                "message": "Token has expired"
            }
        }, HTTPStatus.UNAUTHORIZED
    except jwt.InvalidTokenError:
        return {
            "status": {
                "code": HTTPStatus.UNAUTHORIZED,
                "message": "Invalid token"
            }
        }, HTTPStatus.UNAUTHORIZED