import os
import jwt
from dotenv import load_dotenv
from flask_httpauth import HTTPTokenAuth

load_dotenv()

auth = HTTPTokenAuth(scheme="Bearer")
SECRET_KEY = os.environ.get('SECRET_KEY')

@auth.error_handler
def unauthorized():
    return {
        "status": {
            "code": 401,
            "message": "Unauthorized"
        }
    }, 401

@auth.verify_token
def verify_token(token):
    try:
        result = jwt.decode(token, SECRET_KEY)
        return result
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None