from flask_httpauth import HTTPTokenAuth

auth = HTTPTokenAuth(scheme="Bearer")
SECRET_KEY = "Bloomy-APIModel!"

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
    return SECRET_KEY == token