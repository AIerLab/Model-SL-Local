from flask import Flask, request, jsonify, abort
from typing import Callable

from helper import DATA


class Server:
    """
    A Server class that implements a simple Flask-based API for handling text processing,
    managing chats and roles, and receiving feedback.
    """

    def __init__(self, function: Callable[[str], str]):
        """
        Initialize the Server.

        :param function: The function to be used for processing text.
        """
        self.app = Flask(__name__)
        self.function = function

        # Register routes
        # Create chat, role; switch role, chat; delete role, chat; post text; clear, get history; continue chat, post feedback.
        self.app.route("/api/create/chat", methods=['POST'])(self.create_chat)
        self.app.route("/api/create/role", methods=['POST'])(self.create_role)
        self.app.route("/api/switch/role", methods=['POST'])(self.switch_role)
        self.app.route("/api/switch/chat", methods=['POST'])(self.switch_chat)
        self.app.route("/api/delete/role", methods=['DELETE'])(self.delete_role)
        self.app.route("/api/delete/chat", methods=['DELETE'])(self.delete_chat)
        self.app.route("/api/text", methods=['POST'])(self.post_text)
        self.app.route("/api/history", methods=['DELETE'])(self.clear_history)
        self.app.route("/api/history", methods=['POST'])(self.get_history)
        self.app.route("/api/chat/continue", methods=['POST'])(self.continue_chat)
        self.app.route("/api/feedback", methods=['POST'])(self.post_feedback)

    def create_chat(self):
        """
        Create a new chat.
        """
        chat_uid = request.json.get('chat_uid')
        if chat_uid:
            DATA['chats'][chat_uid] = []
            DATA['current_chat_uid'] = chat_uid
        return jsonify(message='Chat created'), 200

    def create_role(self):
        """
        Create a new role.
        """
        role_uid = request.json.get('role_uid')
        if role_uid:
            DATA['roles'].append(role_uid)
        return jsonify(message='Role created'), 200

    def switch_role(self):
        """
        Switch to a different role.
        """
        role_uid = request.json.get('role_uid')
        if role_uid in DATA["roles"]:
            DATA['current_role_uid'] = role_uid
            return jsonify(message='Role switched'), 200
        else:
            abort(401)  # Unauthorized

    def switch_chat(self):
        """
        Switch to a different chat.
        """
        chat_uid = request.json.get('chat_uid')
        if chat_uid in DATA['chats']:
            DATA['current_chat_uid'] = chat_uid
            return jsonify(message='Chat switched'), 200
        else:
            abort(401)  # Unauthorized

    def delete_role(self):
        """
        Delete a role.
        """
        role_uid = request.json.get('role_uid')
        if role_uid in DATA["roles"]:
            DATA['roles'].remove(role_uid)
            return jsonify(message='Role deleted'), 200
        else:
            abort(401)  # Unauthorized

    def delete_chat(self):
        """
        Delete a chat.
        """
        chat_uid = request.json.get('chat_uid')
        if chat_uid in DATA['chats']:
            del DATA['chats'][chat_uid]
            return jsonify(message='Chat deleted'), 200
        else:
            abort(401)  # Unauthorized

    def post_text(self):
        """
        Post text and get a response.
        """
        text = request.json.get('text')
        DATA["chats"][DATA["current_chat_uid"]].append(
            dict(
                sender="user",
                message=text
            )
        )
        query = DATA["test_blueprint"][DATA["current_role_uid"]] + text
        result = self.function(query)
        DATA["chats"][DATA["current_chat_uid"]].append(
            dict(
                sender=DATA["current_role_uid"],
                message=result
            )
        )
        return jsonify(result=result), 200

    def clear_history(self):
        """
        Clear chat history.
        """
        chat_uid = request.json.get('chat_uid')
        if chat_uid in DATA['chats']:
            DATA['chats'][chat_uid] = []
            return jsonify(message='History cleared'), 200
        else:
            abort(401)  # Unauthorized

    def get_history(self):
        """
        Get chat history.
        """
        chat_uid = request.json.get('chat_uid')
        if chat_uid in DATA['chats']:
            return jsonify(history=DATA['chats'][chat_uid]), 200
        else:
            abort(401)  # Unauthorized

    def continue_chat(self):
        """
        Continue a chat based on chat history.
        """
        # You need to implement this method based on your requirements
        raise NotImplementedError

    def post_feedback(self):
        """
        Post user feedback.
        """
        # You need to implement this method based on your requirements
        raise NotImplementedError

    def run(self, host='127.0.0.1', port=5000):
        """
        Run the Server.

        :param host: The hostname to listen on.
        :param port: The port of the webserver.
        """
        self.app.run(host=host, port=port)


if __name__ == '__main__':
    def process_func(data: str) -> str:
        """
        Example text processing function.

        :param data: The text to be processed.
        :return: The processed text.
        """
        return data  # In reality, this would be replaced by actual processing logic


    server = Server(process_func)
    server.run("localhost", 10086)
