# Global variable
# This dictionary stores all the roles, chats, current role and chat UID.
DATA = dict(
    roles=["test", "test1"],
    chats={
        "test": [
            dict(
                sender="user",
                message="this is a test input"
            ),
            dict(
                sender="test",
                message="this is a test output from user test."
            )
        ],
        "test1": []
    },
    current_role_uid="test",
    current_chat_uid="test",
    test_blueprint={
        "test": "Now you have to act like a business man. ",
        "test1": "Now you have to act like a cat girl, add nya to reply. "
    }
)
