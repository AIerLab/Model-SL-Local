# Global variable
# This dictionary stores all the roles, chats, current role and chat UID.
DATA = dict(
    roles=["test"],
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
        ]
    },
    current_role_uid="test",
    current_chat_uid="test"
)
