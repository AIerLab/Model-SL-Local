import socket
from typing import Dict, Any

from helper import NoneException


class SplitSocket:
    def __init__(self, host: str, port: int):
        # Create a TCP/IP socket

        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.client_socket.connect((host, port))
        except Exception as e:
            print(f"Could not connect to server: {e}")
            return

    def send_data(self, data: Dict[str, Any]) -> None:
        """
        Sends data to the server.

        Args:
            data: The data to send.
        """
        print(f"[CLIENT]: Sending intermediate data.")
        self._send_data(data["byte_data"])
        data["byte_data"] = self._receive_data()
        print(f"[CLIENT]: Received processed data.")
        return data

    def _send_data(self, data: bytes):
        """Sends raw data through the socket."""

        # try:
        #     self.client_socket.connect((self.host, self.port))
        # except Exception as e:
        #     print(f"Could not connect to server: {e}")
        #     return

        try:
            self.client_socket.sendall(data + b"EOF")
            # print(str(len(data)/4096))
        except Exception as e:
            print(f"Error sending data: {e}")

    def _receive_data(self) -> bytes:
        """Receives raw data from the socket."""

        # try:
        #     self.client_socket.connect((self.host, self.port))
        # except Exception as e:
        #     print(f"Could not connect to server: {e}")
        #     return

        data = []
        # try:
        while True:
            chunk = self.client_socket.recv(4096)
            if b"EOF" in chunk:
                data.append(chunk[:-3])
                break  # no more data
            # print(repr(chunk))
            data.append(chunk)
        # print("recvd data :" + str(len(data)))
        data = b"".join(data)
        # print(repr(data))
        if data == b"":
            raise NoneException("None received")
        return data
        # except Exception as e:
        #     print(f"Error receiving data: {e}")

    def close_connection(self):
        if self.client_socket:
            self.client_socket.close()
