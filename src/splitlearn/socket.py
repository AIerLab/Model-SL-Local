import socket
import time
from typing import Dict, Any
import hashlib
from helper import NoneException

# Constants
ACK = b"ACK"
NACK = b"NACK"
EOF = b"EOF"
RETRY_LIMIT = 1000


class SplitSocket:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.initialize_socket()

    def initialize_socket(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((self.host, self.port))
            # Setting a timeout of 2 seconds
            self.client_socket.settimeout(10)
        except Exception as e:
            print(f"Error with socket operation: {e}")
            self.close_connection()
            time.sleep(1.0)
            self.initialize_socket()

    def _compute_checksum(self, data: bytes) -> bytes:
        """Compute and return the MD5 checksum of the given data."""
        return hashlib.md5(data).digest()

    def send_data(self, payload: Dict[str, Any]) -> None:
        """Send data and receive a processed response."""
        print(f"[CLIENT]: Sending intermediate data.")
        self._send_data(payload["byte_data"])
        payload["byte_data"] = self._receive_data()
        print(f"[CLIENT]: Received processed data.")
        return payload

    def receive_data(self, payload: Dict[str, Any]) -> None:
        """Receive data and store in the provided dictionary."""
        payload["byte_data"] = self._receive_data()
        print(f"[CLIENT]: Received processed data.")
        return payload

    def _send_data(self, data: bytes):
        """Send data with checksum and handle ACK/NACK."""
        retries = 0
        while retries < RETRY_LIMIT:
            try:
                checksum = self._compute_checksum(data)
                self.client_socket.sendall(checksum + data + EOF)
                ack = self.client_socket.recv(len(ACK))
                if ack == ACK:
                    break
                else:
                    print("Checksum mismatch while sending data. Retrying...")
                    retries += 1
            except socket.timeout:
                print("Timeout while waiting for ACK. Retrying...")
                retries += 1
            except socket.error as e:
                print(f"Error sending data: {e}")
                retries += 1

        if retries == RETRY_LIMIT:
            print("Failed to send data after multiple retries.")

    def _receive_data(self) -> bytes:
        """Receive data, validate its checksum, and return the data."""
        data = bytearray()
        retries = 0
        while retries < RETRY_LIMIT:
            try:
                while True:
                    chunk = self.client_socket.recv(4096)
                    if EOF in chunk:
                        data.extend(chunk[:-len(EOF)])
                        break
                    data.extend(chunk)

                received_checksum = data[:16]
                actual_data = data[16:]

                if self._compute_checksum(actual_data) == received_checksum:
                    self.client_socket.sendall(ACK)
                    if actual_data == b"":
                        raise NoneException("None received")
                    return actual_data
                else:
                    self.client_socket.sendall(NACK)
                    print("Checksum mismatch while receiving data. Retrying...")
                    retries += 1
            except socket.timeout:
                print("Socket timeout. Reinitializing...")
                self.close_connection()
                self.initialize_socket()
                return self._receive_data()  # Optionally retry receiving
            except socket.error as e:
                print(f"Error receiving data: {e}")
                retries += 1

        if retries == RETRY_LIMIT:
            print("Failed to receive data after multiple retries.")

        if data == b"":
            raise NoneException("None received")
        return data

    def close_connection(self):
        """Close the client socket if it exists."""
        if self.client_socket:
            self.client_socket.close()

# import socket
# import time
# from typing import Dict, Any
# from helper import NoneException
# from .base_socket import BaseSocket, ACK, NACK, EOF, RETRY_LIMIT


# class SplitSocket(BaseSocket):
#     def __init__(self, host: str, port: int):
#         super().__init__(host, port)
#         self.initialize_socket()

#     def initialize_socket(self):
#         self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         try:
#             self.client_socket.connect((self.host, self.port))
#             # Setting a timeout of 2 seconds
#             self.client_socket.settimeout(10)
#         except Exception as e:
#             print(f"Error with socket operation: {e}")
#             self.close_connection()
#             time.sleep(1.0)
#             self.initialize_socket()

#     def send_data(self, payload: Dict[str, Any]) -> None:
#         """Send data and receive a processed response."""
#         print(f"[CLIENT]: Sending intermediate data.")
#         self._send_data(payload["byte_data"])
#         payload["byte_data"] = self._receive_data()
#         print(f"[CLIENT]: Received processed data.")
#         return payload

#     def receive_data(self, payload: Dict[str, Any]) -> None:
#         """Receive data and store in the provided dictionary."""
#         payload["byte_data"] = self._receive_data()
#         print(f"[CLIENT]: Received processed data.")
#         return payload
