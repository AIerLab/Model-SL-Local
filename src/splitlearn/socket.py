import socket
from typing import Dict, Any
import hashlib
from helper import NoneException

# Constants
ACK = b"ACK"
NACK = b"NACK"
EOF = b"EOF"
# TIMEOUT = 60

class SplitSocket:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.client_socket.settimeout(TIMEOUT)

        try:
            self.client_socket.connect((host, port))
        except Exception as e:
            print(f"Could not connect to server: {e}")
            return

    def compute_checksum(self, data):
        return hashlib.md5(data).digest()

    def send_data(self, data: Dict[str, Any]) -> None:
        print(f"[CLIENT]: Sending intermediate data.")
        self._send_data(data["byte_data"])
        data["byte_data"] = self._receive_data()
        print(f"[CLIENT]: Received processed data.")
        return data
    
    def reveive_data(self, format_data: Dict[str, Any]) -> None:
        format_data["byte_data"] = self._receive_data()
        print(f"[CLIENT]: Received processed data.")
        return format_data

    def _send_data(self, data: bytes):
        while True:
            try:
                checksum = self.compute_checksum(data)
                self.client_socket.sendall(checksum + data + EOF)
                ack = self.client_socket.recv(len(ACK))
                if ack == ACK:
                    break
                else:
                    print("Checksum mismatch while sending data. Retrying...")
            except socket.timeout:
                print("Timeout while waiting for ACK. Retrying...")
            except Exception as e:
                print(f"Error sending data: {e}")

    def _receive_data(self) -> bytes:
        while True:
            try:
                data = b""
                while True:
                    chunk = self.client_socket.recv(4096)
                    if EOF in chunk:
                        data += chunk[:-len(EOF)]
                        break
                    data += chunk
                
                received_checksum = data[:16]
                data = data[16:]
                if self.compute_checksum(data) == received_checksum:
                    self.client_socket.sendall(ACK)
                    if data == b"":
                        raise NoneException("None received")
                    return data
                else:
                    self.client_socket.sendall(NACK)
                    print("Checksum mismatch while receiving data. Retrying...")
            except socket.timeout:
                print("Timeout while receiving data. Retrying...")
            except Exception as e:
                print(f"Error receiving data: {e}")

    def close_connection(self):
        if self.client_socket:
            self.client_socket.close()
