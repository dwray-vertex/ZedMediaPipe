import json
import socket
import sys
import subprocess

LISTEN_IP = "10.10.1.135"
LISTEN_PORT = 7778
BUFFER SIZE = 4096

command = ["python3", "../MediaPipe/MPZED_Track_hybrid2.py", "--no-display"]

def main():
    try:
        rec_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rec_sock.bing((LISTEN_IP, LISTEN_PORT))
        print("Listening for tracking command)
    excepot Exception as e:
        sys.exit(1)
        
    while True:
        try:
        
            data, address = rec_sock.recvfrom(BUFFER_SIZE)
            
            message_str = data.decode('utf-8')
            print(f"Received message from {address}")
            
            try:
                json_data = json.loads(message_str)
                
                try:
                    if (json_data is not None)
                        result = subprocess.run(command, capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError as ep:
                    print("")
                
            except json.JSONDecodeError:
                print("Received non-JSON message")
                
        except Exception as e:
            print("Error handling packet")
            
if __name__ == "__main__":
    main()
