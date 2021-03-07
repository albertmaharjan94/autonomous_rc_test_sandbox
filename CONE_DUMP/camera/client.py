import socket

HOST = '192.168.43.219'
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            data = s.recv(1024)
            print('Received', repr(data))
    except Exception as e:
        print(e)
