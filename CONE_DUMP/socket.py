import socket
import time
import threading
import time
import cv2 
import base64
import threading
 
# socket init
HOST = '192.168.43.238'
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
 
_socket = True
command = None
 
 
cap = cv2.VideoCapture(0)
open = True
buffer = None
 
def camera():
    global buffer
    global open
    while True:
        _,frame = cap.read()
        cv2.imshow("k", frame)
        t = cv2.waitKey(1)
        if t== ord('q'):
            open = True
            break
        _, buffer = cv2.imencode('.jpg', frame)
 
 
 
 
 
thread_camera = threading.Thread(target=camera).start()
 
 
while open:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()
    while _socket:
        conn, addr = s.accept()   
        try:
            # with conn:
            if buffer is not None:
                print("imbuffer")
                conn.sendall(base64.b64encode(buffer))
        except Exception as e:
            print(e)
            conn.close() 
 
cap.release()
thread_camera.join()
 
 
 