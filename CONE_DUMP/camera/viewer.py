import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new

HOST=''
PORT=9999

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.connect(('',9999))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

### new
data = ""
payload_size = struct.calcsize("L") 
while True:
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    ###

    frame=pickle.loads(frame_data)
    print(frame)
    cv2.imshow('frame',frame)

    key = cv2.waitKey(10)
    if (key == 27) or (key == 113):
        break

cv2.destroyAllWindows()

