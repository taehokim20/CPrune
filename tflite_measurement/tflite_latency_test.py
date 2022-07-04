#!/usr/bin/env python3

import socket
import sys

HOST_PC = '0.0.0.0' #'10.201.90.27' #'128.138.224.82'
#HOST_PHONE = '10.201.59.155' #(S9)
#HOST_PHONE = '10.201.40.106' #(S20+)
HOST_PHONE = '192.168.0.152'
#PORT_PC = 8888
PORT = 7801 

# create the socket
# AF_INET == ipv4
# SOCK_STREAM == TCP
#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

########### Client part ###############
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
    s2.connect((HOST_PHONE, PORT))
    s2.sendall(b'Hello, world')
#    data = s.recv(1024)

#print('Received', repr(data))
s2.close()

########### Server part ###############
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind socket to Host and Port
try:
    s.bind((HOST_PC, PORT))
except socket.error as err:
    print('Bind Failed, Error Code: ' + str(err[0]) + ', Message: ' + err[1])
    sys.exit()

print('Socket Bind Success!')

#listen(): This method sets up and start TCP listener.
s.listen(10)
print('Socket is now listening')

while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, addr = s.accept()
    print('Connect with ' + addr[0] + ':' + str(addr[1]))
    buf = clientsocket.recv(64)
    print(buf)
    break
s.close()


