import io
import socket
import time

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 5000))
server_socket.listen(0)
c, addr = server_socket.accept()


try:	
        while True:
                file1 = open("myfile.txt","r+") 
                L= file1.read()
                time.sleep(0.5)
                c.send(L.encode())
                file1.close()
                
finally:
        c.close()
        server_socket.close()
