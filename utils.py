import pickle, struct, socket
import time
import os

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    send_start = time.time()
    sock.sendall(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())
    send_over = time.time()
    return send_over - send_start

def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    recvstart = time.time()
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    restop = time.time()

    print(msg[0], 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg, restop-recvstart

def file_send(sock, filepath):

    # 判断是否为文件
    if os.path.isfile(filepath):
        print('filepath true')
        # 定义定义文件信息。128s表示文件名为128bytes长，l表示一个int或log文件类型，在此为文件大小
        fileinfo_size = struct.calcsize('256sl')
        # 定义文件头信息，包含文件名和文件大小
        fhead = struct.pack('256sl', os.path.basename(filepath).encode('utf-8'), os.stat(filepath).st_size)
        # 发送文件名称与文件大小
        print('filename sending...')
        sock.send(fhead)
        sock.recv(1024)

        print('Start sending......')
        # 将传输文件以二进制的形式分多次上传至服务器
        fp = open(filepath, 'rb')
        while 1:
            data = fp.read(1024)
            if not data:
                print('{0} file send over...'.format(os.path.basename(filepath)))
                break
            sock.send(data)
            sock.recv(1024)
    else:
        print('filepath error')


def file_recv(sock, ):
    fileinfo_size = struct.calcsize('256sl')
    # 接收文件名与文件大小信息
    buf = sock.recv(fileinfo_size)
    # 判断是否接收到文件头信息
    if buf:
        sock.send(b'00')
        # 获取文件名和文件大小
        filename, filesize = struct.unpack('256sl', buf)
        fn = filename.strip(b'\00')
        fn = fn.decode()
        print('file new name is {0}, filesize if {1}'.format(str(fn), filesize))

        recvd_size = 0  # 定义已接收文件的大小
        # 存储在该脚本所在目录下面
        fp = open('./test/' + str(fn), 'wb')
        print('start receiving...')

        # 将分批次传输的二进制流依次写入到文件
        while not recvd_size == filesize:
            if filesize - recvd_size > 1024:
                data = sock.recv(1024)
                recvd_size += len(data)
                sock.send(b'00')
            else:
                data = sock.recv(filesize - recvd_size)
                recvd_size = filesize
                sock.send(b'00')
            fp.write(data)
            fp.flush()
        fp.close()
        time.sleep(0.5)
        print('end receive...')
    else:
        print('buf error')

