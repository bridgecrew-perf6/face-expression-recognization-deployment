import socket
import struct

""">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
import thuface
affectnet7_table = {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger"}
detector = thuface.CV2FaceDetector("ckpt/haarcascade_frontalface_default.xml")
classifier = thuface.model.MobileNetV3_Small()
classifier = thuface.get_trained_model(classifier,"ckpt/checkpoint_best.pth.tar")

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
batch_solver = thuface.BatchSolver(detector, classifier, device, affectnet7_table, batch_size=100, DEBUG=True)


"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"""

# 回调函数，用于处理接收到的视频文件
def on_video_complete(video_path, datetime, total_frame, format, frame_width, frame_height, video_size):
    ret = 0
    print("Callback Start")
    print("datetime", datetime)
    print("total_frame", total_frame)
    print("format", format)
    print("frame_width", frame_width)
    print("frame_height", frame_height)
    print("video_size", video_size)

    # 得到视频路径
    print("video_path", video_path)

    # 开始模型分析
    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    batch_solver.set_batch_size(total_frame)
    result = batch_solver.solve_path(video_path)

    # 返回分析结果
    json = batch_solver.to_json(result)
    """<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"""
    
    print("Recognise Result", json)
    print("Callback End")
    return ret, json


class SocketTCP:
    __stx = "02"  # 起始位
    __etx = "03"  # 结束位
    __crc = "FFFFFFFF"  # CRC32,当前先用F表示

    def __init__(self):
        self.ret = 0
        self.__on_video_complete = None  # 接收视频之后调用该回调函数
        self.__datetime = ''  # 视频创建时间
        self.__target_ip = '127.0.0.1'  # 目标地址
        self.__target_port = 3000  # 目标端口
        self.__total_frame = 0  # 视频总帧数
        self.__format = 0  # 视频格式1代表mp4
        self.__frame_width = 0  # 帧宽
        self.__frame_height = 0  # 帧高
        self.__video_size = 0  # 视频总大小字节
        self.__pkg_cnt = 0  # 报文总帧数
        self.__md5 = ''  # 视频md5
        self.__send_buffer = 2048  # 一次接收最大数据

        self.listenSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置超时，仅仅是linux上的写法，设置接收超时时间为 5s 50000us ，即5.05s这样使用：
        # val = struct.pack("QQ", 500, 50000)

        # windows
        val = struct.pack("QQ", 5000, 50000)
        self.listenSock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, val)
        self.listenSock.bind((self.__target_ip, self.__target_port))
        self.listenSock.listen(5)

    # 0x01指令，报文包含一些待发送视频基本信息，之后开始传输视频文件
    def case_cmd_start(self, data):
        self.ret = 0
        print("--------------------------")
        print("case_cmd_start")
        data_str = data.hex()
        self.__datetime = data[:14].decode('utf-8')
        self.__total_frame = int(data_str[28:32], 16)
        self.__format = int(data_str[32:34], 16)
        self.__frame_width = int(data_str[34:38], 16)
        self.__frame_height = int(data_str[38:42], 16)
        self.__video_size = int(data_str[42:50], 16)
        self.__pkg_cnt = int(data_str[50:58], 16)
        self.__md5 = data[29:].decode('utf-8')

    # 0x05 传输停止，暂时用不到
    def case_cmd_end(self, data):
        self.ret = 0
        print("case_cmd_end")
        return self.ret, data

    # 0x08 返回的json数据
    def case_cmd_result(self, data):
        self.ret = 0
        print("case_cmd_result")
        return self.ret, data

    # 回调函数，接收到完整视频之后调用
    def set_callback_func(self, func):
        self.__on_video_complete = func

    # 主循环
    def main_loop(self):
        if self.__on_video_complete is None:
            return

        while True:
            print("main loop")
            self.recv()

    # 生成报文数据域
    def data_pack(self, cmd, frame_num, frame):
        if cmd == 1:  # 开始传输帧信息
            # 日期时间，总帧，格式，宽，高
            data_head = "%04x%02x%04x%04x%08x%08x" % (self.__total_frame, 1, self.__frame_width, self.__frame_height,
                                                      self.__video_size, self.__pkg_cnt)
            data_body = bytes.fromhex("01") + self.__datetime.encode('utf-8') + bytes.fromhex(
                data_head) + self.__md5.encode('utf-8')
            return data_body
        elif cmd == 3:  # 传输数据帧指令
            frame_head = "03%08x" % frame_num
            return bytes.fromhex(frame_head) + frame
        elif cmd == 5:  # 结束传输指令
            # 发送文件头、数据:
            return bytes.fromhex('05')
        elif cmd == 8:  # 结束传输指令
            data_head = "%08x" % len(frame)
            data_body = bytes.fromhex("08") + self.__datetime.encode('utf-8') + bytes.fromhex(
                data_head) + frame
            return data_body
        else:
            return None

    # 给数据报文添加协议包头和包尾
    def protocol_pack(self, data):
        tail = bytes.fromhex(self.__crc + self.__etx)
        data_head = self.__stx + "%08x" % (len(data) + 4)  # 长度加4个字节的crc32
        head = bytes.fromhex(data_head)
        send_data = head + data + tail
        return send_data

    # 去除报文的协议头和尾
    def protocol_unpack(self, buf):
        real_data = ''
        while True:
            data = buf.hex()
            if len(data) < 22:  # 错误数据，或报文不完整
                print("invalid data, drop it")
                self.ret = -1
                break

            start_bit = int(data[:2], 16)
            pack_len = int(data[2:10], 16)
            # cmd = int(data[10:12], 16)
            crc1 = int(data[len(data) - 10:len(data) - 6], 16)
            crc2 = int(data[len(data) - 6:len(data) - 2], 16)
            end_bit = int(data[len(data) - 2:], 16)
            if start_bit != 0x02 or end_bit != 0x03:
                print("协议错误")
                self.ret = -1
                break
            if crc1 != 0xffff or crc2 != 0xffff:
                print("crc错误")
                self.ret = -1
                break

            real_data = buf[5:pack_len + 5 - 4]  # cmd+data
            self.ret = 0
            break

        if len(real_data) > 1:
            return self.ret, real_data[0], real_data[1:]
        else:
            return self.ret, real_data[0], None

    # 接收视频数据
    def recv(self):
        try:
            conn, addr = self.listenSock.accept()
            data_cmd = conn.recv(self.__send_buffer)
            # 处理指令
            if data_cmd:
                # print_hex(data_cmd)
                ret, real_cmd, real_data = self.protocol_unpack(data_cmd)
                if real_cmd == 1:
                    self.case_cmd_start(real_data)
                    conn.send(self.protocol_pack(bytes.fromhex("0100")))
                elif real_cmd == 5:
                    self.case_cmd_end(real_data)
                    return
                elif real_cmd == 8:
                    self.case_cmd_result(real_data)
                    return
                else:
                    print("Error Cmd")
                    return

                video_path = "/tmp/tmp_server.mp4"
                print("filename:" + video_path + "\nfilesize:" + str(self.__video_size))
                recved_size = 0
                fd = open(video_path, 'wb')
                # count = 0
                while True:
                    data = conn.recv(self.__send_buffer)
                    recved_size = recved_size + len(data)  # 虽然buffer大小是4096，但不一定能收满4096
                    fd.write(data)
                    if recved_size >= self.__video_size:
                        print("recv file successfully")
                        break
                fd.close()
                ret, json = self.__on_video_complete(video_path, self.__datetime, self.__total_frame, self.__format,
                                                     self.__frame_width, self.__frame_height, self.__video_size)
                if ret == 0:
                    conn.send(self.protocol_pack(self.data_pack(8, 0, json.encode('utf-8'))))
                    data_reply = conn.recv(self.__send_buffer)
                    ret, real_cmd, real_data = self.protocol_unpack(data_reply)
                    if real_cmd == 8:
                        print("recognise successfully")
                        print("--------------------------\n\n")
                conn.close()
        except Exception as err:
            print(err)
            return


# 测试Demo
server_m = SocketTCP()      # 创建对象
server_m.set_callback_func(on_video_complete)  # 传入回调函数，当接收到新的完整视频时会被调用
server_m.main_loop()                           # 启动监听进程
