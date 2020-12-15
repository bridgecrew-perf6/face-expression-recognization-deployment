import os
import socket
import cv2
import time
import struct
import datetime
import hashlib

# 测试Demo参数
video_width = 1280
video_height = 720
frame_cnt = 10
interval_sec = 2
video_url = 0#'rtsp://admin:bwton123@192.168.24.64:554'  # 0代表本机摄像头，如果使用网络摄像头，把此处的0换成url

# video_url = 0


class CommonTools:
    # 获取时间搓
    @staticmethod
    def get_timestamp():
        return int(time.time())

    # 获取今天明天昨天的日期
    @staticmethod
    def get_today_day():
        return datetime.date.today().strftime("%Y%m%d")

    @staticmethod
    def get_new_filename():
        return time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

    # 创建文件夹函数
    @staticmethod
    def mkdir(path):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

    # 打印16进制
    @staticmethod
    def print_hex(bytes):
        l = [hex(int(i)) for i in bytes]
        print(" ".join(l))

    @staticmethod
    def get_file_md5(file_name):
        """
        计算文件的md5
        :param file_name:
        :return:
        """
        m = hashlib.md5()  # 创建md5对象
        with open(file_name, 'rb') as fobj:
            while True:
                data = fobj.read(4096)
                if not data:
                    break
                m.update(data)  # 更新md5对象
        return m.hexdigest()  # 返回md5对象


# 监控摄像头和打包视频文件
class CameraVideo:
    # 类属性，可以通过类对象调用
    __path = '.'
    __datetime = ''
    __out_name = ''

    def __init__(self, w, h):
        self.root_path = "/tmp/"
        frame_width = w  # init里定义的是实例对象，只能通过实例对象使用
        frame_height = h
        self.size = (frame_width, frame_height)
        self.fps = 30  # 每秒播放帧数

        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.out = cv2.VideoWriter()

    # 打开视频文件保存
    def open_video(self):
        self.__datetime = CommonTools.get_new_filename()
        self.__out_name = "tmp_client.mp4"
        self.__path = self.root_path
        CommonTools.mkdir(self.__path)
        print(self.__path + self.__out_name)
        if self.__out_name == '':
            self.out.open(self.__path + r'./output.mp4', self.fourcc, self.fps, self.size, True)
        else:
            self.out.open(self.__path + self.__out_name, self.fourcc, self.fps, self.size, True)
        return self.__path + self.__out_name, self.__datetime

    def get_datetime(self):
        return self.__datetime

    # 帧写入视频文件
    def write(self, frame):
        self.out.write(frame)

    # 关闭视频文件
    def close(self):
        self.out.release()


class SocketTCP:
    __stx = "02"
    __etx = "03"
    __crc = "FFFFFFFF"

    def __init__(self, w, h, f):
        self.ret = 0
        self.__datetime = ''
        self.__total_frame = f  # 视频总帧数
        self.__format = 1  # 视频格式1代表mp4
        self.__target_ip = '127.0.0.1'  # 目标地址
        self.__target_port = 3000  # 目标端口
        self.__frame_width = w  # 帧宽
        self.__frame_height = h  # 帧高
        self.__video_size = 0  # 视频总大小字节
        self.__pkg_cnt = 0  # 报文总帧数
        self.__md5 = ''
        self.__send_buffer = 2048  #

        self.file_path = ''
        self.file_name = ''

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # # 设置超时，仅仅是linux上的写法，设置接收超时时间为 5s 50000us ，即5.05s这样使用：
        # val = struct.pack("QQ", 5, 50000)

        # windows
        val = struct.pack("QQ", 5000, 50000)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, val)
        self.sock.connect((self.__target_ip, self.__target_port))

    def __del__(self):  # 当程序结束时运行
        print("析构函数")

    def case_cmd_start(self, data):
        self.ret = 0
        print("case_cmd_start")
        data_str = data.hex()
        self.__datetime = data[:14].decode('utf-8')
        self.__total_frame = int(data_str[28:32], 16)
        self.__format = int(data_str[32:34], 16)
        self.__frame_width = int(data_str[34:38], 16)
        self.__frame_height = int(data_str[38:42], 16)
        self.__video_size = int(data_str[42:50], 16)
        self.__pkg_cnt = int(data_str[50:58], 16)
        self.__md5 = data_str[58:]

        print(self.__datetime)
        print(self.__total_frame)
        print(self.__frame_width)
        print(self.__frame_height)
        print(self.__video_size)
        print(self.__pkg_cnt)
        print(self.__md5)

    def case_cmd_end(self, data):
        self.ret = 0
        print("case_cmd_end")
        return self.ret, data

    def case_cmd_result(self, data):
        self.ret = 0
        print("case_cmd_result")
        datetime = data[:14].decode('utf-8')
        result_len = int(data[14:18].hex(), 16)
        result = data[22:].decode('utf-8')

        print("datetime", datetime)
        print("result_len", result_len)
        print("result json", result)
        return self.ret, data

    def open_video(self, path, datetime):
        self.file_path = path
        self.__md5 = CommonTools.get_file_md5(self.file_path)
        self.file_name = os.path.split(self.file_path)[1]
        self.__datetime = datetime
        self.__video_size = os.path.getsize(self.file_path)
        pkg_cnt = self.__video_size // self.__send_buffer

        if self.__video_size % self.__send_buffer == 0:
            self.__pkg_cnt = pkg_cnt
        else:
            self.__pkg_cnt = pkg_cnt + 1

        print("filename:" + self.file_name + "\nfilesize:" + str(self.__video_size))

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
        else:
            return None

    def protocol_pack(self, data):
        tail = bytes.fromhex(self.__crc + self.__etx)
        data_head = self.__stx + "%08x" % (len(data) + 4)  # 长度加4个字节的crc32
        head = bytes.fromhex(data_head)
        send_data = head + data + tail
        CommonTools.print_hex(send_data)
        return send_data

    def protocol_unpack(self, buf):
        real_data = '00'

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
            # print(data)
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

    def send(self):
        head = self.protocol_pack(self.data_pack(1, 0, None))
        self.sock.sendall(head)
        try:
            data_cmd = self.sock.recv(self.__send_buffer)
            if data_cmd:
                CommonTools.print_hex(data_cmd)
                ret, real_cmd, real_data = self.protocol_unpack(data_cmd)  # 确认已经接收到视频信息
                if ret == 0 and real_cmd == 1:
                    pass
                else:
                    return
        except Exception as err:
            print(err)
            return

        rest_size = self.__video_size
        fd = open(self.file_path, 'rb')
        count = 0

        while rest_size >= self.__send_buffer:
            data = fd.read(self.__send_buffer)
            self.sock.sendall(data)
            rest_size = rest_size - self.__send_buffer
            # print(str(count) + " ")
            count = count + 1

        data = fd.read(rest_size)
        self.sock.sendall(data)
        fd.close()
        print("send successfully")

    def recv(self):
        try:
            data_cmd = self.sock.recv(self.__send_buffer)
            if data_cmd:
                CommonTools.print_hex(data_cmd)
                ret, real_cmd, real_data = self.protocol_unpack(data_cmd)
                if real_cmd == 1:
                    self.case_cmd_start(real_data)
                elif real_cmd == 5:
                    self.case_cmd_end(real_data)
                elif real_cmd == 8:
                    self.case_cmd_result(real_data)
                    self.sock.sendall(self.protocol_pack(bytes.fromhex('08')))
                else:
                    print("Error Cmd")
            self.sock.close()
        except Exception as err:
            print(err)
            return


# 测试Demo
m_video = CameraVideo(video_width, video_height)
video_path, date_time = m_video.open_video()
cap = cv2.VideoCapture(video_url)
cap.set(3, video_width)
cap.set(4, video_height)

cnt = 0
m_latest_time = 0
while cap.isOpened():
    ret_flag, frame = cap.read()
    cv2.putText(frame, str(cnt), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('Capture', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    if m_latest_time != 0:
        if (CommonTools.get_timestamp() - m_latest_time) >= interval_sec:
            m_latest_time = 0
        else:
            continue

    if cnt < frame_cnt:  # 写n帧
        cnt += 1
        m_video.write(frame)
    else:
        cnt = 0
        m_video.close()

        # 开始发送
        client_m = SocketTCP(video_width, video_height, frame_cnt)
        print("-----------------------------------")
        print(video_path)
        # client_m.open_video("./20201201173152.mp4")
        client_m.open_video(video_path, date_time)
        client_m.send()
        client_m.recv()
        print("-----------------------------------\n\n")
        m_latest_time = CommonTools.get_timestamp()
        video_path, date_time = m_video.open_video()



cap.release()
cv2.destroyAllWindows()
