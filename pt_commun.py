import socket
import struct
from ctypes import *


class CaptrackConnection:
    def __init__(self):
        self.client_socket = None

    def connect_to_captrack(self):
        captrack_socket = socket.socket()
        captrack_port = 4949
        captrack_ip = '192.168.10.126'
        captrack_socket.connect((captrack_ip, captrack_port))
        self.client_socket = captrack_socket

    @staticmethod
    def val2packet(val) -> list:
        if val == 0:
            c = ['00', '00', '00', '00']
            return c
        else:
            #######STEP 1: convert int to hex##########
            a = hex(struct.unpack('<I', struct.pack('<f', val))[0])
            #######STEP 2: split hex into pairs##########
            a = a[2:] if len(a) % 2 == 0 else "0" + a[2:]
            b = " ".join(a[i:i + 2] for i in range(0, len(a), 2))
            #######STEP 3: convert sliced hex into list ##########
            return b.split(" ")

    @staticmethod
    def calc_check_sum(packet):
        '''caculation of the checksome component'''
        msg_sum = 0
        length = packet[2]
        for i in range(2, 3 + length, 1):
            msg_sum += packet[i]
        if msg_sum > 255:
            return msg_sum % 256
        else:
            return msg_sum

    @staticmethod
    def convert_to_float(data):
        i = int(data, 16)
        cp = pointer(c_int(i))
        fp = cast(cp, POINTER(c_float))
        return fp.contents.value

    def speed_movement(self, axis, speed, acceleration):
        acc_p = self.val2packet(acceleration)
        speed_p = self.val2packet(speed)

        # building the packets 
        MOT_SetTum = [0x50, 0x54, 0x04, 0x00, axis, 0x01, 0x3F, 0x45]
        MOT_SetSpeedMode = [0x50, 0x54, 0x04, 0x00, axis, 0x01, 0x3A, 0x00]
        MOT_SetAcceleration = [0x50, 0x54, 0x08, 0x00, axis, 0x01, 0x30, int(acc_p[0], 16), int(acc_p[1], 16), int(acc_p[2], 16), int(acc_p[3], 16), 0x44]
        MOT_SetSpeed = [0x50, 0x54, 0x08, 0x00, axis, 0x01, 0x31, int(speed_p[0],16), int(speed_p[1], 16), int(speed_p[2], 16), int(speed_p[3], 16), 0x08]

        MOT_Update = [0x50, 0x54, 0x04, 0x00, axis, 0x01, 0x34, 0x3A]

        MOT_SetTum[-1] = self.calc_check_sum(MOT_SetTum)
        MOT_SetSpeedMode[-1] = self.calc_check_sum(MOT_SetSpeedMode)
        MOT_SetAcceleration[-1] = self.calc_check_sum(MOT_SetAcceleration)
        MOT_SetSpeed[-1] = self.calc_check_sum(MOT_SetSpeed)
        MOT_Update[-1] = self.calc_check_sum(MOT_Update)

        self.client_socket.send(bytes(MOT_SetSpeedMode))
        data1 = bytes.hex(self.client_socket.recv(1024))
        self.client_socket.send(bytes(MOT_SetTum))
        data2 = bytes.hex(self.client_socket.recv(1024))
        self.client_socket.send(bytes(MOT_SetAcceleration))
        data3 = bytes.hex(self.client_socket.recv(1024))
        self.client_socket.send(bytes(MOT_SetSpeed))
        data4 = bytes.hex(self.client_socket.recv(1024))
        self.client_socket.send(bytes(MOT_Update))
        data5 = bytes.hex(self.client_socket.recv(1024))

    def position_movement(self, axis, pos, speed, acc):
        # acceleration
        convert_acc = self.val2packet(acc)
        # speed
        convert_speed = self.val2packet(speed)
        # position
        convert_pos = self.val2packet(pos)

        # building the packets 
        MOT_SetTum = [0x50, 0x54, 0x04, 0x00, axis, 0x01, 0x3F, 0x45]
        MOT_SetPositionRelative = [0x50, 0x54, 0x04, 0x00, axis, 0x01, 0x38, 0x3F]
        MOT_SetPositionMode = [0x50, 0x54, 0x04, 0x00, axis, 0x01, 0x3B, 0x00]
        MOT_SetAcceleration = [0x50, 0x54, 0x08, 0x00, axis, 0x01, 0x30, int(convert_acc[0], 16), int(convert_acc[1], 16), int(convert_acc[2], 16), int(convert_acc[3], 16), 0x44]
        MOT_SetSpeed = [0x50, 0x54, 0x08, 0x00, axis, 0x01, 0x31, int(convert_speed[0], 16), int(convert_speed[1], 16), int(convert_speed[2], 16), int(convert_speed[3], 16), 0x08]
        MOT_SetPosition = [0x50, 0x54, 0x08, 0x00, axis, 0x01, 0x32, int(convert_pos[0], 16), int(convert_pos[1], 16), int(convert_pos[2], 16), int(convert_pos[3], 16), 0x5F]
        MOT_Update = [0x50, 0x54, 0x04, 0x00, axis, 0x01, 0x34, 0x3A]

        # calculation of checksome 
        MOT_SetTum[-1] = self.calc_check_sum(MOT_SetTum)
        MOT_SetPositionRelative[-1] = self.calc_check_sum(MOT_SetPositionRelative)
        MOT_SetPositionMode[-1] = self.calc_check_sum(MOT_SetPositionMode)
        MOT_SetAcceleration[-1] = self.calc_check_sum(MOT_SetAcceleration)
        MOT_SetSpeed[-1] = self.calc_check_sum(MOT_SetSpeed)
        MOT_SetPosition[-1] = self.calc_check_sum(MOT_SetPosition)
        MOT_Update[-1] = self.calc_check_sum(MOT_Update)

        # sending the packets 
        self.client_socket.send(bytes(MOT_SetTum))
        data1 = bytes.hex(self.client_socket.recv(1024))

        self.client_socket.send(bytes(MOT_SetPositionRelative))
        data2 = bytes.hex(self.client_socket.recv(1024))

        self.client_socket.send(bytes(MOT_SetPositionMode))
        data3 = bytes.hex(self.client_socket.recv(1024))

        self.client_socket.send(bytes(MOT_SetAcceleration))
        data4 = bytes.hex(self.client_socket.recv(1024))

        self.client_socket.send(bytes(MOT_SetSpeed))
        data5 = bytes.hex(self.client_socket.recv(1024))

        self.client_socket.send(bytes(MOT_SetPosition))
        data6 = bytes.hex(self.client_socket.recv(1024))
        
        self.client_socket.send(bytes(MOT_Update))
        data7 = bytes.hex(self.client_socket.recv(1024))

    def axis_off(self):
        axis_1_off = [0x50, 0x54, 0x04, 0x00, 0x01, 0x01, 0x3D, 0x00]
        axis_2_off = [0x50, 0x54, 0x04, 0x00, 0x02, 0x01, 0x3D, 0x00]

        axis_1_off[-1] = self.calc_check_sum(axis_1_off)
        axis_2_off[-1] = self.calc_check_sum(axis_2_off)

        self.client_socket.send(bytes(axis_1_off))
        data1 = bytes.hex(self.client_socket.recv(1024))

        self.client_socket.send(bytes(axis_2_off))
        data2 = bytes.hex(self.client_socket.recv(1024))

    def axis_on(self):
        axis_1_off = [0x50, 0x54, 0x04, 0x00, 0x01, 0x01, 0x3C, 0x00]
        axis_2_off = [0x50, 0x54, 0x04, 0x00, 0x02, 0x01, 0x3C, 0x00]

        axis_1_off[-1] = self.calc_check_sum(axis_1_off)
        axis_2_off[-1] = self.calc_check_sum(axis_2_off)

        self.client_socket.send(bytes(axis_1_off))
        data1 = bytes.hex(self.client_socket.recv(1024))

        self.client_socket.send(bytes(axis_2_off))
        data2 = bytes.hex(self.client_socket.recv(1024))