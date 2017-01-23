import signal
import socket
import csv
import time
import struct
from enum import Enum


class RecordFlags(Enum):
    RECORD_FLAG_PAUSE = 0
    RECORD_FLAG_QUESTION = 1
    RECORD_FLAG_ANSWER_TRUE = 2
    RECORD_FLAG_ANSWER_FALSE = 3
    RECORD_FLAG_CONTROL_SHAPE_TRUE = 4
    RECORD_FLAG_CONTROL_SHAPE_FALSE = 5
    RECORD_FLAG_START_SESSION = 6
    RECORD_FLAG_END_SESSION = 7


BLOCK_ID_TRACKING_STATE = 33433  # According to faceshift docs

# These are the names of the FaceShift control channels
blend_shape_names = [
    "EyeBlink_L",
    "EyeBlink_R",
    "EyeSquint_L",
    "EyeSquint_R",
    "EyeDown_L",
    "EyeDown_R",
    "EyeIn_L",
    "EyeIn_R",
    "EyeOpen_L",
    "EyeOpen_R",
    "EyeOut_L",
    "EyeOut_R",
    "EyeUp_L",
    "EyeUp_R",
    "BrowsD_L",
    "BrowsD_R",
    "BrowsU_C",
    "BrowsU_L",
    "BrowsU_R",
    "JawFwd",
    "JawLeft",
    "JawOpen",
    "JawChew",
    "JawRight",
    "MouthLeft",
    "MouthRight",
    "MouthFrown_L",
    "MouthFrown_R",
    "MouthSmile_L",
    "MouthSmile_R",
    "MouthDimple_L",
    "MouthDimple_R",
    "LipsStretch_L",
    "LipsStretch_R",
    "LipsUpperClose",
    "LipsLowerClose",
    "LipsUpperUp",
    "LipsLowerDown",
    "LipsUpperOpen",
    "LipsLowerOpen",
    "LipsFunnel",
    "LipsPucker",
    "ChinLowerRaise",
    "ChinUpperRaise",
    "Sneer",
    "Puff",
    "CheekSquint_L",
    "CheekSquint_R"]


class FaceShiftReceiver:
    """This is the receiving Thread listening for FaceShift UDP messages on some port."""

    @staticmethod
    def decode_faceshift_datastream(data_dict, data):
        """ Takes as input the bytes of a binary DataStream received via network.

         If it is a Tracking State block (ID 33433) then extract some data (info, blendshapes, markers, ...).
         Otherwise None is returned.
        """

        # block_id = struct.unpack_from("H", data)
        # print("Received block id " + str(block_id)) ;

        offset = 0
        block_id, version, block_size = struct.unpack_from("HHI", data, offset)

        # print("ID, v, size = " + str(block_id) + "," + str(version) + "," + str(block_size) )

        offset += 8

        if block_id == BLOCK_ID_TRACKING_STATE:
            n_blocks, = struct.unpack_from("H", data, offset)
            # print("n_blocks = " + str(n_blocks))
            offset += 2

            track_ok = 0  # Will be a byte: 1 if tracking ok, 0 otherwise.
            head_rotation_quat = None  # Will be filled with the rotation using mathutils.Quaternion
            blend_shape_values = []  # Will be a list of float in the range 0-1
            # eyes_values = None          # Will be a sequence of 4 angle values
            markers_position = []  # Will be a list of mathutils.Vector

            curr_block = 0
            while curr_block < n_blocks:
                block_id, version, block_size = struct.unpack_from("HHI", data, offset)
                # print("ID, v, size = " + str(block_id) + "," + str(version) + "," + str(block_size) )

                # put the offset at the beginning of the block
                offset += 8

                if block_id == 101:  # Frame Information blobk (timestamp and tracking status)
                    ts, track_ok = struct.unpack_from("dB", data, offset)
                    # print("timestamp, track_ok " + str(ts) + ", " + str(track_ok) )
                    # offset += 9
                elif (block_id == 102):  # Pose block (head rotation and position)
                    x, y, z, w = struct.unpack_from("ffff", data, offset)
                    # head_rotation_quat = mathutils.Quaternion((w, x, y, z))
                elif (block_id == 103):  # Blendshapes block (blendshape values)
                    n_coefficients, = struct.unpack_from("I", data, offset)
                    # print("Blend shapes count="+ str(n_coefficients) )
                    i = 0
                    # coeff_list = ""
                    while i < n_coefficients:
                        # Offset of the block, plus the 4 bytes for int n_coefficients, plus 4 bytes per float
                        val, = struct.unpack_from("f", data, offset + 4 + (i * 4))
                        blend_shape_values.append(val)
                        # coeff_list += repr(val) + " "
                        i += 1
                        # print("Values: " + coeff_list)
                elif (block_id == 104):  # Eyes block (eyes gaze)
                    leye_theta, leye_phi, reye_theta, reye_phi = struct.unpack_from("ffff", data, offset)
                elif (block_id == 105):  # Markers block (absolute position of mark points)
                    n_markers, = struct.unpack_from("H", data, offset)
                    # print("n markers="+str(n_markers))
                    i = 0
                    while (i < n_markers):
                        # Offset of the block, plus the 2 bytes for int n_markers, plus 4 bytes for each x,y,z floats
                        x, y, z = struct.unpack_from("fff", data, offset + 2 + (i * 4 * 3))
                        # print("m" + str(i) + " " + str(x) + "\t" + str(y) + "\t" + str(z))
                        markers_position.append((x, y, z))
                        i += 1

                curr_block += 1
                offset += block_size

            # end -- while on blocks. Track State scan complete

            # Handle HEAD ROTATION
            if track_ok == 1:
                if head_rotation_quat is not None:
                    # HeadRot2Rig(target_object, head_rotation_quat)
                    print("Head rotation")

            #
            # Handle BLEND SHAPES
            # print(str(track_ok) + " - " + str(len(blend_shape_values)))
            if track_ok == 1:
                # FS2Rig_MappingDict(target_object, blend_shape_names, blend_shape_values)
                # print("Blend shapes")
                data_dict["blend_shapes"]["values"].append((ts, data_dict["record_flag"], blend_shape_values))
                #
                # Handle EYES
                # print(str(track_ok) + " - " + str(len(blend_shape_values)))
                # if (track_ok == 1):
                # EyesRot2Skeleton(target_object, leye_theta, leye_phi, reye_theta, reye_phi)
                # print("Eyes")


DATA = {
    "blend_shapes": {
        "names": blend_shape_names,
        "values": []  # tuples (timestamp, record_flag, [values])
    },
    "record_flag": False
}


def fin_handler(signal, frame):
    save_and_exit()


def save_and_exit():
    print("Stop signal received")

    fn = "data/output/fs_shapes.{}.csv".format(time.time())
    with open(fn, "w", newline='') as out:
        wr = csv.writer(out)

        header = ["record_flag", "timestamp"]
        header.extend(DATA["blend_shapes"]["names"])
        wr.writerow(header)

        for block in DATA["blend_shapes"]["values"]:
            row = [block[1], block[0]]
            row.extend(map(lambda x: str(x), block[2]))

            wr.writerow(row)

    print("Data saved to " + fn)
    exit()


def connect_to_questions_udp(binding_addr="127.0.0.1", listening_port=33444):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(0)  # Non-blocking socket, no flag data - your problem
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1500)  # No buffer. We take the latest, if present, or nothing.

    print("Connecting to questions...")
    sock.bind((binding_addr, listening_port))
    print("Connected.")

    return sock


def connect_to_fs_udp(binding_addr="127.0.0.1", listening_port=33433):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1500)  # No buffer. We take the latest, if present, or nothing.

    print("Connecting to fs...")
    sock.bind((binding_addr, listening_port))
    print("Connected.")

    return sock


def read_block(sock, fsr, data_dict):
    msg = sock.recv(4096)
    fsr.decode_faceshift_datastream(data_dict, msg)


def read_record_flag(sock, data_dict):
    try:
        msg = sock.recv(4096)

        if int(msg.decode('utf-8')) == RECORD_FLAG_END_SESSION:
            save_and_exit()

        data_dict["record_flag"] = int(msg.decode('utf-8'))
    except socket.error as e:
        if e.args[0] == socket.errno.EWOULDBLOCK:
            return  # No flag received but this is "ok"
        raise e


def record():
    signal.signal(signal.SIGINT, fin_handler)

    q_sock = None
    fs_sock = None
    try:
        q_sock = connect_to_questions_udp()
        fs_sock = connect_to_fs_udp()

        fsr = FaceShiftReceiver()

        while True:
            print("Waiting for record flag... ")
            read_record_flag(q_sock, DATA)
            print("Flag received: {}".format(DATA["record_flag"]))

            print("Waiting for fs block... ")
            read_block(fs_sock, fsr, DATA)
            print("Block received")

    except Exception as e:

        if fs_sock is not None:
            fs_sock.close()

        if q_sock is not None:
            q_sock.close()

        raise e


if __name__ == "__main__":
    record()
