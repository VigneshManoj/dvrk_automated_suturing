from tkinter import *
import rosbag
from datetime import datetime
import subprocess
import signal
import shlex

import rospy
from std_msgs.msg import Time
from std_msgs.msg import Empty
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import JointState

import time

import os

import threading


class UserStudy:
    def __init__(self):

        rospy.init_node('user_study_data')
        self._time_pub = rospy.Publisher('/ambf/env/user_study_time', Time, queue_size=1)
        self._dvrk_on_pub = rospy.Publisher('/dvrk/console/power_on', Empty, queue_size=1)
        self._dvrk_off_pub = rospy.Publisher('/dvrk/console/power_off', Empty, queue_size=1)
        self._dvrk_home_pub = rospy.Publisher('/dvrk/console/home', Empty, queue_size=1)
        self._time_msg = 0
        self._start_time = 0
        self._active = False
        self._time_pub_thread = 0
        self._my_bag = 0
        self._traj_count = 0
        self._traj_bag_process = None

        self._PSM2_JS = None
        self._MTML_Cart_States = None
        self._PSM2_Cart_States = None

        # self._PSM2_cart_sub = self.sub_temp_pub_psm_cart = rospy.Subscriber('/dvrk/PSM2/position_cartesian_current'.format(self._traj_count),
        #                                          PoseStamped, self.PSM2CartCb)
        # self._PSM2_js_sub = self.sub_temp_pub_psm_js = rospy.Subscriber('/dvrk/PSM2/current_state_{}'.format(self._traj_count), JointState,
        #                                        self.PSM2JointStatesCb)
        # self._MTML_cart_sub = self.sub_temp_pub_mtm_cart = rospy.Subscriber('/dvrk/MTML/position_cartesian_current'.format(self._traj_count),
        #                                          PoseStamped, self.MTMLCartCb)

        self._topic_names = ["/dvrk/ECM/current_state",
                             "/dvrk/ECM/error",
                             "/dvrk/ECM/jacobian_body",
                             "/dvrk/ECM/jacobian_spatial",
                             "/dvrk/ECM/position_cartesian_current",
                             "/dvrk/ECM/joint_velocity_ratio",
                             "/dvrk/ECM/set_wrench_body",
                             "/dvrk/MTML/current_state",
                             "/dvrk/MTML/error",
                             "/dvrk/MTML/state_joint_current",
                             "/dvrk/MTML/jacobian_body",
                             "/dvrk/MTML/jacobian_spatial",
                             "/dvrk/MTML/position_cartesian_current",
                             "/dvrk/MTML/joint_velocity_ratio",
                             "/dvrk/MTML/set_wrench_body",
                             "/dvrk/MTMR/current_state",
                             "/dvrk/MTMR/error",
                             "/dvrk/MTMR/jacobian_body",
                             "/dvrk/MTMR/jacobian_spatial",
                             "/dvrk/MTMR/position_cartesian_current",
                             "/dvrk/MTMR/joint_velocity_ratio",
                             "/dvrk/MTMR/set_wrench_body",
                             "/dvrk/PSM1/current_state",
                             "/dvrk/PSM1/current_state",
                             "/dvrk/PSM1/jacobian_body",
                             "/dvrk/PSM1/jacobian_spatial",
                             "/dvrk/PSM1/joint_velocity_ratio",
                             "/dvrk/PSM1/position_cartesian_current",
                             "/dvrk/PSM1/set_wrench_body",
                             "/dvrk/PSM2/current_state",
                             "/dvrk/PSM2/current_state",
                             "/dvrk/PSM2/jacobian_body",
                             "/dvrk/PSM2/jacobian_spatial",
                             "/dvrk/PSM2/joint_velocity_ratio",
                             "/dvrk/PSM2/position_cartesian_current",
                             "/dvrk/PSM2/set_wrench_body",
                             "/dvrk/PSM2/state_joint_current",
                             "/dvrk/PSM2/state_jaw_current",
                             "/dvrk/footpedals/coag",
                             "/dvrk/footpedals/clutch",
                             "/dvrk/MTMR/set_wrench_body"]

        self._topic_names_str = ""
        self._rosbag_filepath = 0
        self._rosbag_process = 0

        for name in self._topic_names:
            self._topic_names_str = self._topic_names_str + ' ' + name

    def call(self, filepath=None):
        if self._rosbag_filepath is 0:
            self._active = True
            self._start_time = rospy.Time.now()
            self._time_pub_thread = threading.Thread(target=self.time_pub_thread_func)
            self._time_pub_thread.start()
            print("Start Recording ROS Bag")
            date_time_str = str(datetime.now()).replace(' ', '_')
            self._rosbag_filepath = '/home/vignesh/Desktop/user_study_data/' + date_time_str
            # self._rosbag_filepath = './user_study_data/' + e1.get() + '_' + b1.get() + '_' + date_time_str
            command = "rosbag record -O" + ' ' + self._rosbag_filepath + self._topic_names_str
            print "Running Command", command
            command = shlex.split(command)
            self._rosbag_process = subprocess.Popen(command)
        elif filepath is not None:
            print "Start Recording Trajectories"
            fp = '/home/vignesh/Desktop/user_study_data/' + filepath
            command = "rosbag record -O" + ' ' + fp + self._topic_names_str
            print "Starting trajectory id: {} recording".format(self._traj_count)
            command = shlex.split(command)
            self._traj_bag_process = subprocess.Popen(command,shell=True)
        else:
            print "Already recording a ROSBAG file, please save that first before starting a new record"

    def save(self, fp=None):

        if self._rosbag_filepath is not 0:

            # self._active = False
            filepath = None
            if fp is not None:
                filepath = fp
                print "Stopping id: {} recording".format(self._traj_count)
                self._traj_count += 1
                self._traj_bag_process.kill()
                return
            else:
                filepath= self._rosbag_filepath
                self._rosbag_filepath = 0

            node_prefix = "/record"
            # Adapted from http://answers.ros.org/question/10714/start-and-stop-rosbag-within-a-python-script/
            list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
            list_output = list_cmd.stdout.read()
            retcode = list_cmd.wait()
            assert retcode == 0, "List command returned %d" % retcode
            for node_name in list_output.split("\n"):
                if node_name.startswith(node_prefix):
                    os.system("rosnode kill " + node_name)

            print("Saved As:", filepath, ".bag")
            self._active = False

        else:
            print("You should start recording first before trying to save")
        return

    def time_pub_thread_func(self):

        while self._active:
            self._time_msg = rospy.Time.now() - self._start_time
            self._time_pub.publish(self._time_msg)
            time.sleep(0.05)

    # def cart_logging_thread(self):
    #     pub_temp_pub_psm_cart = rospy.Publisher('/dvrk/PSM2/position_cartesian_current_{}'.format(self._traj_count),
    #                                             PoseStamped, queue_size=10)
    #     pub_temp_pub_psm_js = rospy.Publisher('/dvrk/PSM2/current_state_{}'.format(self._traj_count), JointState,
    #                                           queue_size=10)
    #     pub_temp_pub_mtm_cart = rospy.Publisher('/dvrk/MTML/position_cartesian_current'.format(self._traj_count),
    #                                             PoseStamped, queue_size=10)
    #     PSM2JSMsg = JointState()
    #     PSM2CartMsg = PoseStamped()
    #     MTMLCartMsg = PoseStamped()
    #
    #     while self._active:


    def dvrk_power_on(self):
        self._dvrk_on_pub.publish(Empty())
        time.sleep(0.1)

    def dvrk_power_off(self):
        self._dvrk_off_pub.publish(Empty())
        time.sleep(0.1)

    def dvrk_home(self):
        self._dvrk_home_pub.publish(Empty())
        time.sleep(0.1)

    def traj_start_recording(self):
        self.call('exp_{}'.format(self._traj_count))
        time.sleep(0.1)

    def traj_stop_recording(self):
        self.save('exp_{}'.format(self._traj_count))
        time.sleep(0.1)


study = UserStudy()

master = Tk()
master.title("DVRK AUTOMATED")
width = 550
height = 600
master.geometry(str(width)+'x'+str(height))
Label(master, text='Human Subject Name').grid(row=0)

e1 = Entry(master)
e1.grid(row=0, column=1)

b1 = StringVar()

rb1 = Radiobutton(master,  text="TRAINING", padx=20, variable=b1, value='TRAINING')

rb2 = Radiobutton(master, text="1a SINGLE", padx=20, variable=b1, value='1A_SINGLE')
rb3 = Radiobutton(master, text="2a SISO", padx=20, variable=b1, value='2A_SISO')
rb4 = Radiobutton(master, text="3a SIAO", padx=20, variable=b1, value='3A_SIAO')

rb5 = Radiobutton(master, text="1b SINGLE", padx=20, variable=b1, value='1B_SINGLE')
rb6 = Radiobutton(master, text="2b SISO", padx=20, variable=b1, value='2B_SISO')
rb7 = Radiobutton(master, text="3b SIAO", padx=20, variable=b1, value='3B_SIAO')

# Set Default Value

b1.set('TRAINING')

button_start = Button(master, text="Start Record", bg="green", fg="white", height=8, width=20, command=study.call)
button_stop = Button(master, text="Stop Record (SAVE)", bg="red", fg="white", height=8, width=20, command=study.save)
button_destroy = Button(master, text="Close App", bg="black", fg="white", height=8, width=20, command=master.destroy)

button_start_traj_recording = Button(master, text="TRAJ START", bg="green", fg="white", height=4, width=10, command=study.traj_start_recording)
button_stop_traj_recording = Button(master, text="TRAJ STOP", bg="red", fg="white", height=4, width=10, command=study.traj_stop_recording)


button_on = Button(master, text="DVRK ON", bg="green", fg="white", height=4, width=10, command=study.dvrk_power_on)
button_off = Button(master, text="DVRK OFF", bg="red", fg="white", height=4, width=10, command=study.dvrk_power_off)
button_home = Button(master, text="DVRK HOME", bg="purple", fg="white", height=4, width=10, command=study.dvrk_home)

rb1.grid(row=10, column=0)

rb2.grid(row=20, column=0)
rb3.grid(row=30, column=0)
rb4.grid(row=40, column=0)

rb5.grid(row=50, column=0)
rb6.grid(row=60, column=0)
rb7.grid(row=70, column=0)


button_on.grid(row=20, column=1)
button_off.grid(row=40, column=1)
button_home.grid(row=60, column=1)

button_start.grid(row=20, column=2)
button_stop.grid(row=40, column=2)
button_destroy.grid(row=60, column=2)

master.mainloop()