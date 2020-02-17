import numpy as np
import gym
from gym import spaces
import rospy
from std_msgs.msg import String


def __init__(self):


    self._observation = []
    self.action_space = spaces.Box(-1., 1., shape=(self.action_space_dim,), dtype='float32')
    self.observation_space = spaces.Box(-1000.0, 1000.0, shape=(self.action_space_dim,), dtype='float32')
    self._seed()


def _step(self, action):
    self._assign_throttle(action)
    self._observation = self._compute_observation()
    reward = self._compute_reward()
    done = self._compute_done()

    self._envStepCounter += 1

    return np.array(self._observation), reward, done, {}


def talker():
    pub = rospy.Publisher('psm/baselink', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass