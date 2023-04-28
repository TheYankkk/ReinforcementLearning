import tensorflow as tf
import numpy as np
import gym
import random
from collections import  deque

num_episodes=500 #游戏训练总epi数
num_exploration_episode=100
max_len_episode=1000
batch_size=32
learning_rate=1e-3
gamma=1. #折扣因子
initial_epsilon=1. #探索起始时的探索率
final_epsilon=0.01 #探索终止时的探索率


class QNetwork(tf.keras.Model):
    def __int__(self):
        super().__init__()
        self.dense1=tf.keras.layers.Dense(units=24,activation=tf.nn.relu)
        self.dense2=tf.keras.layers.Dense(units=24,activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)

    def call(self,inputs):
        x=self.dense1(inputs)
        x=self.dense2(x)
        x=self.dense3(x)
        return x

    def predict(self,inputs):
        q_values=self(inputs)
        return tf.argmax(q_values,axis=-1)




if __name__=='__main__':
    env=gym.make('CartPole-v1')
    model=QNetwork
