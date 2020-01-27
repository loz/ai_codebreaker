from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import codebreaker as cb
import numpy as np

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory


class Q4Network(network.Network):
  def __init__(self,
               input_tensor_spec,
               action_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=(75, 40),
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               batch_squash=True,
               dtype=tf.float32,
               name='Q4Network'):

    num_actions = action_spec.maximum - action_spec.minimum + 1
    encoder_input_tensor_spec = input_tensor_spec

    encoder = encoding_network.EncodingNetwork(
        encoder_input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=batch_squash,
        dtype=dtype)

    q_value_layer = tf.keras.layers.Dense(
      num_actions*4,
      activation=None,
      kernel_initializer=tf.compat.v1.initializers.random_uniform(
        minval=-0.03, maxval=0.03),
      bias_initializer=tf.compat.v1.initializers.constant(-0.2),
      dtype=dtype)

    super(Q4Network, self).__init__(
      input_tensor_spec=input_tensor_spec,
      state_spec=(),
      name=name)

    self._encoder = encoder
    self._q_value_layer = q_value_layer

  def call(self, observation, step_type=None, network_state=(), training=False):
    """Runs the given observation through the network.
    Args:
      observation: The observation to provide to the network.
      step_type: The step type for the given observation. See `StepType` in
        time_step.py.
      network_state: A state tuple to pass to the network, mainly used by RNNs.
      training: Whether the output is being used for training.
    Returns:
      A tuple `(logits, network_state)`.
    """
    state, network_state = self._encoder(
        observation, step_type=step_type, network_state=network_state,
        training=training)
    q_value = self._q_value_layer(state, training=training)
    return q_value, network_state

class D4qnAgent(dqn_agent.DqnAgent):
  def _check_action_spec(self, action_spec):
    pass

tf.compat.v1.enable_v2_behavior()

num_iterations = 20000
inital_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-3
log_interval = 200
num_eval_episodes = 10
eval_interval = 1000

env = cb.RLEnv()
env.reset()
env.render()

print("=====================")
print("Observation Spec:")
print(env.time_step_spec().observation)

print("Reward Spec:")
print(env.time_step_spec().reward)

print("Action Spec:")
print(env.action_spec())

time_step = env.reset()
print("Time step:")
print(time_step)

action = [1,1,2,2]
next_time_step = env.step(action)
print("Next time step")
print(next_time_step)

train_py_env = cb.RLEnv()
eval_py_env = cb.RLEnv()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env =  tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (100,)
q_net = Q4Network(
  train_env.observation_spec(),
  train_env.action_spec(),
  fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = D4qnAgent(
  train_env.time_step_spec(),
  train_env.action_spec(),
  q_network=q_net,
  optimizer=optimizer,
  td_errors_loss_fn=common.element_wise_squared_loss,
  train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

time_step = env.reset()
print(time_step)
print("------")
action = random_policy.action(time_step)
print(action)

def compute_avg_return(environment, policy, num_episodes=10):
  total_return = 0.0
  for _ in range(num_episodes):
    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

avg = compute_avg_return(eval_env, random_policy, num_eval_episodes)
print(avg)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

print(agent.collect_data_spec)

def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, steps=100)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy, replay_buffer)

  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
