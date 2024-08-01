import numpy as np
from envs import make_env
from algorithm.replay_buffer import Trajectory
from learner.hgg import TrajectoryPool
from envs.utils import get_goal_distance
import torch
class NormalLearner:
	def __init__(self, args):
		self.args = args
		self.device = self.args.device
		self.goal_distance = get_goal_distance(args)
		self.env_List = []
		for i in range(args.episodes):
			self.env_List.append(make_env(args))
		self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
		self.counter = 0
	def learn(self, args, env, env_test, agent, buffer, diffusion_model, aim_discriminator):
		initial_goals = []
		desired_goals = []
		env_goal_temporary_container = []
		for i in range(args.episodes):
			obs = self.env_List[i].reset()
			goal_a = obs['achieved_goal'].copy()
			goal_d = obs['desired_goal'].copy()
			initial_goals.append(goal_a.copy())
			desired_goals.append(goal_d.copy())

		achieved_trajectories = []
		achieved_states = []
		for i in range(args.episodes):
			obs = self.env_List[i].get_obs()


			#Here is the goal thorugh episode
			# self.env_List[i].goal = None
			obs = self.env_List[i].get_obs()
			trajectory = [obs['achieved_goal'].copy()]
			traj_achieved_states = [obs['observation'].copy()]
			current = Trajectory(obs)
			for timestep in range(args.timesteps):
				action = agent.step(obs, explore=True)
				obs, reward, done, info = self.env_List[i].step(action)
				trajectory.append(obs['achieved_goal'].copy())
				traj_achieved_states.append(obs['observation'].copy())
				if timestep==args.timesteps-1: done = True
				current.store_step(action, obs, reward, done)
				if done: break
			env_goal_temporary = diffusion_model.sample_goal(obs)
			env_goal_temporary_container.append(env_goal_temporary)
			achieved_trajectories.append(np.array(trajectory))
			achieved_states.append(np.array(traj_achieved_states))
			buffer.store_trajectory(current)
			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
				agent.target_update()

		selection_trajectory_idx = {}
		for i in range(self.args.episodes):
			if self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>0.01:
				selection_trajectory_idx[i] = True
		for idx in selection_trajectory_idx.keys():
			self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_states[idx].copy())

		sample_buffer = buffer.sample_batch()
		achieved_states_tensor = torch.tensor(np.array(np.array(sample_buffer['obs'])[:,3:6]),dtype=torch.float32).to(self.device)
		desired_states_tensor  = torch.tile(torch.tensor(np.array(desired_goals[0]), dtype=torch.float32).to(self.device),[achieved_states_tensor.shape[0], 1])
		policy_states = torch.cat([achieved_states_tensor,desired_states_tensor], dim=-1)
		achieved_states_tensor_next = torch.tensor(np.array(np.array(sample_buffer['obs_next'])[:,3:6]), dtype=torch.float32).to(self.device)
		policy_next_states = torch.cat([achieved_states_tensor_next,desired_states_tensor], dim=-1)
		target_states = torch.cat([desired_states_tensor + torch.from_numpy(np.random.normal(scale=0.01, size=desired_states_tensor.shape)).float().to(self.device),desired_states_tensor], dim=-1)  # s_g, s_g
		aim_disc_loss, wgan_loss, graph_penalty, min_aim_f_loss = aim_discriminator.optimize_discriminator(target_states, policy_states, policy_next_states)


		diffusion_model.train(agent, aim_discriminator, self.achieved_trajectory_pool, self.counter)
		np.save("log/env_goals" +str(self.counter) + ".npy", np.array(env_goal_temporary_container))
		self.counter += 1


