import minerl
from collections import deque
from copy import deepcopy
from dataset import Transition
from collections import OrderedDict

def transfer_data_to_buffer(environment, action_mapper, buffer, data_directory, stack_size=3, filter_success=True, max_steps=None, max_reward_value=256.0, is_test_mode=False):
    print(f"\nTransferring data from {environment}\n")

    is_treechop_environment = environment == "MineRLTreechop-v0"

    def check_success(sample):
        return sample[-1]['success'] and (max_steps is None or sample[-1]['duration_steps'] < max_steps)

    def check_no_op(sample):
        return action_mapper.get_id(sample[1]) == 0

    def process_and_store(sample, previous_reward):
        reward = sample[2]
        log_sample = previous_reward < 2.0

        if is_treechop_environment:
            for key, value in action_mapper.zero_action.items():
                sample[1].setdefault(key, value)

            sample[0]['equipped_items'] = OrderedDict([('mainhand', OrderedDict([('damage', 0), ('maxDamage', 0), ('type', 0)]))])
            sample[0]["inventory"] = OrderedDict([
                ('coal', 0), ('cobblestone', 0), ('crafting_table', 0), ('dirt', 0), ('furnace', 0), 
                ('iron_axe', 0), ('iron_ingot', 0), ('iron_ore', 0), ('iron_pickaxe', 0), ('log', 0), 
                ('planks', 0), ('stick', 0), ('stone', 0), ('stone_axe', 0), ('stone_pickaxe', 0), 
                ('torch', 0), ('wooden_axe', 0), ('wooden_pickaxe', 0)
            ])

        if reward != 0.0:
            if reward > max_reward_value:
                change_count = -buffer.remove_new_data()
            else:
                buffer.add_sample(sample, log_sample, is_treechop_environment)
                buffer.update_last_reward_index()
                change_count = 1
        else:
            if not check_no_op(sample) or sample[4]:
                buffer.add_sample(sample, log_sample, is_treechop_environment)
                change_count = 1
            else:
                change_count = 0

        return change_count, previous_reward

    minerl_data = minerl.data.make(environment, data_dir=data_directory)
    trajectory_list = minerl_data.get_trajectory_names()
    sample_queue = deque(maxlen=stack_size)
    total_trajectories = 0
    samples_added = 0
    initial_sample_count = buffer.buffer.current_size()

    for traj_index, trajectory in enumerate(trajectory_list):
        for step_index, sample in enumerate(minerl_data.load_data(trajectory, include_metadata=True)):
            if step_index == 0:
                print(sample[-1])

                if filter_success and not check_success(sample):
                    print("Skipping trajectory")
                    break

                total_trajectories += 1
                previous_reward = 0.0

            sample_queue.append(sample)

            if len(sample_queue) == stack_size:
                for i in range(1, stack_size):
                    sample_queue[0][1]['camera'] += sample_queue[i][1]['camera']
                    if sample_queue[i][2] != 0.0:
                        break

                new_samples, previous_reward = process_and_store(sample_queue[0], previous_reward)
                samples_added += new_samples

        if len(sample_queue) > 0:
            for i in range(1, stack_size):
                new_samples, previous_reward = process_and_store(sample_queue[i], previous_reward)
                samples_added += new_samples

            samples_added -= buffer.remove_new_data()
            last_transition = deepcopy(buffer.buffer.buffer[buffer.buffer.current_index - 1])
            buffer.buffer.buffer[buffer.buffer.current_index - 1] = Transition(
                last_transition.state, last_transition.vector,
                last_transition.action, last_transition.reward, False
            )

        sample_queue.clear()
        print(f"{traj_index + 1} / {len(trajectory_list)}, total added: {total_trajectories}")
        assert buffer.buffer.current_size() - initial_sample_count == samples_added

        if is_test_mode and total_trajectories >= 2:
            assert total_trajectories == 2
            break
