# %%

revised_integrals = np.array(
[[1.0000, 0.4004, 0.6875, 0.9941, 0.8437, 0.9258, 0.6877, 0.5293, 0.7012, 0.5801, 0.9121, 0.7109],
[0.9961, 0.5124, 0.7109, 0.9941, 0.6548, 0.9296, 0.6624, 0.5527, 0.7050, 0.5820, 0.9062, 0.7402],
[0.9980, 0.2930, 0.6602, 0.9883, 0.3408, 0.9043, 0.6624, 0.5254, 0.7012, 0.5293, 0.9043, 0.6992],
[0.9863, 0.2930, 0.6992, 0.9987, 0.6710, 0.9238, 0.6817, 0.5156, 0.7070, 0.5898, 0.8984, 0.7148],
[0.9922, 0.4219, 0.6992, 0.9121, 0.8260, 0.9219, 0.6624, 0.5254, 0.6992, 0.5898, 0.9121, 0.4238],
[0.9961, 0.3671, 0.7266, 0.9785, 0.8238, 0.9307, 0.6624, 0.5293, 0.7031, 0.5781, 0.8916, 0.6943],
[0.9941, 0.3320, 0.7148, 0.9980, 0.6475, 0.9355, 0.6989, 0.5273, 0.6934, 0.5820, 0.9102, 0.7246],
[0.9922, 0.3164, 0.7363, 0.9941, 0.7593, 0.9238, 0.7363, 0.5381, 0.6895, 0.5762, 0.9102, 0.7363],
[0.9961, 0.3750, 0.6855, 0.9980, 0.7915, 0.9414, 0.6871, 0.5332, 0.7090, 0.5879, 0.8867, 0.7129],
[0.9941, 0.3066, 0.7266, 0.9766, 0.7972, 0.9043, 0.6624, 0.5215, 0.7090, 0.6387, 0.9004, 0.6758],
[0.9941, 0.3027, 0.6777, 0.9902, 0.4991, 0.9307, 0.6672, 0.5098, 0.6953, 0.5762, 0.8916, 0.6387],
[0.9805, 0.2675, 0.6973, 0.9902, 0.8971, 0.9307, 0.6624, 0.4883, 0.6836, 0.5449, 0.9082, 0.6943],]
)*100

names = ['bfs', "dfs", "topological_sort", "articulation_points", "bridges", "strongly_connected_components", "mst_kruskal", "mst_prim", "dijkstra", "bellman_ford", 'dag_shortest_paths', "floyd_warshall"]

revised_integrals = {names[i]: {names[j]: revised_integrals[i][j] for j in range(len(names))} for i in range(len(names))}

# Minimize Error rather than Maximize Accuracy
for key in revised_integrals:
	for task in revised_integrals[key]:
		revised_integrals[key][task] = 100.0 - revised_integrals[key][task]

# %%
import math 
import numpy as np

def gen_task_combinations(tasks, rtn, index, path, path_dict):
  if index >= len(tasks):
    return 

  for i in range(index, len(tasks)):
    cur_task = tasks[i]
    new_path = path
    new_dict = {k:v for k,v in path_dict.items()}
    
    # Building from a tree with two or more tasks...
    if new_path:
      new_dict[cur_task] = 0.
      for prev_task in path_dict:
        new_dict[prev_task] += revised_integrals[prev_task][cur_task]
        new_dict[cur_task] += revised_integrals[cur_task][prev_task]
      new_path = '{}|{}'.format(new_path, cur_task)
      rtn[new_path] = new_dict
    else: # First element in a new-formed tree
      new_dict[cur_task] = 0.
      new_path = cur_task

    gen_task_combinations(tasks, rtn, i+1, new_path, new_dict)

    # Fix single-task accuracy since dfs is finished for this task.
    if '|' not in new_path:
      new_dict[cur_task] = revised_integrals[cur_task][cur_task]
      rtn[new_path] = new_dict

# %%
rtn = {}
tasks = list(revised_integrals.keys())
num_tasks = len(tasks)
task_combinations = gen_task_combinations(tasks=tasks, rtn=rtn, index=0, path='', path_dict={})

# Normalize by the number of times the accuracy of any given element has been summed. 
# i.e. (a,b,c) => [acc(a|b) + acc(a|c)]/2 + [acc(b|a) + acc(b|c)]/2 + [acc(c|a) + acc(c|b)]/2
for group in rtn:
  if '|' in group:
    for task in rtn[group]:
      rtn[group][task] /= (len(group.split('|')) - 1)

print(rtn)
assert(len(rtn.keys()) == 2**len(revised_integrals.keys()) - 1)
rtn_tup = [(key,val) for key,val in rtn.items()]

# %%
def select_groups(index, cur_group, best_group, best_val, splits):
  # Check if this group covers all tasks.
  task_set = set()
  for group in cur_group:
    for task in group.split('|'): task_set.add(task)
  if len(task_set) == num_tasks:
    best_tasks = {task:1e6 for task in task_set}
    
    # Compute the per-task best scores for each task and average them together.
    for group in cur_group:
      for task in cur_group[group]:
        # Minimize error.
        best_tasks[task] = min(best_tasks[task], cur_group[group][task])
    group_avg = np.mean(list(best_tasks.values()))
    
    # Compare with the best grouping seen thus far.
    if group_avg < best_val[0]:
      print(cur_group)
      best_val[0] = group_avg
      best_group.clear()
      for entry in cur_group:
        best_group[entry] = cur_group[entry]
  
  # Base case.
  if len(cur_group.keys()) == splits:
    return

  # Back to combinatorics 
  for i in range(index, len(rtn_tup)):
    selected_group, selected_dict = rtn_tup[i]

    new_group = {k:v for k,v in cur_group.items()}
    new_group[selected_group] = selected_dict

    if len(new_group.keys()) <= splits:
      select_groups(i + 1, new_group, best_group, best_val, splits)

selected_group = {}
selected_val = [100000000]
select_groups(index=0, cur_group={}, best_group=selected_group, best_val=selected_val, splits=4)
print(list(selected_group.keys()))
print(selected_val)


