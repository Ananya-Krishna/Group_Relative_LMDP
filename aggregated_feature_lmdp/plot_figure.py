import numpy as np
from numba import jit
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9})

# opt = np.asarray([0.99 ** 9 / 10 * (_ + 1) for _ in range(20000)])
#
# result_ada_lsvi_ucb_restart = opt - np.load('result_ada_2.npy')
# result_lsvi_ucb = opt - np.load('result_lsvi_2.npy')
# result_lsvi_ucb_restart = opt - np.load('result_restart_2.npy')
# result_eps_greedy = opt - np.load('result_greedy_2.npy')
# result_random = opt - np.load('result_random_2.npy')

# result_ada_lsvi_ucb_restart = np.load('./ada_runtime_speed20_a.npy')
result_lsvi_ucb = np.load('result_lsvi_300.npy')
# result_lsvi_ucb_restart = np.load('./restart_runtime_speed10g.npy')
result_boovi_5 = np.load('result_boovi_5.npy')
result_boovi_10 = np.load('result_boovi_300_10.npy')
result_boovi_100 = np.load('result_boovi_100.npy')
result_boovi_la = np.load('result_boovi_la_300.npy')
result_eps_greedy = np.load('result_greedy_300.npy')
result_random = np.load('result_random_300.npy')

# result_ada_lsvi_ucb_restart = np.load('./result_speed20_abrupt/result_ada_speed20_a.npy')
# result_lsvi_ucb = np.load('./result_speed20_abrupt/result_lsvi_speed20_a.npy')
# result_lsvi_ucb_restart = np.load('./result_speed20_abrupt/result_restart_speed20_a.npy')
# result_eps_greedy = np.load('./result_speed20_abrupt/result_greedy_speed20_a.npy')
# result_random = np.load('./result_speed20_abrupt/result_random_speed20_a.npy')

t = result_lsvi_ucb.shape[1]
x = [_ for _ in range(1, t + 1)]

result_lsvi_ucb_mean = np.mean(result_lsvi_ucb, axis=0)
result_lsvi_ucb_standard_dev = np.std(result_lsvi_ucb, axis=0)
# result_lsvi_ucb_restart_mean = np.mean(result_lsvi_ucb_restart, axis=0)
# result_lsvi_ucb_restart_standard_dev = np.std(result_lsvi_ucb_restart, axis=0)
result_boovi5_mean = np.mean(result_boovi_5, axis=0)
result_boovi5_standard_dev = np.std(result_boovi_5, axis=0)
result_boovi10_mean = np.mean(result_boovi_10, axis=0)
result_boovi10_standard_dev = np.std(result_boovi_10, axis=0)
result_boovi100_mean = np.mean(result_boovi_100, axis=0)
result_boovi100_standard_dev = np.std(result_boovi_100, axis=0)
result_boovi_la_mean = np.mean(result_boovi_la,axis=0)
result_boovi_la_standard_dev = np.std(result_boovi_la, axis= 0)
result_eps_greedy_mean = np.mean(result_eps_greedy, axis=0)
result_eps_greedy_standard_dev = np.std(result_eps_greedy, axis=0)
result_random_mean = np.mean(result_random, axis=0)
result_random_standard_dev = np.std(result_random, axis=0)

# result_ada_lsvi_ucb_restart_mean = np.mean(result_ada_lsvi_ucb_restart, axis=0)
# result_ada_lsvi_ucb_restart_standard_dev = np.std(result_ada_lsvi_ucb_restart, axis=0)
# result_lsvi_ucb_mean = np.mean(result_lsvi_ucb, axis=0)
# result_lsvi_ucb_standard_dev = np.std(result_lsvi_ucb, axis=0)
# result_lsvi_ucb_restart_mean = np.mean(result_lsvi_ucb_restart, axis=0)
# result_lsvi_ucb_restart_standard_dev = np.std(result_lsvi_ucb_restart, axis=0)
# result_eps_greedy_mean = np.mean(result_eps_greedy, axis=0)
# result_eps_greedy_standard_dev = np.std(result_eps_greedy, axis=0)
# result_random_mean = np.mean(result_random, axis=0)
# result_random_standard_dev = np.std(result_random, axis=0)

plt.plot(x, result_eps_greedy_mean, label='Epsilon-Greedy')
plt.plot(x, result_random_mean, label='Random-Exploration')
plt.plot(x, result_lsvi_ucb_mean, label='LSVI-UCB')
plt.plot(x, result_boovi10_mean, label='BooVI')
# plt.plot(x, result_lsvi_ucb_restart_mean, label='LSVI-UCB-Restart')
# plt.plot(x, result_ada_lsvi_ucb_restart_mean, label='ADA-LSVI-UCB-Restart')

num_sigma = 1
plt.fill_between(x, result_eps_greedy_mean - num_sigma * result_eps_greedy_standard_dev,
                 result_eps_greedy_mean + num_sigma * result_eps_greedy_standard_dev, alpha=.1)
plt.fill_between(x, result_random_mean - num_sigma * result_random_standard_dev,
                 result_random_mean + num_sigma * result_random_standard_dev, alpha=.1)
plt.fill_between(x, result_lsvi_ucb_mean - num_sigma * result_lsvi_ucb_standard_dev,
result_lsvi_ucb_mean + num_sigma * result_lsvi_ucb_standard_dev, alpha=.1)
plt.fill_between(x, result_boovi10_mean - num_sigma * result_boovi10_standard_dev,
                 result_boovi10_mean + num_sigma * result_boovi10_standard_dev, alpha=.1)
# plt.fill_between(x, result_ada_lsvi_ucb_restart_mean - num_sigma * result_ada_lsvi_ucb_restart_standard_dev,
#                  result_ada_lsvi_ucb_restart_mean + num_sigma * result_ada_lsvi_ucb_restart_standard_dev, alpha=.1)
# plt.fill_between(x, result_lsvi_ucb_restart_mean - num_sigma * result_lsvi_ucb_restart_standard_dev,
#                  result_lsvi_ucb_restart_mean + num_sigma * result_lsvi_ucb_restart_standard_dev, alpha=.1)

# plt.plot(mean)
# plt.fill_between(mean, mean-standard_dev, mean+standard_dev)


# add variance or confidence interval

plt.xlabel('Total timestep')
plt.ylabel('Cumulative reward')
plt.legend()
plt.title('BooVI vs. Baselines')
plt.savefig('Figure 1.png')
plt.savefig('Figure 1.pdf')
plt.show()

plt.plot(x, result_lsvi_ucb_mean, label ='LSVI-UCB')
plt.plot(x, result_boovi5_mean, label='BooVI(N_k=5)')
plt.plot(x, result_boovi10_mean, label='BooVI(N_k=10)')
plt.plot(x, result_boovi100_mean, label='BooVI(N_k=100)')
# plt.plot(x, result_lsvi_ucb_restart_mean, label='LSVI-UCB-Restart')
# plt.plot(x, result_ada_lsvi_ucb_restart_mean, label='ADA-LSVI-UCB-Restart')

num_sigma = 1
plt.fill_between(x, result_lsvi_ucb_mean - num_sigma * result_lsvi_ucb_standard_dev,
                 result_lsvi_ucb_mean + num_sigma * result_lsvi_ucb_standard_dev, alpha=.1)
plt.fill_between(x, result_boovi5_mean - num_sigma * result_boovi5_standard_dev,
                 result_boovi5_mean + num_sigma * result_boovi5_standard_dev, alpha=.1)
plt.fill_between(x, result_boovi10_mean - num_sigma * result_boovi10_standard_dev,
                 result_boovi10_mean + num_sigma * result_boovi10_standard_dev, alpha=.1)
plt.fill_between(x, result_boovi100_mean - num_sigma * result_boovi5_standard_dev,
                 result_boovi100_mean + num_sigma * result_boovi5_standard_dev, alpha=.1)

plt.xlabel('Total timestep')
plt.ylabel('Cumulative reward')
plt.legend()
plt.title('Different Sample Sizes for BooVI')
plt.savefig('Figure 2.png')
plt.savefig('Figure 2.pdf')
plt.show()


plt.plot(x, result_boovi10_mean, label='BooVI')
plt.plot(x, result_boovi_la_mean, label = 'BooVI-Langevin')
# plt.plot(x, result_lsvi_ucb_restart_mean, label='LSVI-UCB-Restart')
# plt.plot(x, result_ada_lsvi_ucb_restart_mean, label='ADA-LSVI-UCB-Restart')

num_sigma = 1
plt.fill_between(x, result_boovi10_mean - num_sigma * result_boovi10_standard_dev,
                 result_boovi10_mean + num_sigma * result_boovi10_standard_dev, alpha=.1)
plt.fill_between(x, result_boovi_la_mean - num_sigma * result_boovi_la_standard_dev,
                 result_boovi_la_mean + num_sigma * result_boovi_la_standard_dev, alpha = .1)
# plt.fill_between(x, result_ada_lsvi_ucb_restart_mean - num_sigma * result_ada_lsvi_ucb_restart_standard_dev,
#                  result_ada_lsvi_ucb_restart_mean + num_sigma * result_ada_lsvi_ucb_restart_standard_dev, alpha=.1)
# plt.fill_between(x, result_lsvi_ucb_restart_mean - num_sigma * result_lsvi_ucb_restart_standard_dev,
#                  result_lsvi_ucb_restart_mean + num_sigma * result_lsvi_ucb_restart_standard_dev, alpha=.1)

# plt.plot(mean)
# plt.fill_between(mean, mean-standard_dev, mean+standard_dev)


# add variance or confidence interval

plt.xlabel('Total timestep')
plt.ylabel('Cumulative reward')
plt.legend()
plt.title('BooVI vs. BooVI-Langevin')
plt.savefig('Figure 3.png')
plt.savefig('Figure 3.pdf')
plt.show()
# plt.rcParams["figure.figsize"] = [30, 18]
# plt.rcParams.update({'font.size': 30})
# ax = plt.axes()
# ax.spines['right'].set_position(("outward", 35))
# ax.spines['top'].set_bounds(0, 221)
# ax.spines['bottom'].set_bounds(0, 221)
# ax.spines['top'].set_bounds(0, 241)
# ax.spines['bottom'].set_bounds(0, 241)
#
#
# # result_ada_lsvi_ucb_restart = np.load('./result_speed20_gradual/ada_spped_speed20_g.npy')
# # result_lsvi_ucb = np.load('./result_speed20_gradual/lsvi_spped_speed20_g.npy')
# # result_lsvi_ucb_restart = np.load('./result_speed20_gradual/restart_spped_speed20_g.npy')
# # result_eps_greedy = np.load('./result_speed20_gradual/greedy_spped_speed20_g.npy')
# # result_random = np.load('./result_speed20_gradual/random_spped_speed20_g.npy')
#
# #result_ada_lsvi_ucb_restart = np.load('./ada_spped_speed20_a.npy')
# result_lsvi_ucb = np.load('./lsvi_spped_speed10g.npy')
# #result_lsvi_ucb_restart = np.load('./restart_spped_speed10g.npy')
# result_boovi = np.load('./result_boovi.npy')
# result_eps_greedy = np.load('./greedy_spped_speed10g.npy')
# result_random = np.load('./random_spped_speed10g.npy')
#
# #result_ada_lsvi_ucb_restart_mean = np.mean(result_ada_lsvi_ucb_restart, axis=0)
# # result_ada_lsvi_ucb_restart_standard_dev = np.std(result_ada_lsvi_ucb_restart, axis=0)
# result_lsvi_ucb_mean = np.mean(result_lsvi_ucb, axis=0)
# result_lsvi_ucb_standard_dev = np.std(result_lsvi_ucb, axis=0)
# # result_lsvi_ucb_restart_mean = np.mean(result_lsvi_ucb_restart, axis=0)
# # result_lsvi_ucb_restart_standard_dev = np.std(result_lsvi_ucb_restart, axis=0)
# result_boovi_mean = np.mean(result_boovi, axis= 0)
# result_boovi_standard_dev = np.std(result_boovi, axis= 0)
# result_eps_greedy_mean = np.mean(result_eps_greedy, axis=0)
# result_eps_greedy_standard_dev = np.std(result_eps_greedy, axis=0)
# result_random_mean = np.mean(result_random, axis=0)
# result_random_standard_dev = np.std(result_random, axis=0)
#
# x = ['UCB', 'Epsilon-Greedy', 'LSVI-UCB-Restart', 'Random-Exploration', 'BooVI']
#
# x_pos = [i for i, _ in enumerate(x)]
#
# result_mean = [result_lsvi_ucb_mean, result_eps_greedy_mean,  result_random_mean, result_boovi_mean]
# result_standard_dev = [result_lsvi_ucb_standard_dev, result_eps_greedy_standard_dev,
#                        result_random_standard_dev, result_boovi_standard_dev]
#
# plt.barh(x_pos, result_mean, xerr=result_standard_dev, align='center', alpha=.5, ecolor='black', capsize=5)
# # ax.set_yticks(x_pos)
#
# # y = ['ADA-LSVI-UCB-Restart', 'LSVI-UCB-Restart', 'LSVI-UCB', 'Epsilon-Greedy', 'Random']
# # for index, value in enumerate(y):
# #    plt.text(value, index, str(value))
# # plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)
#
# for i, v in enumerate(range(5)):
#     plt.text(result_mean[i] + result_standard_dev[i] + 2, i - .05,
#              str(round(result_mean[i] + result_standard_dev[i], 2)), color='purple')
#
# plt.yticks(x_pos, x, rotation=45)
# # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
# # plt.yaxis.grid(True)
# # plt.spines["right"].set_visible(False)
#
#
# # plt.ylabel('Total timestep')
# plt.xlabel('Time (seconds)')
# plt.title('Gradual change')
# plt.savefig('runtime.png')
# plt.savefig('runtime.pdf')
# plt.show()
#
#

#
# result_ada_lsvi_ucb_restart_g = np.load('./result_speed20_gradual/ada_runtime_speed20_g.npy')
# result_lsvi_ucb_g = np.load('./result_speed20_gradual/lsvi_runtime_speed20_g.npy')
# result_lsvi_ucb_restart_g = np.load('./result_speed20_gradual/restart_runtime_speed20_g.npy')
# result_eps_greedy_g = np.load('./result_speed20_gradual/greedy_runtime_speed20_g.npy')
# result_random_g = np.load('./result_speed20_gradual/random_runtime_speed20_g.npy')
#
# result_ada_lsvi_ucb_restart_mean_g = np.mean(result_ada_lsvi_ucb_restart_g, axis=0)
# result_ada_lsvi_ucb_restart_standard_dev_g = np.std(result_ada_lsvi_ucb_restart_g, axis=0)
# result_lsvi_ucb_mean_g = np.mean(result_lsvi_ucb_g, axis=0)
# result_lsvi_ucb_standard_dev_g = np.std(result_lsvi_ucb_g, axis=0)
# result_lsvi_ucb_restart_mean_g = np.mean(result_lsvi_ucb_restart_g, axis=0)
# result_lsvi_ucb_restart_standard_dev_g = np.std(result_lsvi_ucb_restart_g, axis=0)
# result_eps_greedy_mean_g = np.mean(result_eps_greedy_g, axis=0)
# result_eps_greedy_standard_dev_g = np.std(result_eps_greedy_g, axis=0)
# result_random_mean_g = np.mean(result_random_g, axis=0)
# result_random_standard_dev_g = np.std(result_random_g, axis=0)
#
# result_ada_lsvi_ucb_restart_a = np.load('./result_speed20_abrupt/ada_runtime_speed20_a.npy')
# result_lsvi_ucb_a = np.load('./result_speed20_abrupt/lsvi_runtime_speed20_a.npy')
# result_lsvi_ucb_restart_a = np.load('./result_speed20_abrupt/restart_runtime_speed20_a.npy')
# result_eps_greedy_a = np.load('./result_speed20_abrupt/greedy_runtime_speed20_a.npy')
# result_random_a = np.load('./result_speed20_abrupt/random_runtime_speed20_a.npy')
#
# result_ada_lsvi_ucb_restart_mean_a = np.mean(result_ada_lsvi_ucb_restart_a, axis=0)
# result_ada_lsvi_ucb_restart_standard_dev_a = np.std(result_ada_lsvi_ucb_restart_a, axis=0)
# result_lsvi_ucb_mean_a = np.mean(result_lsvi_ucb_a, axis=0)
# result_lsvi_ucb_standard_dev_a = np.std(result_lsvi_ucb_a, axis=0)
# result_lsvi_ucb_restart_mean_a = np.mean(result_lsvi_ucb_restart_a, axis=0)
# result_lsvi_ucb_restart_standard_dev_a = np.std(result_lsvi_ucb_restart_a, axis=0)
# result_eps_greedy_mean_a = np.mean(result_eps_greedy_a, axis=0)
# result_eps_greedy_standard_dev_a = np.std(result_eps_greedy_a, axis=0)
# result_random_mean_a = np.mean(result_random_a, axis=0)
# result_random_standard_dev_a = np.std(result_random_a, axis=0)
#
# result_mean_a = [result_lsvi_ucb_mean_a, result_eps_greedy_mean_a, result_ada_lsvi_ucb_restart_mean_a,
#                  result_lsvi_ucb_restart_mean_a, result_random_mean_a]
# result_standard_dev_a = [result_lsvi_ucb_standard_dev_a, result_eps_greedy_standard_dev_a,
#                          result_ada_lsvi_ucb_restart_standard_dev_a, result_lsvi_ucb_restart_standard_dev_a,
#                          result_random_standard_dev_a]
#
# result_mean_g = [result_lsvi_ucb_mean_g, result_eps_greedy_mean_g, result_ada_lsvi_ucb_restart_mean_g,
#                  result_lsvi_ucb_restart_mean_g, result_random_mean_g]
# result_standard_dev_g = [result_lsvi_ucb_standard_dev_g, result_eps_greedy_standard_dev_g,
#                          result_ada_lsvi_ucb_restart_standard_dev_g, result_lsvi_ucb_restart_standard_dev_g,
#                          result_random_standard_dev_g]
#
# barWidth = 0.15
#
# x = ['LSVI-UCB', 'Epsilon-Greedy', 'ADA-LSVI-UCB-Restart', 'LSVI-UCB-Restart', 'Random-Exploration']
# # fig = plt.subplots(figsize=(12, 8))
#
# # set height of bar
#
#
# # Set position of bar on X axis
# # br1 = np.arange(len(x))
# # br2 = [x + barWidth for x in br1]
# # br3 = [x + barWidth for x in br2]
# #
# # # Make the plot
# # plt.bar(br1, IT, color='r', width=barWidth,
# #         edgecolor='grey', label='IT')
# # plt.bar(br2, ECE, color='g', width=barWidth,
# #         edgecolor='grey', label='ECE')
# # plt.bar(br3, CSE, color='b', width=barWidth,
# #         edgecolor='grey', label='CSE')
# #
# # # Adding Xticks
# # plt.xlabel('Branch', fontweight='bold')
# # plt.ylabel('Students passed', fontweight='bold')
# # plt.xticks([r + barWidth for r in range(len(IT))],
# #            ['2015', '2016', '2017', '2018', '2019'])
# #
# # plt.show()
#
#
# x = ['LSVI-UCB', 'Epsilon-Greedy', 'ADA-LSVI-UCB-Restart', 'LSVI-UCB-Restart', 'Random-Exploration']
#
# xpos = [[_ * (barWidth + 0.0), 1 + _ * (barWidth + 0.0)] for _ in range(5)]
#
# for i in range(5):
#     plt.bar(xpos[i], [result_mean_a[i], result_mean_g[i]], width=barWidth,
#             yerr=[result_standard_dev_a[i], result_standard_dev_g[i]], align='center', alpha=.5, ecolor='black',
#             capsize=5, label=x[i])
#
# # plt.bar(x_pos1, result_mean_a, width=barWidth, yerr=result_standard_dev_a, align='center', alpha=.5, ecolor='black', capsize=5)
#
# # plt.bar(x_pos2, result_mean_a, width=barWidth, yerr=result_standard_dev_a, align='center', alpha=.5, ecolor='black', capsize=5)
# # ax.set_yticks(x_pos)
#
# # y = ['ADA-LSVI-UCB-Restart', 'LSVI-UCB-Restart', 'LSVI-UCB', 'Epsilon-Greedy', 'Random']
# # for index, value in enumerate(y):
# #    plt.text(value, index, str(value))
# # plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)
#
# textloc1 = [0.15 - .07, 0.15 * 2 - .07, 0.15 * 3 - .06, 0.15 * 4 - .03, 0.15 * 5 - .06]
# textloc2 = [1 + 0.15 - .07, 1 + 0.15 * 2 - .07, 1 + 0.15 * 3 - .06, 1 + 0.15 * 4 - .07 , 1 + 0.15 * 5 - .05]
#
# for i, v in enumerate(range(5)):
#     plt.text(textloc1[i] - 0.15, result_mean_a[i] + result_standard_dev_a[i] + 2,
#              str(round(result_mean_a[i] + result_standard_dev_a[i], 1)), color='purple')
#
# for i, v in enumerate(range(5)):
#     plt.text(textloc2[i] - 0.15, result_mean_g[i] + result_standard_dev_g[i] + 2,
#              str(round(result_mean_g[i] + result_standard_dev_g[i], 1)), color='purple')
# plt.text(0.07, -15, 'Abrupt Change', color='black')
# plt.text(1.07, -15, 'Gradual Change', color='black')
# plt.xticks([])
# plt.xlim(right=2.2)
# # plt.yticks(x_pos, x, rotation=45)
# # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
# # plt.yaxis.grid(True)
# # plt.spines["right"].set_visible(False)
#
#
# # plt.ylabel('Total timestep')
# plt.ylabel('Time (seconds)')
# plt.title('Gradual change')
# plt.legend()
# #plt.show()
# plt.savefig('runtime.png')
# plt.savefig('runtime.pdf')
