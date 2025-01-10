import matplotlib.pyplot as plt
import numpy as np
base0_baseplayer_res = np.sum([9, 10, 16, 15, 13])
base1_baseplayer_res = np.sum([11, 9, 7, 8, 7])
base2_baseplayer_res = np.sum([5, 8, 6, 8, 9])
base3_baseplayer_res = np.sum([2, 2, 4, 1, 1])
base4_baseplayer_res = np.sum([5, 1, 7, 3, 4])
base5_baseplayer_res = np.sum([5, 2, 3, 3, 4])
base6_baseplayer_res = np.sum([6, 4, 7, 3, 4])
base7_baseplayer_res = np.sum([3, 2, 5, 4, 5])

base0_monteplayer_res = np.sum([17, 15, 20, 17, 17])
base1_monteplayer_res = np.sum([16, 15, 13, 16, 17])
base2_monteplayer_res = np.sum([15, 13, 13, 15, 15])
base3_monteplayer_res = np.sum([7, 5, 8, 10, 13])
base4_monteplayer_res = np.sum([9, 8, 4, 10, 7])
base5_monteplayer_res = np.sum([9, 4, 4, 2, 7])
base6_monteplayer_res = np.sum([7, 3, 5, 8, 8])
base7_monteplayer_res = np.sum([4, 4, 4, 3, 4])

base0_RLplayer_res = np.sum([16, 14, 15, 13, 14])
base1_RLplayer_res = np.sum([19, 14, 17, 16, 14])
base2_RLplayer_res = np.sum([12, 12, 14, 14, 17])
base3_RLplayer_res = np.sum([19, 20, 15, 17, 18])
base4_RLplayer_res = np.sum([5, 6, 6, 7, 7])
base5_RLplayer_res = np.sum([8, 10, 6, 8, 5])
base6_RLplayer_res = np.sum([8, 5, 7, 9, 10])
base7_RLplayer_res = np.sum([7, 6, 8, 9, 5])

baseplayer = []
for i in range(8):
    baseplayer.append(eval(f"base{i}_baseplayer_res"))
print(baseplayer)

monteplayer = []
for i in range(8):
    monteplayer.append(eval(f"base{i}_monteplayer_res"))
print(monteplayer)

RLplayer  = []
for i in range(8):
    RLplayer.append(eval(f"base{i}_RLplayer_res"))
print(RLplayer)

# Baselines
baselines = [f'baseline{i}' for i in range(8)]

# 创建一个新的figure
plt.figure(figsize=(10, 6))

# 设置柱子的宽度和位置
bar_width = 0.25
r1 = np.arange(len(baseplayer))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# 绘制第一个列表的直方图
plt.bar(r1, baseplayer, color='blue', width=bar_width, edgecolor='grey', label='baseplayer')

# 绘制第二个列表的直方图
plt.bar(r2, monteplayer, color='red', width=bar_width, edgecolor='grey', label='monteplayer')

# 绘制第三个列表的直方图
plt.bar(r3, RLplayer, color='green', width=bar_width, edgecolor='grey', label='RLplayer')

# 添加标题和标签
plt.title('Histogram of different agents Competing with Baselines')
plt.ylabel('win times')
plt.xticks([r + bar_width for r in range(len(baseplayer))], baselines)

# 添加图例
plt.legend()

# 显示图形
plt.savefig("result.png")