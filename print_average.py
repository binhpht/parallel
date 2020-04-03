#!/usr/bin/python3
import re
import numpy as np

f = open("parallel_stdout.txt")

n_pre = 2
n_warmup = 2
for r in range(n_pre + n_warmup):
	f.readline()

time_list = []
for text in f:
	time_list.append(np.asarray(list(map(int, re.findall("\d+", text))))[None, :])
time = np.concatenate(time_list, 0)

print(time)

print("MEAN")
mean_time = np.round(np.mean(time, 0))
for r in range(3):
	print("{:4d}".format(int(mean_time[r])), end=", ")
print("")
#print(np.round(np.mean(time, 0)))
