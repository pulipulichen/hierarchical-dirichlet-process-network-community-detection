import numpy as np
import matplotlib.pyplot as plt
import re

network_lst = ['yeast', 'GSE', 'facebook', 'powergrid', 'ca-GrQc']
metrics = ['density', 'TPR', 'cut ratio', 'conductance']
labels = ['RW-HDP', 'SIP-LDA', 'Walktrap', 'BCD']
fs = 10 # font size

fig, axes = plt.subplots(nrows=len(network_lst), ncols=len(metrics),
                           sharex=True, sharey=False, figsize=(6,10))


for i, row in enumerate(axes):
	network = network_lst[i]
	rwhdp_f = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.online_rwhdp.out'%(network, network)
	siplda_f = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.siplda.out'%(network, network)
	walktrap_f = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.walktrap.out'%(network, network)
	bcd_f = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.bcd.out'%(network, network)

	density = []
	TPR = []
	cut_ratio = []
	conductance = []

	with open(rwhdp_f) as f:
	    rwhdp_out = f.readlines()
	with open(siplda_f) as f:
	    siplda_out = f.readlines()
	with open(walktrap_f) as f:
	    walktrap_out = f.readlines()
	with open(bcd_f) as f:
	    bcd_out = f.readlines()

	for n in range(len(rwhdp_out)):
	    if rwhdp_out[n] == 'performance\n':
	        rwhdp_out = rwhdp_out[n:]
	        break
	for n in range(len(siplda_out)):
	    if siplda_out[n] == 'performance\n':
	        siplda_out = siplda_out[n:]
	        break
	for n in range(len(walktrap_out)):
	    if walktrap_out[n] == 'performance\n':
	        walktrap_out = walktrap_out[n:]
	        break
	for n in range(len(bcd_out)):
	    if bcd_out[n] == 'performance\n':
	        bcd_out = bcd_out[n:]
	        break

	# density
	tmp = re.split(r'[;,\s\{\}]\s*', rwhdp_out[3])[1:-1]
	density.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', siplda_out[3])[1:-1]
	density.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', walktrap_out[3])[1:-1]
	density.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', bcd_out[3])[1:-1]
	density.append([float(x) for x in tmp])

	# TPR
	tmp = re.split(r'[;,\s\{\}]\s*', rwhdp_out[4])[1:-1]
	TPR.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', siplda_out[4])[1:-1]
	TPR.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', walktrap_out[4])[1:-1]
	TPR.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', bcd_out[4])[1:-1]
	TPR.append([float(x) for x in tmp])

	# cut ratio
	tmp = re.split(r'[;,\s\{\}]\s*', rwhdp_out[5])[2:-1]
	cut_ratio.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', siplda_out[5])[2:-1]
	cut_ratio.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', walktrap_out[5])[2:-1]
	cut_ratio.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', bcd_out[5])[2:-1]
	cut_ratio.append([float(x) for x in tmp])

	# conductance
	tmp = re.split(r'[;,\s\{\}]\s*', rwhdp_out[6])[1:-1]
	conductance.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', siplda_out[6])[1:-1]
	conductance.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', walktrap_out[6])[1:-1]
	conductance.append([float(x) for x in tmp])
	tmp = re.split(r'[;,\s\{\}]\s*', bcd_out[6])[1:-1]
	conductance.append([float(x) for x in tmp])

	metric_lst = []
	metric_lst.append(density)
	metric_lst.append(TPR)
	metric_lst.append(cut_ratio)
	metric_lst.append(conductance)

	for j, cell in enumerate(row):
		cell.boxplot(metric_lst[j], labels=labels)
		if i == 0:
			cell.set_title(metrics[j])
		if j == 0:
			cell.set_ylabel(network_lst[i])
		

plt.tight_layout()
plt.show()