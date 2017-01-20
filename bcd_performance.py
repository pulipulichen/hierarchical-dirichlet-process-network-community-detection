# BCD performance

from graph import np, Graph

# hyperparameter
# ...


network = 'ca-CondMat'
f_net = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.elst.csv'%(network, network)
f_com = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.bcd.com'%(network, network)
f_out = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.bcd.out'%(network, network)
f_ref = ''


print('loading graph...')
graph = Graph(f_net, 'edge list', directed=False, weighted=False, memory_control=True)
graph.read_community(f_com, algo='BCD')


print("\nmetrics:")
print("number of communities: %d" %len(graph.communities))

graph.get_modularity()
print("modularity: %s" %graph.modularity)

graph.get_density()
print('density:')
print(np.sort(graph.density))

graph.get_TPR()
print('TPR:')
print(np.sort(graph.TPR))

graph.get_cut_ratio()
print('cut ratio:')
print(np.sort(graph.cut_ratio))

graph.get_conductance()
print('conductance:')
print(np.sort(graph.conductance))

graph.perplexity = 'NA'
print('perplexity: NA')

if len(f_ref) == 0:
	graph.F_measure = graph.ARI = graph.NMI = 'NA'
else:
	ref_partition = np.loadtxt(f_ref)
	graph.get_partition_intersections(ref_partition)
	graph.get_F_measure()
	graph.get_ARI()
	graph.get_NMI()
	print('F_measure: %s\nAdjusted Rand Index: %s\nNormalized Mutual Information: %s' 
	                                          %(graph.F_measure, graph.ARI, graph.NMI))


# save results
with open(f_out, 'w') as f:
	f.write('network statistics\n')
	f.write('nodes: %s\nedges: %s\ndirected: %s\n'%(graph.n, graph.m, graph.directed))

	f.write('\nmodel parameters\nNA\n')

	f.write('\npartition\n')
	for i in range(len(graph.communities)):
		f.write('community %d: '%i)
		for mem in graph.communities[i]:
			f.write('%d '%mem)
		f.write('\n')

	f.write('\nperformance\n')
	f.write('modularity: %.4f\nperplexity: NA'%graph.modularity)

	f.write('\ndensity: ')
	for ele in graph.density:
		f.write('%.4f '%ele)

	f.write('\nTPR: ')
	for ele in graph.TPR:
		f.write('%.4f '%ele)

	f.write('\ncut ratio: ')
	for ele in graph.cut_ratio:
		f.write('%.4f '%ele)

	f.write('\nconductance: ')
	for ele in graph.conductance:
		f.write('%.4f '%ele)

	if len(f_ref) == 0:
		f.write('\nF measure: %s\nARI: %s\nNMI: %s\n'%(graph.F_measure, graph.ARI, graph.NMI))
	else:
		f.write('\nF measure: %.3f\nARI: %.3f\nNMI: %.3f\n'%(graph.F_measure, graph.ARI, graph.NMI))