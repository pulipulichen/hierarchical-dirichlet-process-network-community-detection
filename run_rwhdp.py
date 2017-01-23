from graph import Graph
from corpus import Corpus
from hdp import HDP, np, lda_e_step_split, lda_e_step
from utils import Sys, printProgress

import time
import random

random.seed(123)

# hyperparameters:
# avg_l, D, T, and K matter the most
# -------------------------------
# 		|yeast|ca-GrQc|
# -------------------------------
# performance (results may vary on different computers)
# K     |179  |247    |
# Q     |0.761|0.783  |
# P     |62.8 |466.0
# C     |     |       |
# F     |NA   |NA     |
# RDI   |NA   |NA     |
# NMI   |NA   |NA     |
# -------------------------------
# corpus parameters
# avg_l |100  |100    |
# D     |8000 |16000  |
# -------------------------------
# HDP topic model parameters
# T     |500  |400    |
# K     |8    |10     |
# eta   |2    |2      |
# alpha |2    |2      |
# gamma |2    |2      |
# -------------------------------
# SVB training parameters
# kappa |0.95 |0.95   |
# tau   |1    |1      |
# scale |1    |1      |
# batchD|1    |2      |
# epochs|3    |3      |
# convg |0.1  |0.1    |
# -------------------------------

avg_l = 100 # average length of random walks
D = 8000 # number of documents in the training corpus

T = 500 # corpus level truncation
K = 8 # document level truncation
eta = 2 # Dirichlet prior of topics
gamma = 2 # corpus level concentration parameter
alpha = 2 # document level concentration parameter


kappa = 0.95 # forgetting rate parameter
tau = 1
scale = 1
adding_noise = False
batchsize = 1 # mini-batch size
epochs = 3 # epochs
var_converge = 0.1

root_dir = ''
f_net = '%s/data/example.csv'%root_dir
f_log = '%s/output/example.rwhdp.log'%root_dir
f_out = '%s/output/example.rwhdp.out'%root_dir
f_ref = ''

# load graph
print('loading graph...')
graph = Graph(f_net, 'edge list', directed=False, weighted=False, memory_control=True)


# generate corpus
print('generating training/testing corpus...')
train_corpus = Corpus()
train_corpus.generate_corpus_from_graph_using_random_walk(graph, avg_l, D, 'deterministic+random')
test_corpus = Corpus()
test_corpus.generate_corpus_from_graph_using_random_walk(graph, avg_l, 3000)


# stochastic variational inference
hdp = HDP(T, K, D, graph.n, eta, alpha, gamma, kappa, tau, scale, adding_noise)
log_file = open(f_log, "w") 
log_file.write("iteration time doc.count score word.count unseen.score unseen.word.count\n")

max_iter_per_epoch = np.ceil(D / batchsize)
total_doc_count = 0
total_time = 0
doc_seen = set()
print("stochastic variational inference...")
for epoch in range(epochs):
	iter = 0
	printProgress(iter, max_iter_per_epoch, prefix='epoch %s'%int(epoch+1), suffix='complete', barLength=50)
	while iter < max_iter_per_epoch:
	    iter += 1
	    t0 = time.clock()

	    # Sample the documents.
	    ids = random.sample(range(D), batchsize)
	    docs = [train_corpus.docs[id] for id in ids]
	    unseen_ids = set([i for (i, id) in enumerate(ids) if id not in doc_seen])
	    if len(unseen_ids) != 0:
	        doc_seen.update([id for id in ids])

	    total_doc_count += batchsize
	    (score, count, unseen_score, unseen_count) = hdp.process_documents(docs, var_converge, unseen_ids)
	    total_time += time.clock() - t0
	    log_file.write("%d %d %d %.5f %d %.5f %d\n" % (iter, total_time,
	                    total_doc_count, score, count, unseen_score, unseen_count))
	    log_file.flush()
	    printProgress(iter, max_iter_per_epoch, prefix='epoch %s'%int(epoch+1), suffix='complete', barLength=50)
log_file.close()
Sys.stdout.write('stochastic variational inference finished                                                  '),
Sys.stdout.flush()


# metrics
print("\nmetrics:")
hdp.most_likely_topic()
print("number of communities: %d" %len(np.unique(hdp.m_word_topic)))

graph.reorder_community(hdp.m_word_topic)
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

hdp.get_perplexity(test_corpus)
print('perplexity: %s' %hdp.perplexity)

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

	f.write('\ncorpus statistics\n')
	f.write('average length: %s\nnumber of training docs: %s\nnumber of testing docs: %s\n'
		%(avg_l, train_corpus.num_docs, test_corpus.num_docs))

	f.write('\nmodel parameters\n')
	f.write('T truncation: %s\nK truncation: %s\neta: %s\nalpha: %s\ngamma: %s\n'
		%(T, K, eta, alpha, gamma))

	f.write('\ntraining parameters\n')
	f.write('kappa: %.2f\ntau: %.2f\nscale: %.2f\nbatch size: %d\nepochs: %d\n'
		%(kappa, tau, scale, batchsize, epochs))

	f.write('\npartition\n')
	for i in range(len(graph.communities)):
		f.write('community %d: '%i)
		for mem in graph.communities[i]:
			f.write('%d '%mem)
		f.write('\n')

	#f.write('\nperformance\n')
	#f.write('modularity: %.4f\nperplexity: %.2f\n'%(graph.modularity, hdp.perplexity))
	#f.write('conductance: ')
	#for ele in graph.conductance:
	#	f.write('%.4f '%ele)

	f.write('\nperformance\n')
	f.write('modularity: %.4f\nperplexity: %.2f'%(graph.modularity, hdp.perplexity))

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
		f.write('F measure: %.3f\nARI: %.3f\nNMI: %.3f\n'%(graph.F_measure, graph.ARI, graph.NMI))