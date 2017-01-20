from graph import Graph
from corpus import Corpus
from hdp import HDP, np, lda_e_step_split, lda_e_step
from utils import Sys, printProgress

import time
import random

random.seed(123)


# hyperparameters:
# avg_l, D, T, and K matter the most
# ------------------------------------------------------------------
# 		      |yeast|ca-GrQc|facebook|GSE     |powergrid|ca-CondMat|
# type        |bio  |co-auth|social  |bio     |		    |co-author | 
# n           |1540 |5242   |4039    |9112    |4941	    |16264     |
# m           |8703 |14478  |88234   |244928  |6594     |47594     |
# ------------------------------------------------------------------
# performance (results may vary on different computers)            |
# K           |184  |262    |        |125     |66       |464       |
# Q           |0.760|0.784  |0.815   |0.597   |0.909    |0.759     |
# Perplexity  |62.3 |504    |        |1124.5  |235.5    |1262.2    |
# density     |             ................                       |
# TPR     	  |			    ................                       |
# cut ratio   |				................                       |
# conductance |    			................                       |
# F measure   |             not applicatable                       |
# RDI         |				not applicatable                       |
# NMI         |    			not applicatable                       |
# ------------------------------------------------------------------
# corpus parameters   											   |
# avg_l       |100  |100    |100     |90      |90 	    |90        |
# D           |8000 |16000  |2000*8  |2000*18 |2000*12  |2000*40   |
# te_D        |2000 |2000   |2000    |2000    |2000     |2000      |
# ------------------------------------------------------------------
# HDP topic model parameters    								   |
# T           |500  |       |500     |150     |500   	|700       |
# K           |8    |       |8       |7       |8 		|8         |
# eta         |2    |       |2       |2       |2 		|2         |     
# alpha       |2    |       |2       |2       |2   	    |2         |
# gamma       |2    |       |6       |2       |2 		|2         |
# ------------------------------------------------------------------
# SVB training parameters    									   |
# kappa       |0.95 |       |0.95    |0.95    |0.95 	|0.95      |
# tau         |1    |       |1       |1       |1 		|1         |
# scale       |1    |       |1       |1       |1 		|1         |
# batchD      |1    |       |2       |2       |2   	    |2         |
# epochs      |3    |       |3       |3       |3  	    |2         |
# convg       |0.1  |       |0.1     |0.00005 |0.1      |0.1       |
# ------------------------------------------------------------------


avg_l = 100 # average length of random walks, random walk length ~ Poisson(avg_l)
D = int(2000*8) # number of documents in the training corpus
num_spl = 8 # number of splits
spl_D = int(D / num_spl) # number of documets per split
te_D = 2000 # number of documents in the testing corpus

T = 500 # corpus level truncation
K = 10 # document level truncation
eta = 2 # Dirichlet prior of topics
gamma = 2 # corpus level concentration parameter
alpha = 2 # document level concentration parameter

kappa = 0.95 # forgetting rate parameter
tau = 1
scale = 1
adding_noise = False
batchsize = 2 # number of documents per batch
epochs = 3 # epochs
var_converge = 0.1

network = 'facebook'
f_net = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.elst.csv'%(network, network) # network file
f_log = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.online_rwhdp.log'%(network, network) # log file
f_out = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.online_rwhdp.out'%(network, network) # output file
f_ref = '' # ground truth community structure file

# load graph
print('loading graph...')
graph = Graph(f_net, 'edge list', directed=False, weighted=False, memory_control=True)


# online stochastic variational inference
train_corpus = Corpus()
hdp = HDP(T, K, D, graph.n, eta, alpha, gamma, kappa, tau, scale, adding_noise)
log_file = open(f_log, "w") 
log_file.write("iteration time doc.count score word.count unseen.score unseen.word.count\n")

max_iter_per_epoch = np.ceil(spl_D / batchsize)
total_doc_count = 0
total_time = 0
doc_seen = set()
print("online stochastic variational inference...")
for spl in range(num_spl):
	print('process split %d in %d splits...'%(int(spl+1), num_spl))
	train_corpus.generate_corpus_from_graph_using_random_walk(graph, avg_l, spl_D, 'deterministic+random')
	for epoch in range(epochs):
		iter = 0
		printProgress(iter, max_iter_per_epoch, prefix='epoch %s'%int(epoch+1), suffix='complete', barLength=50)
		while iter < max_iter_per_epoch:
			iter += 1
			t0 = time.clock()
			# Sample the documents.
			ids = random.sample(range(spl_D*spl, spl_D*(spl+1)), batchsize)
			docs = [train_corpus.docs[id-spl_D*spl] for id in ids]
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

		hdp.most_likely_topic()
		graph.reorder_community(hdp.m_word_topic)
		graph.get_modularity()
		print("%d split(s), %d epoch(s), number of communities: %d, modularity: %s" 
								%(int(spl+1), int(epoch+1), len(np.unique(hdp.m_word_topic)), graph.modularity))
log_file.close()
print('online stochastic variational inference finished%s'%(' '*40))


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

test_corpus = Corpus()
test_corpus.generate_corpus_from_graph_using_random_walk(graph, avg_l, te_D)
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
		f.write('\nF measure: %.3f\nARI: %.3f\nNMI: %.3f\n'%(graph.F_measure, graph.ARI, graph.NMI))