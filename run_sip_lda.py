from graph import Graph
from corpus import Corpus, corpus_split
from lda import LDA, np
from utils import Sys, printProgress

import time
import random
random.seed(123)


# -----------------------------------------------------------------------------
# 		      |yeast  |ca-GrQc   |facebook  |GSE       |powergrid  |ca-CondMat|
# type        |bio    |co-auth   |social    |bio       |		   |co-author | 
# n           |1540   |5242      |4039      |9112      |4941	   |16264     |
# m           |8703   |14478     |88234     |244928    |6594       |47594     |
# -----------------------------------------------------------------------------
# performance (results may vary on different computers)                       |
# K           |210    |320       |8         |207       |256        |573       |
# Q           |0.699  |0.747     |0.712     |0.588     |0.778      |0.661     |
# Perplexity  |280    |2902.8    |826.4     |1664.8    |7197.5     |41920.7   |
# density     |             ................                                  |
# TPR     	  |			    ................                                  |
# cut ratio   |				................                                  |
# conductance |    			................                                  |
# F measure   |             not applicatable                                  |
# RDI         |				not applicatable                                  |
# NMI         |    			not applicatable                                  |
# -----------------------------------------------------------------------------
# SIP-LDA model parameters    								                  |
# K           |300    |350       |300       |300       |300 	   |600       |
# -----------------------------------------------------------------------------
# SVB training parameters    									              |
# kappa       |0.7    |0.7       |0.7       |0.7       |0.7 	   |0.7       |
# tau0        |1024   |1024      |1024      |1024      |1024 	   |1024      |
# batchD      |5      |5         |5         |5         |5   	   |5         |
# epochs      |3      |3         |3         |3         |3  	       |2         |
# -----------------------------------------------------------------------------


K = 600 # incremental size 50
alpha = 1.0/K
eta = 1.0/K

tau0 = 1024
kappa = 0.7
batchsize = 5
epochs = 3


network = 'yeast'
f_net = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.elst.csv'%(network, network)
f_log = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.siplda.log'%(network, network)
f_out = '/home/ruimin/anaconda3/network_analysis/data/%s/%s.siplda.out'%(network, network)
f_ref = ''


# load graph
print('loading graph...')
graph = Graph(f_net, 'edge list', directed=False, weighted=False, memory_control=True)


# generate corpus
print('generating training/testing corpus...')
corpus = Corpus()
corpus.generate_corpus_from_graph_using_SIP(graph, '012-SIP')
train_corpus, test_corpus = corpus_split(corpus)


# stochastic variational inference
hyper_params_svb = {}
hyper_params_svb['num_topics'] = K
hyper_params_svb['alpha'] = alpha # uniform [1/K, ..., 1/K]
hyper_params_svb['eta'] = eta # uniform [1/K, ..., 1/K]
hyper_params_svb['size_vocab'] = graph.n
hyper_params_svb['num_docs'] = train_corpus.num_docs
hyper_params_svb['tau0'] = tau0
hyper_params_svb['kappa'] = kappa

lda_svb = LDA(hyper_params_svb, 'SVB')
log_file = open(f_log, "w") 
log_file.write("iteration time rthot held-out log-perplexity estimate\n")

total_time = 0
D = train_corpus.num_docs
max_iter_per_epoch = np.ceil(D / batchsize)

print('stochastic variational inference...')
for epoch in range(epochs):
	iter = 0
	printProgress(iter, max_iter_per_epoch, prefix='epoch %s'%int(epoch+1), suffix='complete', barLength=50)
	while iter < max_iter_per_epoch:
	    iter += 1
	    t0 = time.clock()
	    ids = random.sample(range(D), batchsize)
	    docs = [train_corpus.docs[id] for id in ids]
	    wordids = [] # word ids
	    wordcts = [] # word counts
	    for d in range(batchsize):
	        wordids.append(docs[d].words)
	        wordcts.append(docs[d].counts)
	        
	    bound = lda_svb.update_lambda(docs)
	    perwordbound = bound*len(docs) / (D*sum(map(sum, wordcts)))
	    
	    total_time += time.clock() - t0
	    log_file.write("%d %d %.5f %.5f\n" % (iter, total_time,
	                    lda_svb._rhot, -perwordbound))
	    log_file.flush()
	    printProgress(iter, max_iter_per_epoch, prefix='epoch %s'%int(epoch+1), suffix='complete', barLength=50)

log_file.close()
Sys.stdout.write('stochastic variational inference finished%s'%(' '*40)),
Sys.stdout.flush()


# metrics
print("\nmetrics:")
lda_svb.most_likely_topic()
print("number of communities: %d" %len(np.unique(lda_svb.word_topic)))

graph.reorder_community(lda_svb.word_topic)
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

lda_svb.get_perplexity(test_corpus, 100) # this may take a while
print('perplexity: %s' %lda_svb.perplexity)

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
	f.write('number of training docs: %s\nnumber of testing docs: %s\n'
		%(train_corpus.num_docs, test_corpus.num_docs))

	f.write('\nmodel parameters\n')
	f.write('K: %d\n'%K)

	f.write('\ntraining parameters\n')
	f.write('tau0: %.2f\nkappa: %.2f\nbatchsize: %d\nepochs: %d\n'%(tau0, kappa, batchsize, epochs))

	f.write('\npartition\n')
	for i in range(len(graph.communities)):
		f.write('community %d: '%i)
		for mem in graph.communities[i]:
			f.write('%d '%mem)
		f.write('\n')

	f.write('\nperformance\n')
	f.write('modularity: %.4f\nperplexity: %.2f'%(graph.modularity, lda_svb.perplexity))

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