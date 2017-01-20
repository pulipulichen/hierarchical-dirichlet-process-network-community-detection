from graph import np, Graph
from numpy.random import choice
from utils import discrete_sample

np.random.seed(123)


class Document:
    def __init__(self):
        self.words = [] # word id
        self.counts = [] # word frequency
        self.length = 0 # number of unique words
        self.total = 0 # doc length


class Corpus:
    def __init__(self):
        self.size_vocab = 0
        self.docs = []
        self.num_docs = []

    def generate_corpus_from_graph_using_random_walk(self, graph, avg_len, num_docs, st_mod='random'):
        """
        param graph: class object
        param avg_len: doc length ~ Poisson(avg_len)
        param num_docs: number of random walks(number of documents)
        param train: if set true, will generate training corpus, otherwise testing corpus
        """
        #if train == True and num_docs < graph.n: raise ValueError('Corpus size should be greater than graph size')
        self.size_vocab = graph.n # vocabulary size
        self.num_docs = num_docs # number of documents
        self.docs = []

        # starting points of random walks, if training, ensure each node appears at least once in the corpus
        if st_mod == 'deterministic+random':
            #if len(graph.st_once) == graph.n: # all nodes being starting points at least once
            if len(np.where(graph.st_times >= 1)[0]) == graph.n: # all nodes being starting points at least once
                st_points = np.random.choice(np.array(range(graph.n)), num_docs, replace=True)
            else:
                #st_never = graph.nodes_set.difference(graph.st_once)
                st_never = np.where(graph.st_times == 0)[0] # nodes that never being starting points, st_never[0] is numpy array
                if len(st_never) >= num_docs:
                    st_points = st_never[0:num_docs]
                    #graph.st_once = graph.st_once.union(set(st_points))
                else:
                    st_points = np.concatenate((st_never, np.random.choice(np.array(range(graph.n)), 
                                                                            num_docs-len(st_never), replace=True)))
            graph.st_times[st_points] += 1

        elif st_mod == 'deterministic':
            cts = 0
            st_points = []
            while(cts < num_docs):
                t = np.min(graph.st_times)
                nodes_ = np.where(graph.st_times == t)[0]
                st_points += list(nodes_[0: min(len(nodes_), num_docs-cts)])
                graph.st_times[st_points] += 1
                cts += min(len(nodes_), num_docs-cts)
            st_points = np.array(st_points)

        elif st_mod == 'random':
            st_points = np.random.choice(np.array(range(graph.n)), num_docs, replace=True)
            graph.st_times[st_points] += 1

        for d in range(num_docs):
            doc = Document()
            
            N = max(np.random.poisson(avg_len), 1) # doc length
            rnd_wlk = np.zeros(N).astype(int)
            rnd_wlk[0] = st_points[d]
            for w in range(1, N):
                current = rnd_wlk[w-1]
                rnd_wlk[w] = discrete_sample(np.array(graph.adj_lst[current][0]), 
                                                graph.adj_lst[current][1], 1)[0]

            words = np.unique(rnd_wlk)
            doc.length = len(words)
            doc.words = [0 for i in range(doc.length)]
            doc.counts = [0 for i in range(doc.length)]
            for i in range(doc.length):
                doc.words[i] = words[i] 
                doc.counts[i] = np.sum(rnd_wlk==words[i])

            doc.total = sum(doc.counts)
            self.docs.append(doc)


    def generate_corpus_from_graph_using_SIP(self, graph, SIP='01-SIP'):
        '''param SIP: 01-SIP, or 012-SIP, or k-SIP, for details, see the paper:
        An LDA-based Community Structure Discovery Approach for Large-Scale Social Networks'''
        self.size_vocab = graph.n
        self.num_docs = num_docs = graph.n

        for d in range(num_docs):
            doc = Document()

            if SIP == '01-SIP':
                doc.words = words = graph.adj_lst[d][0]
                doc.length = len(words)
                doc.counts = [1.0 for i in range(doc.length)]
                doc.total = sum(doc.counts)
            elif SIP == '012-SIP':
                fst_nghbrs = graph.adj_lst[d][0] # immediate neighbors of the node
                snd_nghbrs = [] # immediate neighbors' neighbors
                for nghbr in fst_nghbrs:
                    snd_nghbrs += graph.adj_lst[nghbr][0]
                snd_nghbrs = list(set(np.unique(snd_nghbrs)).difference(set([d]).union(set(fst_nghbrs))))

                doc.words = fst_nghbrs + snd_nghbrs
                doc.length = len(doc.words)
                doc.counts = [2.0 for i in range(len(fst_nghbrs))] + [1.0 for i in range(len(snd_nghbrs))]
                doc.total = sum(doc.counts)

            self.docs.append(doc)


def corpus_split(corpus, split_ratio=0.9):
    '''split the corpus generated by using SIP into two parts: one for training and one for testing
    param split_ratio: default 0.9 means that 90/100 of documents will be training documents'''
    
    train_ids = list(choice(corpus.num_docs, np.ceil(corpus.num_docs*split_ratio), replace=False))
    test_ids = list(set(range(corpus.num_docs)).difference(set(train_ids)))

    train_corpus = Corpus()
    train_corpus.num_docs = len(train_ids)
    train_corpus.size_vocab = corpus.size_vocab
    for i in train_ids:
        train_corpus.docs.append(corpus.docs[i])

    test_corpus = Corpus()
    test_corpus.num_docs = len(test_ids)
    test_corpus.size_vocab = corpus.size_vocab
    for i in test_ids:
        test_corpus.docs.append(corpus.docs[i])

    return(train_corpus, test_corpus)


