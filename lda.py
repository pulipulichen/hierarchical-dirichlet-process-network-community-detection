import numpy as np
from utils import discrete_sample
from scipy.special import gammaln, psi
import utils

np.random.seed(123)
meanchangethresh = 0.001


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])



class LDA:
	'''Latent Dirichlet Allocation topic model'''
	def __init__(self, hyper_params, pos_infer='Gibbs', *corpus):
		'''
		to inference the LDA topic model, we either use Gibbs sampling or Stochastic 
		Variational Inference. We know that Gibbs sampling can be really slow when the model 
		is complicated, that's why I added SVB later. The codes of the SVB part was partly
		borrowed from David Blei' weibsite: https://www.cs.princeton.edu/~blei/topicmodeling.html

		if use Gibbs sampling for posterior inference, pass corpus for initialization
		'''
		self.pos_infer = pos_infer
		if pos_infer == 'Gibbs':
			# if we use Gibbs sampling for posterior inference...
			num_topics = hyper_params['num_topics']
			size_vocab = hyper_params['size_vocab']
			alpha = hyper_params['alpha']
			beta = hyper_params['beta']

			if len(alpha) != num_topics or len(beta) != size_vocab:
				raise ValueError('Hyper-parameter(s) dimension not matched')

			self.corpus = corpus[0]
			self.num_topics = num_topics
			self.size_vocab = size_vocab
			self.D = self.corpus.num_docs
			self.alpha = alpha
			self.beta = beta
			
			self.stat_topic_word_c = np.zeros((num_topics, size_vocab)) + beta
			self.stat_doc_topic_c = np.zeros((self.D, num_topics)) + alpha
			self.stat_topic_c = np.sum(self.stat_topic_word_c, axis=1)
			self.topics_set = np.array([i for i in range(num_topics)], dtype=np.int16)
			
			# initialize topic of each word in the corpus
			self.topics = []
			for d in range(self.D):
				doc = self.corpus.docs[d]
				num_words = doc.length # number of unique words in the document

				topics_lst = []
				for i in range(num_words):
					word = doc.words[i]
					word_count = doc.counts[i] # count of word in the document
					topics = list(discrete_sample(self.topics_set, [1.0]*num_topics, word_count))

					for topic in topics:
						self.stat_doc_topic_c[d][topic] += 1 # increase doc-topic count
						self.stat_topic_word_c[topic][word] += 1 # increase topic-word count
						self.stat_topic_c[topic] += 1
				
					topics_lst.append(topics)
				self.topics.append(topics_lst)


		elif pos_infer == 'SVB':
			# if we use Stochastic Variatonal Inference for posterior inference...
			self._W = hyper_params['size_vocab'] # size of vocabulary
			self._K = hyper_params['num_topics'] # number of topics
			self._D = hyper_params['num_docs'] # number of documents
			self._alpha = hyper_params['alpha'] # hyperparameter for prior on weight vectors theta
			self._eta = hyper_params['eta'] # hyperparameter for prior on topics beta
			self._tau0 = hyper_params['tau0'] + 1 # learning parameter that downweights early iterations
			self._kappa = hyper_params['kappa'] # exponential decay rate, (0.5, 1.0]
			self._updatect = 0

			# initialize the variational distribution
			self._gamma = np.zeros(self._K) # topic weights
			self._lambda = 1*np.random.gamma(100., 1./100., (self._K, self._W)) 
			self._Elogbeta = dirichlet_expectation(self._lambda) # expectation of log(beta)
			self._expElogbeta = np.exp(self._Elogbeta) # expectation of beta


	def do_e_step(self, docs):
		'''
		Given a mini-batch of documents, estimates the parameters
		gamma controlling the variational distribution over the topic
		weights for each document in the mini-batch.

		Arguments:
		docs:  List of D documents. Each document must be represented
				as a string. (Word order is unimportant.) Any
				words not in the vocabulary will be ignored.

		Returns a tuple containing the estimated values of gamma,
		as well as sufficient statistics needed to update lambda.
		'''
		batchD = len(docs) # batch size
		wordids = [] # word ids
		wordcts = [] # word counts
		for d in range(batchD):
			wordids.append(docs[d].words)
			wordcts.append(docs[d].counts)

		# Initialize the variational distribution q(theta|gamma) for
		# the mini-batch
		gamma = 1*np.random.gamma(100., 1./100., (batchD, self._K))
		Elogtheta = dirichlet_expectation(gamma)
		expElogtheta = np.exp(Elogtheta)

		sstats = np.zeros(self._lambda.shape)
		# Now, for each document d update that document's gamma and phi
		it = 0
		meanchange = 0
		for d in range(0, batchD):
			# These are mostly just shorthand (but might help cache locality)
			ids = wordids[d]
			cts = wordcts[d]
			gammad = gamma[d, :]
			Elogthetad = Elogtheta[d, :]
			expElogthetad = expElogtheta[d, :]
			expElogbetad = self._expElogbeta[:, ids]
			# The optimal phi_{dwk} is proportional to 
			# expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
			phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
			# Iterate between gamma and phi until convergence
			for it in range(0, 100):
				lastgamma = gammad
				# We represent phi implicitly to save memory and time.
				# Substituting the value of the optimal phi back into
				# the update for gamma gives this update. Cf. Lee&Seung 2001.
				gammad = self._alpha + expElogthetad * \
				    np.dot(cts / phinorm, expElogbetad.T)
				Elogthetad = dirichlet_expectation(gammad)
				expElogthetad = np.exp(Elogthetad)
				phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
				# If gamma hasn't changed much, we're done.
				meanchange = np.mean(abs(gammad - lastgamma))
				if (meanchange < meanchangethresh):
				    break
			gamma[d, :] = gammad
			# Contribution of document d to the expected sufficient
			# statistics for the M step.
			sstats[:, ids] += np.outer(expElogthetad.T, cts/phinorm)

		# This step finishes computing the sufficient statistics for the
		# M step, so that
		# sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
		# = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
		sstats = sstats * self._expElogbeta

		return((gamma, sstats))


	def update_lambda(self, docs):
		"""
		First does an E step on the mini-batch given in wordids and
		wordcts, then uses the result of that E step to update the
		variational parameter matrix lambda.

		Arguments:
		docs:  List of D documents. Each document must be represented
		       as a string. (Word order is unimportant.) Any
		       words not in the vocabulary will be ignored.

		Returns gamma, the parameters to the variational distribution
		over the topic weights theta for the documents analyzed in this
		update.

		Also returns an estimate of the variational bound for the
		entire corpus for the OLD setting of lambda based on the
		documents passed inp. This can be used as a (possibly very
		noisy) estimate of held-out likelihood.
		"""

		# rhot will be between 0 and 1, and says how much to weight
		# the information we got from this mini-batch.
		rhot = pow(self._tau0 + self._updatect, -self._kappa)
		self._rhot = rhot
		# Do an E step to update gamma, phi | lambda for this
		# mini-batch. This also returns the information about phi that
		# we need to update lambda.
		(gamma, sstats) = self.do_e_step(docs)
		# Estimate held-out likelihood for current values of lambda.
		bound = self.approx_bound(docs, gamma)
		# Update lambda based on documents.
		self._lambda = self._lambda * (1-rhot) + \
		    rhot * (self._eta + self._D * sstats / len(docs))
		self._Elogbeta = dirichlet_expectation(self._lambda)
		self._expElogbeta = np.exp(self._Elogbeta)
		self._updatect += 1

		self._gamma += np.sum(gamma, axis=0)
		self.bound = bound

		return(bound)

	def approx_bound(self, docs, gamma):
		"""
		Estimates the variational bound over *all documents* using only
		the documents passed in as "docs." gamma is the set of parameters
		to the variational distribution q(theta) corresponding to the
		set of documents passed inp.

		The output of this function is going to be noisy, but can be
		useful for assessing convergence.
		"""

		batchD = len(docs) # batch size
		wordids = [] # word ids
		wordcts = [] # word counts
		for d in range(batchD):
			wordids.append(docs[d].words)
			wordcts.append(docs[d].counts)

		score = 0
		Elogtheta = dirichlet_expectation(gamma)
		# should do normalization to avoid overflow
		#
		#
		#(log_var_phi, log_norm) = utils.log_normalize(var_phi)
		expElogtheta = np.exp(Elogtheta)

		# E[log p(docs | theta, beta)]
		for d in range(0, batchD):
			gammad = gamma[d, :]
			ids = wordids[d]
			cts = np.array(wordcts[d])
			phinorm = np.zeros(len(ids))
			for i in range(0, len(ids)):
				temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
				tmax = max(temp)
				phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
			score += np.sum(cts * phinorm)

		# E[log p(theta | alpha) - log q(theta | gamma)]
		score += np.sum((self._alpha - gamma)*Elogtheta)
		score += np.sum(gammaln(gamma) - gammaln(self._alpha))
		score += sum(gammaln(self._alpha*self._K) - gammaln(np.sum(gamma, 1)))

		# Compensate for the subsampling of the population of documents
		score = score * self._D / len(docs)

		# E[log p(beta | eta) - log q (beta | lambda)]
		score = score + np.sum((self._eta-self._lambda)*self._Elogbeta)
		score = score + np.sum(gammaln(self._lambda) - gammaln(self._eta))
		score = score + np.sum(gammaln(self._eta*self._W) - 
		                      gammaln(np.sum(self._lambda, 1)))

		return(score)


	def lda_e_step(self, doc, max_iter=100):
		alpha = self._alpha * np.ones(self._K)
		beta = self._expElogbeta

		gamma = np.ones(len(alpha))  
		expElogtheta = np.exp(dirichlet_expectation(gamma)) 
		betad = beta[:, doc.words]
		phinorm = np.dot(expElogtheta, betad) + 1e-100
		counts = np.array(doc.counts)
		iter = 0
		while iter < max_iter:
		    lastgamma = gamma
		    iter += 1
		    likelihood = 0.0
		    gamma = alpha + expElogtheta * np.dot(counts/phinorm,  betad.T)
		    Elogtheta = dirichlet_expectation(gamma)
		    expElogtheta = np.exp(Elogtheta)
		    phinorm = np.dot(expElogtheta, betad) + 1e-100
		    meanchange = np.mean(abs(gamma-lastgamma))
		    if (meanchange < meanchangethresh):
		        break

		likelihood = np.sum(counts * np.log(phinorm))
		likelihood += np.sum((alpha-gamma) * Elogtheta)
		likelihood += np.sum(gammaln(gamma) - gammaln(alpha))
		likelihood += gammaln(np.sum(alpha)) - gammaln(np.sum(gamma))

		return (likelihood, gamma)
		

	def Gibbs_sampling(self, L=200, display=True, display_stride=20):
		"""
		Inference of the LDA topic model using Gibbs sampling
		param L: maximum iteration
		param display: if true, print processing course
		param display_stride
		"""
		V = self.size_vocab
		K = self.num_topics
		D = self.D

		# Gibbs sampling
		if display: print('Gibbs sampling...')
		for l in range(L):
			if display and (l+1) % display_stride == 0:
				print("finished %s iterations" %(l+1))				
			for d in range(D):
				doc = self.corpus.docs[d]
				num_words = doc.length 
				
				topics_lst = self.topics[d]
				new_topics_lst = []
				for i in range(num_words):
					word = doc.words[i]
					word_count = doc.counts[i]
					topics = topics_lst[i]

					for topic in topics:
						self.stat_doc_topic_c[d][topic] -= 1 # decrease doc-topic count
						self.stat_topic_word_c[topic][word] -= 1 # decrease topic-word count
						self.stat_topic_c[topic] -= 1 # decrease topic count
											
					# sample new topics
					weights = self.stat_topic_word_c[:, word] * \
										self.stat_doc_topic_c[d, :] / self.stat_topic_c
					new_topics = discrete_sample(self.topics_set, weights, word_count)
					new_topics_lst.append(new_topics)

					for topic in new_topics:
						self.stat_doc_topic_c[d][topic] += 1 
						self.stat_topic_word_c[topic][word] += 1
						self.stat_topic_c[topic] += 1
					
				self.topics[d] = new_topics_lst
		if display: print('Gibbs sampling finished')


	def model_inference(self):
		'''LDA parameter inference
		param pos_infer: posterior inference method, default is SVB, Stochastical Variational Inference, 
						another option is Gibbs
		'''
		if self.pos_infer == 'Gibbs':
			self.Phi = (self.stat_topic_word_c.T / self.stat_topic_c).T # K * V
			self.Theta = (self.stat_doc_topic_c.T / np.sum(self.stat_doc_topic_c, axis=1)).T # D*K
			self.topics_mass  = np.sum(self.Theta, axis=0)

		elif self.pos_infer == 'SVB':
			pass
	

	def get_perplexity(self, *options):
		'''calculate perplexity on the test corpus
		param options: for Gibbs sampling, options[0] test corpus, options[1] maximum iteration for
						Gibbs sampling in test corpus
		'''
		if self.pos_infer == 'Gibbs':
			num_topics = self.num_topics
			corpus = options[0]
			L = options[1]
			# initialize topic for each word in the test corpus
			self.t_stat_doc_topic_c = np.zeros((self.D, num_topics)) + self.alpha
			self.t_stat_topic_word_c = self.stat_topic_word_c
			self.t_stat_topic_c = self.stat_topic_c
			self.t_topics = []

			for d in range(corpus.num_docs):
				doc = corpus.docs[d]
				num_words = doc.length # number of unique words in the document

				topics_lst = []
				for i in range(num_words):
					word = doc.words[i]
					word_count = doc.counts[i] # count of word in the document
					topics = list(discrete_sample(self.topics_set, [1.0]*num_topics, word_count))

					for topic in topics:
						self.t_stat_doc_topic_c[d][topic] += 1 # increase doc-topic count
						self.t_stat_topic_word_c[topic][word] += 1 # increase topic-word count
						self.t_stat_topic_c[topic] += 1 # increase topic count
					topics_lst.append(topics)

				self.t_topics.append(topics_lst)

			# Gibbs sampling for topics inference in the test corpus
			for l in range(L): # default iterations: 300	
				for d in range(corpus.num_docs):
					doc = corpus.docs[d]
					num_words = doc.length 
					
					topics_lst = self.t_topics[d]
					new_topics_lst = []
					for i in range(num_words):
						word = doc.words[i]
						word_count = doc.counts[i]
						topics = topics_lst[i]

						for topic in topics:
							self.t_stat_doc_topic_c[d][topic] -= 1 # decrease doc-topic count
							self.t_stat_topic_word_c[topic][word] -= 1 # decrease topic-word count
							self.t_stat_topic_c[topic] -= 1 # decrease topic count
												
						# sample new topics
						weights = self.t_stat_topic_word_c[:, word] * \
										self.t_stat_doc_topic_c[d, :] / self.t_stat_topic_c
						new_topics = discrete_sample(self.topics_set, weights, word_count)
						new_topics_lst.append(new_topics)

						for topic in new_topics:
							self.t_stat_doc_topic_c[d][topic] += 1 
							self.t_stat_topic_word_c[topic][word] += 1
							self.t_stat_topic_c[topic] += 1
					self.t_topics[d] = new_topics_lst

			# get Theta on the test corpus
			self.t_Theta = (self.t_stat_doc_topic_c.T / np.sum(self.t_stat_doc_topic_c, axis=1)).T # corpus.num_docs*K
			# get perplexity
			scores = np.empty(corpus.num_docs).astype(float)
			self.t_Psi = np.log(np.dot(self.t_Theta, self.Phi))
			totals = []
			for d in range(corpus.num_docs):
				doc = corpus.docs[d]
				scores[d] = np.sum([self.t_Psi[d][w]*doc.counts[i] for i, w in enumerate(doc.words)])
				totals.append(doc.total)
			self.perplexity = np.exp(-np.sum(scores) / np.sum(totals))

		elif self.pos_infer == 'SVB':
			corpus = options[0] # test corpus
			L = options[1] # maximum iteration
			total_doc_count = corpus.num_docs
			test_score = 0.0
			c_test_word_count = sum([doc.total for doc in corpus.docs])

			for doc in corpus.docs:
				(likelihood, gamma) = self.lda_e_step(doc, L)
				test_score += likelihood

			self.perplexity = np.exp(-test_score / c_test_word_count)


	def most_likely_topic(self):
		"""find the most likely community of each node using Bayesian rule"""
		if self.pos_infer == 'Gibbs':
			word_topic_prob = self.Phi.T * self.topics_mass
			self.word_topic = np.argmax(word_topic_prob, axis=1)

		elif self.pos_infer == 'SVB':
			self._gamma = self._gamma / np.sum(self._gamma)
			word_topic_prob = self._expElogbeta.T * self._gamma # proportional (not normalized)
			self.word_topic = np.argmax(word_topic_prob, axis=1)
