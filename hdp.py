import numpy as np
import scipy.special as sp
import random

from corpus import Corpus
import utils

np.random.seed(123)
random.seed(123)

meanchangethresh = 0.00001
min_adding_noise_point = 10 
min_adding_noise_ratio = 1 
mu0 = 0.3 
rhot_bound = 0.0

def dirichlet_expectation(alpha):
    """ For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha """
    if (len(alpha.shape) == 1): return(sp.psi(alpha) - sp.psi(np.sum(alpha)))
    return(sp.psi(alpha) - sp.psi(np.sum(alpha, 1))[:, np.newaxis])


def expect_log_sticks(sticks):
    """
    For stick-breaking HDP, this returns the E[log(sticks)] 
    v[k] ~ Beta(a[k], b[k]), sticks[0][k] = a[k]
                             sticks[1][k] = b[k]
    """
    dig_sum = sp.psi(np.sum(sticks, 0))
    ElogW = sp.psi(sticks[0]) - dig_sum
    Elog1_W = sp.psi(sticks[1]) - dig_sum

    n = len(sticks[0]) + 1
    Elogsticks = np.zeros(n)
    Elogsticks[0:n-1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W) #cumsum: cumulate summation
    return Elogsticks # len(Elogsticks) = len(sticks[0]) + 1 because len(sticks[0]) = T -1


def lda_e_step_split(doc, alpha, beta, max_iter=100):
    half_len = int(doc.length/2) + 1
    idx_train = [2*i for i in range(half_len) if 2*i < doc.length]
    idx_test = [2*i+1 for i in range(half_len) if 2*i+1 < doc.length]
   
   # split the document
    words_train = [doc.words[i] for i in idx_train]
    counts_train = [doc.counts[i] for i in idx_train]
    words_test = [doc.words[i] for i in idx_test]
    counts_test = [doc.counts[i] for i in idx_test]

    gamma = np.ones(len(alpha))  
    expElogtheta = np.exp(dirichlet_expectation(gamma)) 
    betad = beta[:, words_train]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(counts_train)
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

    gamma = gamma/np.sum(gamma)
    counts = np.array(counts_test)
    betad = beta[:, words_test]
    score = np.sum(counts * np.log(np.dot(gamma, betad) + 1e-100))

    return (score, np.sum(counts), gamma)

def lda_e_step(doc, alpha, beta, max_iter=100):
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
    likelihood += np.sum(sp.gammaln(gamma) - sp.gammaln(alpha))
    likelihood += sp.gammaln(np.sum(alpha)) - sp.gammaln(np.sum(gamma))

    return (likelihood, gamma)


class Suff_Stats: 
    """
    sufficient statistcs
    """
    def __init__(self, T, Wt, Dt):
        self.m_batchsize = Dt
        self.m_var_sticks_ss = np.zeros(T) #?
        self.m_var_beta_ss = np.zeros((T, Wt)) #?
    
    def set_zero(self):
        self.m_var_sticks_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)


class HDP:
    ''' HDP model using stick breaking'''
    def __init__(self, T, K, D, W, eta, alpha, gamma, kappa, tau, scale=1.0, 
                adding_noise=False):
        """
        gamma: first level concentration (sticks concentration)
        alpha: second level concentration 
        eta: the topic Dirichlet 
        T: top level truncation level (number of topics)
        K: second level truncation level (number of topic pointers)
        W: size of vocab 
        D: number of documents in the corpus 
        kappa: learning rate 
        tau: slow down parameter 
        scale: default 1.0
        """

        self.m_W = W
        self.m_D = D
        self.m_T = T
        self.m_K = K
        self.m_alpha = alpha
        self.m_gamma = gamma

        self.m_var_sticks = np.zeros((2, T-1))
        self.m_var_sticks[0] = 1.0
        #self.m_var_sticks[1] = self.m_gamma
        # make a uniform at beginning
        self.m_var_sticks[1] = range(T-1, 0, -1)

        self.m_varphi_ss = np.zeros(T)

        self.m_lambda = np.random.gamma(1.0, 1.0, (T, W)) * D*100 / (T*W)-eta #?
        self.m_eta = eta
        self.m_Elogbeta = dirichlet_expectation(self.m_eta + self.m_lambda)

        self.m_tau = tau + 1
        self.m_kappa = kappa
        self.m_scale = scale 

        self.m_updatect = 0 #?
        self.m_status_up_to_date = True #?
        self.m_adding_noise = adding_noise #?
        self.m_num_docs_parsed = 0 #?

        # Timestamps and normalizers for lazy updates
        self.m_timestamp = np.zeros(self.m_W, dtype=int) #?
        self.m_r = [0] #?
        self.m_lambda_sum = np.sum(self.m_lambda, axis=1) #?


    def new_init(self, c): # c stands for corpus
        self.m_lambda = 1.0/self.m_W + 0.01 * np.random.gamma(1.0, 1.0, \
            (self.m_T, self.m_W))
        self.m_Elogbeta = dirichlet_expectation(self.m_eta + self.m_lambda)

        num_samples = min(c.num_docs, burn_in_samples) # burn_in_samples ?
        ids = random.sample(range(c.num_docs), num_samples)
        docs = [c.docs[id] for id in ids]

        unique_words = dict()
        word_list = []
        for doc in docs:
            for w in doc.words:
                if w not in unique_words:
                    unique_words[w] = len(unique_words)
                    word_list.append(w)
        Wt = len(word_list) # length of words in these documents

        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks -- v[k]
        for doc in docs:
            old_lambda = self.m_lambda[:, word_list].copy() # why use copy()?
            for iter in range(5):
                sstats = Suff_Stats(self.m_T, Wt, 1) 
                doc_score = self.doc_e_step(doc, sstats, Elogsticks_1st, \
                            word_list, unique_words, var_converge=0.0001, max_iter=5) #???

                self.m_lambda[:, word_list] = old_lambda + sstats.m_var_beta_ss / sstats.m_batchsize
                self.m_Elogbeta = dirichlet_expectation(self.m_lambda)

        self.m_lambda_sum = np.sum(self.m_lambda, axis=1)


    def process_documents(self, docs, var_converge, unseen_ids=[], update=True, opt_o=True):
        # Find the unique words in this mini-batch of documents...

        # unseen_ids ids of docs that have never been processed so far
        self.m_num_docs_parsed += len(docs) # size of mini-batch
        adding_noise = False # why adding noise???????????
        adding_noise_point = min_adding_noise_point #?????

        if self.m_adding_noise: # ?
            if float(adding_noise_point) / len(docs) < min_adding_noise_ratio:
                adding_noise_point = min_adding_noise_ratio * len(docs)

            if self.m_num_docs_parsed % adding_noise_point == 0:
                adding_noise = True

        unique_words = dict()
        word_list = []
        if adding_noise:
          word_list = range(self.m_W)
          for w in word_list:
            unique_words[w] = w
        else:
            for doc in docs:
                for w in doc.words:
                    if w not in unique_words:
                        unique_words[w] = len(unique_words)
                        word_list.append(w)
        Wt = len(word_list) # length of words in these documents

        # ...and do the lazy updates on the necessary columns of lambda
        rw = np.array([self.m_r[t] for t in self.m_timestamp[word_list]])
        self.m_lambda[:, word_list] *= np.exp(self.m_r[-1] - rw)
        self.m_Elogbeta[:, word_list] = \
            sp.psi(self.m_eta + self.m_lambda[:, word_list]) - \
            sp.psi(self.m_W*self.m_eta + self.m_lambda_sum[:, np.newaxis])

        ss = Suff_Stats(self.m_T, Wt, len(docs)) 

        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks

        # run variational inference on some new docs
        score = 0.0
        count = 0
        unseen_score = 0.0
        unseen_count = 0
        for i, doc in enumerate(docs):
            doc_score = self.doc_e_step(doc, ss, Elogsticks_1st, \
                        word_list, unique_words, var_converge)
            count += doc.total
            score += doc_score
            if i in unseen_ids:
              unseen_score += doc_score
              unseen_count += doc.total

        if adding_noise:
            # add noise to the ss
            print("adding noise at this stage...")

            ## old noise
            noise = np.random.gamma(1.0, 1.0, ss.m_var_beta_ss.shape)
            noise_sum = np.sum(noise, axis=1)
            ratio = np.sum(ss.m_var_beta_ss, axis=1) / noise_sum
            noise =  noise * ratio[:,np.newaxis]

            ## new noise
            #lambda_sum_tmp = self.m_W * self.m_eta + self.m_lambda_sum
            #scaled_beta = 5*self.m_W * (self.m_lambda + self.m_eta) / (lambda_sum_tmp[:, np.newaxis])
            #noise = np.random.gamma(shape=scaled_beta, scale=1.0)
            #noise_ratio = lambda_sum_tmp / noise_sum
            #noise = (noise * noise_ratio[:, np.newaxis] - self.m_eta) * len(docs)/self.m_D

            mu = mu0 *1000.0 / (self.m_updatect + 1000)
            ss.m_var_beta_ss = ss.m_var_beta_ss * (1.0-mu) + noise * mu 
       
        if update:
            self.update_lambda(ss, word_list, opt_o)
    
        return (score, count, unseen_score, unseen_count) # in the end unseen_score, unseen_count should be 0.0


    def optimal_ordering(self):
        """
        ordering the topics 
        it makes sense, if lambda_sum[k] is big, then there must be a lot of words
        belongs to this topic
        """
        idx = [i for i in reversed(np.argsort(self.m_lambda_sum))] #argsort small --> big
                                                                   # idx lambda_sum big --> small
        self.m_varphi_ss = self.m_varphi_ss[idx]
        self.m_lambda = self.m_lambda[idx,:]
        self.m_lambda_sum = self.m_lambda_sum[idx]
        self.m_Elogbeta = self.m_Elogbeta[idx,:]


    def doc_e_step(self, doc, ss, Elogsticks_1st, word_list, unique_words, var_converge, max_iter=100):
        """
        e step for a single doc, update local hidden variables
        """

        batchids = [unique_words[id] for id in doc.words]

        Elogbeta_doc = self.m_Elogbeta[:, doc.words] 
        ## very similar to the HDP equations
        v = np.zeros((2, self.m_K-1))  # pi[i] ~ Beta(1, alpha), i = 1, ..., T-1
        v[0] = 1.0
        v[1] = self.m_alpha

        # The following line is of no use.
        Elogsticks_2nd = expect_log_sticks(v) # Elogsticks_2nd represents document level

        # back to the uniform
        phi = np.ones((len(doc.words), self.m_K)) * 1.0/self.m_K # this is phi[d, n, i]

        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        eps = 1e-100
        
        iter = 0
        # not yet support second level optimization yet, to be done in the future
        while iter < max_iter and (converge < 0.0 or converge > var_converge):
            ### update variational parameters
            # var_phi 
            # var_phi seems to be zeta
            if iter < 3:
                var_phi = np.dot(phi.T,  (Elogbeta_doc * doc.counts).T)
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(phi.T,  (Elogbeta_doc * doc.counts).T) + Elogsticks_1st
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            
            # phi
            if iter < 3:
                phi = np.dot(var_phi, Elogbeta_doc).T
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)
            else:
                phi = np.dot(var_phi, Elogbeta_doc).T + Elogsticks_2nd
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)

            # v 
            # v seems to be gamma
            phi_all = phi * np.array(doc.counts)[:,np.newaxis]
            v[0] = 1.0 + np.sum(phi_all[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi_all[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)

            likelihood = 0.0
            # compute likelihood
            # var_phi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)

            # v part/ v in john's notation, john's beta is alpha here
            log_alpha = np.log(self.m_alpha)
            likelihood += (self.m_K-1) * log_alpha
            dig_sum = sp.psi(np.sum(v, 0))
            likelihood += np.sum((np.array([1.0, self.m_alpha])[:,np.newaxis]-v) * (sp.psi(v)-dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) - np.sum(sp.gammaln(v))

            # Z part 
            likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)

            # X part, the data part
            likelihood += np.sum(phi.T * np.dot(var_phi, Elogbeta_doc * doc.counts))

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            #if converge < -0.000001:
                #print("warning, likelihood is decreasing!")
            
            iter += 1
            
        # update the suff_stat ss 
        # this time it only contains information from one doc
        ss.m_var_sticks_ss += np.sum(var_phi, 0)   
        ss.m_var_beta_ss[:, batchids] += np.dot(var_phi.T, phi.T * doc.counts)

        return(likelihood)

    def update_lambda(self, sstats, word_list, opt_o): 
        """
        m step, update global hidden variables
        """
        
        self.m_status_up_to_date = False
        if len(word_list) == self.m_W:
          self.m_status_up_to_date = True
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = self.m_scale * pow(self.m_tau + self.m_updatect, -self.m_kappa)
        if rhot < rhot_bound: 
            rhot = rhot_bound
        self.m_rhot = rhot

        # Update appropriate columns of lambda based on documents.
        self.m_lambda[:, word_list] = self.m_lambda[:, word_list] * (1-rhot) + \
            rhot * self.m_D * sstats.m_var_beta_ss / sstats.m_batchsize
        self.m_lambda_sum = (1-rhot) * self.m_lambda_sum+ \
            rhot * self.m_D * np.sum(sstats.m_var_beta_ss, axis=1) / sstats.m_batchsize

        self.m_updatect += 1
        self.m_timestamp[word_list] = self.m_updatect
        self.m_r.append(self.m_r[-1] + np.log(1-rhot))

        self.m_varphi_ss = (1.0-rhot) * self.m_varphi_ss + rhot * \
               sstats.m_var_sticks_ss * self.m_D / sstats.m_batchsize

        if opt_o:
            self.optimal_ordering();

        ## update top level sticks 
        var_sticks_ss = np.zeros((2, self.m_T-1))
        self.m_var_sticks[0] = self.m_varphi_ss[:self.m_T-1]  + 1.0
        var_phi_sum = np.flipud(self.m_varphi_ss[1:])
        self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma


    def update_expectations(self):
        """
        Since we're doing lazy updates on lambda, at any given moment
        the current state of lambda may not be accurate. This function
        updates all of the elements of lambda and Elogbeta so that if (for
        example) we want to print out the topics we've learned we'll get the
        correct behavior.
        """
        for w in range(self.m_W):
            self.m_lambda[:, w] *= np.exp(self.m_r[-1] - 
                                          self.m_r[self.m_timestamp[w]])
        self.m_Elogbeta = sp.psi(self.m_eta + self.m_lambda) - \
            sp.psi(self.m_W*self.m_eta + self.m_lambda_sum[:, np.newaxis])
        self.m_Ebeta = ((self.m_eta + self.m_lambda).T / (self.m_W*self.m_eta + \
            self.m_lambda_sum)).T

        self.m_timestamp[:] = self.m_updatect
        self.m_status_up_to_date = True

    def save_topics(self, filename):
        """
        save beta[t][w], P(W=w|Z=t)
        """
        if not self.m_status_up_to_date:
            self.update_expectations()
        f = file(filename, "w") 
        betas = self.m_lambda + self.m_eta
        for beta in betas:
            line = ' '.join([str(x) for x in beta])  
            f.write(line + '\n')
        f.close()

    def hdp_to_lda(self):
        # compute the lda almost equivalent HDP.
        # alpha
        if not self.m_status_up_to_date:
            self.update_expectations()

        sticks = self.m_var_sticks[0]/(self.m_var_sticks[0]+self.m_var_sticks[1])
        alpha = np.zeros(self.m_T)
        left = 1.0
        for i in range(0, self.m_T-1):
            alpha[i] = sticks[i] * left
            left = left - alpha[i]
        alpha[self.m_T-1] = left      
        alpha = alpha * self.m_alpha
        #alpha = alpha * self.m_gamma
        
        # beta
        beta = (self.m_lambda + self.m_eta) / (self.m_W * self.m_eta + \
                self.m_lambda_sum[:, np.newaxis])

        return (alpha, beta)

    def infer_only(self, docs, half_train_half_test=False, split_ratio=0.9, iterative_average=False):

        # be sure to run update_expectations()
        sticks = self.m_var_sticks[0]/(self.m_var_sticks[0]+self.m_var_sticks[1])
        alpha = np.zeros(self.m_T)
        left = 1.0
        for i in range(0, self.m_T-1):
            alpha[i] = sticks[i] * left
            left = left - alpha[i]
        alpha[self.m_T-1] = left      
        #alpha = alpha * self.m_gamma
        score = 0.0
        count = 0.0
        for doc in docs:
            if half_train_half_test:
                (s, c, gamma) = lda_e_step_half(doc, alpha, self.m_Elogbeta, split_ratio) 
                score += s
                count += c
            else:
                score += lda_e_step(doc, alpha, np.exp(self.m_Elogbeta))
                count += doc.total
        return (score, count)


    def most_likely_topic(self):
        
        self.update_expectations()

        self.m_sticks = self.m_var_sticks[0]/(self.m_var_sticks[0]+self.m_var_sticks[1])
        topics_mass = np.zeros(self.m_T)
        left = 1.0
        for i in range(0, self.m_T-1):
            topics_mass[i] = self.m_sticks[i] * left
            left -= topics_mass[i]       
        topics_mass[self.m_T-1] = left   
        self.m_topics_mass = topics_mass  

        self.m_word_topic = np.argmax(self.m_Ebeta.T * self.m_topics_mass, axis=1)


    def get_perplexity(self, test_corpus):
        '''calculate perplexity on the test corpus'''
        # convert the HDP model to the almost idential LDA model
        (lda_alpha, lda_beta) = self.hdp_to_lda()

        total_doc_count = test_corpus.num_docs
        test_score = 0.0
        test_score_split = 0.0
        c_test_word_count = sum([doc.total for doc in test_corpus.docs])
        c_test_word_count_split = 0

        for doc in test_corpus.docs:
            (likelihood, gamma) = lda_e_step(doc, lda_alpha, lda_beta)
            test_score += likelihood
            # thee following three lines of codes are really unnecessary
            (likelihood, count, gamma) = lda_e_step_split(doc, lda_alpha, lda_beta) 
            test_score_split += likelihood
            c_test_word_count_split += count

        self.perplexity = np.exp(-test_score/c_test_word_count)







        


