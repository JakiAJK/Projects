import math

class Retrieve:
    # Create new Retrieve object storing index and term weighting 
    
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()            # number of documents in the corpus
        self.idf = self.calc_idf()                                    # IDF values of each term in the inverted index
        self.sigma = self.calc_sigma()                                # document vector size for each scheme and document
                            
    def calc_idf(self):
        idf_values = dict()           # a dict which stores term and doc ID with its IDF score
        for term in self.index:
            idf_values[term] = math.log(self.num_docs/len(self.index[term].keys()))       # IDF of term = log(No. of docs/df of term) where df=no. of docs the given term is present in
        return (idf_values)
    
    def calc_sigma(self):          # to calculate vector sizes for each of the term weight scheme
        sigma_of_document = dict()
        for word in self.index:
            for doc_id in self.index[word]:
                
                if self.term_weighting=='tfidf':                             # tf * IDF values for each term in document
                    value = (int(self.index[word][doc_id])*self.idf[word])
                elif self.term_weighting=='tf':                              # tf values for each term in document
                    value = (int(self.index[word][doc_id]))
                elif self.term_weighting=='binary':                          # binary values for each term in the document
                    value = 1
                
                if doc_id not in sigma_of_document:                          # if doc_ID not already in sigma_doc_list, start addition. If already exists, add the above value
                    sigma_of_document[doc_id] = value**2
                else:
                    sigma_of_document[doc_id] += value**2
                    
        for doc_id in sigma_of_document:                                     # calculating sqrt of the above summation for each doc_ID
            sigma_of_document[doc_id]= math.sqrt(sigma_of_document[doc_id])
        return (sigma_of_document)
    
    def compute_number_of_documents(self):                                   # 3204 documents
        doc_ids = set()
        for term in self.index:
            doc_ids.update(self.index[term])
        return len(doc_ids)
    
    
    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    
    def for_query(self, query):
        
        self.query = query
        
        terms_of_query = dict()                                  # make query as a dictionary of term counts i.e if multiple terms appear in same query, we add 1 to the existing value
        for word in query:
            if word in self.index:                             # we only compute similarity for those terms which exist in index 
                if word in terms_of_query:
                    terms_of_query[word] += 1
                else:
                    terms_of_query[word] = 1
    
        def calc_cosine_sim_score(query_tw_scores, match_doc_tw_scores):   # A function to calculate similarity score for the provided query and normalized document weights
            score = []
            for term in query_tw_scores:
                if term in match_doc_tw_scores:                           # multiply the weights of same term from query and matching document
                    score.append(query_tw_scores[term] * match_doc_tw_scores[term])
            return sum(score)
        
        # 3 choices of term weighting schemes i.e binary, term frequency and TFIDF
        
        if self.term_weighting == 'binary':
            
            binary_of_query = dict()                          # Represent the query as a weighted binary weights vector
            for term in terms_of_query:
                binary_of_query[term] =1                     # calculate binary weights for terms in query
                    
            binary_of_docs = dict()                           # Represent each document as a weighted and normalised binary vector 
            for term in terms_of_query:
                for doc_id in self.index[term]:
                    if doc_id not in binary_of_docs:                                 # to create 2nd level in dictionary, we need to assign a empty list first
                        binary_of_docs[doc_id]= {}
                        binary_of_docs[doc_id][term]= 1/(self.sigma[doc_id])         # divide by sigma to normalise the values
                    else:
                        binary_of_docs[doc_id][term]= 1/(self.sigma[doc_id])
            
            cos_sim_score= dict()                           # Compute the cosine similarity between the query vector and each document vector
            for doc_id in binary_of_docs:
                cos_sim_score[doc_id]= calc_cosine_sim_score(binary_of_query, binary_of_docs[doc_id])
        
        
        elif self.term_weighting == 'tf':
            
            tf_of_query= dict()                              # Represent the query as a weighted tf vector
            for term in terms_of_query:
                tf_of_query[term]= terms_of_query[term]     # calculate tf values for terms in query
                    
            tf_of_match_docs= dict()                        # Represent each document as a weighted and normalised tf vector 
            for term in terms_of_query:
                for doc_id in self.index[term]:
                    if doc_id not in tf_of_match_docs:
                        tf_of_match_docs[doc_id]={}
                        tf_of_match_docs[doc_id][term]= int(self.index[term][doc_id])/(self.sigma[doc_id])         # divide by sigma to normalise the values
                    else:
                        tf_of_match_docs[doc_id][term]= int(self.index[term][doc_id])/(self.sigma[doc_id])
        
            cos_sim_score= dict()                           # Compute the cosine similarity between the query vector and each document vector
            for doc_id in tf_of_match_docs:
                cos_sim_score[doc_id]= calc_cosine_sim_score(tf_of_query, tf_of_match_docs[doc_id])
        
        
        
        elif self.term_weighting=='tfidf':
            
            tfIDF_of_query= dict()                              # Represent the query as a weighted tf-idf vector 
            for term in terms_of_query:
                tfIDF_of_query[term]= terms_of_query[term]*self.idf[term]                                    # calculate tfIDF values for terms in query
            
            tfIDF_of_match_docs=dict()                        # Represent each document as a weighted and normalised tf-idf vector            
            for term in terms_of_query:
                for doc_id in self.index[term]:
                    if doc_id not in tfIDF_of_match_docs:
                        tfIDF_of_match_docs[doc_id]={}
                        tfIDF_of_match_docs[doc_id][term]= (int(self.index[term][doc_id])*self.idf[term])/(self.sigma[doc_id])         # divide by sigma to normalise the values
                    else:
                        tfIDF_of_match_docs[doc_id][term]= (int(self.index[term][doc_id])*self.idf[term])/(self.sigma[doc_id])
            
            cos_sim_score= dict()                           # Compute the cosine similarity between the query vector and each document vector
            for doc_id in tfIDF_of_match_docs:
                cos_sim_score[doc_id]= calc_cosine_sim_score(tfIDF_of_query, tfIDF_of_match_docs[doc_id])
        
        
        sorted_sim_scores = dict(sorted(cos_sim_score.items(), key=lambda item: item[1], reverse=True))      # Rank documents IDs with respect to the similarity score in decending order
        
        return list(sorted_sim_scores.keys())            # Return the ranked documents IDs as a list