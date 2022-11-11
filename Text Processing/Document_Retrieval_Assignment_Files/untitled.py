import math
import time
class Retrieve:
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    
    def __init__(self, index, term_weighting):
        
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.idf = self.calc_idf()
        self.all_docs=self.calc_all_docs()
        self.tfidf_sigma=self.calc_tfidf_sigma()
        
                        
    def calc_all_docs(self):
        self.all_documents=dict()          # create a dictionary containing all terms in the document ID
        for term in self.index:
            for doc_id in self.index[term]:
                if doc_id not in self.all_documents:
                    self.all_documents[doc_id]={}
                    self.all_documents[doc_id]=(term,)*int(self.index[term][doc_id])
                else:
                    self.all_documents[doc_id]+=(term,)*int(self.index[term][doc_id])
        return (self.all_documents)
    
    def calc_sigma(self):
        self.sigma_of_document=dict()
        for word in self.index:
            for doc_id in self.index[word]:
                if doc_id not in self.sigma_of_document:
                    if self.term_weighting=='tfidf':
                        k=(int(self.index[word][doc_id])*self.idf[word])
                    elif self.term_weighting=='tf':
                        k=(int(self.index[word][doc_id]))
                    elif self.term_weighting=='binary':
                        k=1
                    self.sigma_of_document[doc_id]=k**2
                else:
                    if self.term_weighting=='tfidf':
                        k=(int(self.index[word][doc_id])*self.idf[word])
                    elif self.term_weighting=='tf':
                        k=(int(self.index[word][doc_id]))
                    elif self.term_weighting=='binary':
                        k=1
                    self.sigma_of_document[doc_id]+=k**2
    
    def calc_sigma(self):
        self.sigma_of_document=dict()
        for doc_id in self.all_docs:
            self.sq_of_values=[]
            self.word_list=[]
            for word in self.all_docs[doc_id]:
                if word not in self.word_list:
                    self.word_list.append(word)
                    if self.term_weighting=='tfidf':
                        k=(int(self.index[word][doc_id])*self.idf[word])
                    elif self.term_weighting=='tf':
                        k=(int(self.index[word][doc_id]))
                    elif self.term_weighting=='binary':
                        k=1
                    self.sq_of_values.append(k*k) 
            self.sigma_of_document[doc_id]= math.sqrt(sum(self.sq_of_values))
        return (self.sigma_of_document)
                            
    def calc_idf(self):
        self.idf_values=dict()           # a dict which stores term and doc ID with its IDF score
        for term in self.index:
            self.idf_values[term]=math.log(self.num_docs/len(self.index[term].keys()))
        return (self.idf_values)
    
    def calc_tfidf_sigma(self):
        self.sigma_of_document=dict()
        for doc_id in self.all_docs:
            self.sq_of_values=[]
            for word in self.all_docs[doc_id]:
                k=self.idf[word]
                self.sq_of_values.append(k*k)
            self.sigma_of_document[doc_id]= math.sqrt(sum(self.sq_of_values))
        return (self.sigma_of_document)
    
    def calc_tfidf_sigma(self):
        self.sigma_of_document=dict()
        for doc_id in self.all_docs:
            self.sq_of_values=[]
            for word in self.index:
                if doc_id in self.index[word]:
                    k=(int(self.index[word][doc_id])*self.idf[word])
                    self.sq_of_values.append(k**2)
            self.sigma_of_document[doc_id]= math.sqrt(sum(self.sq_of_values))
        return (self.sigma_of_document)
    
    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    
    def for_query(self, query):
        self.t1=time.time()
        self.query=query
        self.terms_in_query=dict()
        for word in self.query:
            if word in self.index:
                if word in self.terms_in_query:
                    self.terms_in_query[word] += 1
                else:
                    self.terms_in_query[word] = 1
        
        def get_doc_IDs(query_terms):
            matching_docs=set()
            self.query_terms=query_terms
            for term in query_terms:
                #if term in self.index:
                matching_docs.update(self.index[term])
            return matching_docs
    
        def normalise(doc_list):
            sigma_of_doc=dict()
            self.doc_list=doc_list
            doc_list_normalised=dict()
            for doc_id in doc_list:
                sq_of_values=[]
                doc_list_normalised[doc_id]={}
                for word in doc_list[doc_id]:
                    k=doc_list[doc_id][word]
                    sq_of_values.append(k*k)
                sigma_of_doc[doc_id]= math.sqrt(sum(sq_of_values))
                for word in doc_list[doc_id]:
                    doc_list_normalised[doc_id][word]=(doc_list[doc_id][word]/sigma_of_doc[doc_id])
            return doc_list_normalised
    
        def calc_cosine_sim_score(query_tw_scores,match_doc_tw_scores):
            self.query_tw_scores=query_tw_scores
            self.match_doc_tw_scores=match_doc_tw_scores
            score=[]
            for term in query_tw_scores:
                if term in match_doc_tw_scores:
                    score.append(query_tw_scores[term]*match_doc_tw_scores[term])
                #else:
                 #   score.append(0)
            return sum(score)
        
        
        self.doc_ID_matching_query=get_doc_IDs(self.terms_in_query)
        
        if self.term_weighting=='tfidf':
        
            tfIDF_of_query=dict()
            for term in self.terms_in_query:
                #if term in self.index:
                tfIDF_of_query[term]=(int(self.terms_in_query[term])*self.idf[term])
        
            tfIDF_of_match_docs=dict()
            for doc_ID in self.doc_ID_matching_query:
                tfIDF_of_match_docs[doc_ID]={}
                for word in self.all_docs[doc_ID]:
                    tfIDF_of_match_docs[doc_ID][word]=((int(self.index[word][doc_ID])*self.idf[word])/self.tfidf_sigma[doc_ID])
                    
                    
            for doc_ID in self.doc_ID_matching_query:
                tfIDF_of_match_docs[doc_ID]={}
                for word in self.index:
                    if doc_ID in self.index[word]:
                        tfIDF_of_match_docs[doc_ID][word]=(int(self.index[word][doc_ID])*self.idf[word])/self.tfidf_sigma[doc_ID]
            
            #for doc_ID in self.doc_ID_matching_query:
            #    tfIDF_of_match_docs[doc_ID]={}
            #    for word in self.all_docs[doc_ID]:
            #        if word in self.terms_in_query:
            #            tfIDF_of_match_docs[doc_ID][word]=((int(self.index[word][doc_ID])*self.idf[word])/self.sigma[doc_ID])
            
            tfIDF_match_docs_normalised=normalise(tfIDF_of_match_docs)
            
            cosine_score=dict()
            for doc_id in self.doc_ID_matching_query:
                cosine_score[doc_id]=calc_cosine_sim_score(tfIDF_of_query, tfIDF_of_match_docs[doc_id])
                
        elif self.term_weighting=='tf':
            
            tf_of_query=dict()
            for term in self.terms_in_query:    
                tf_of_query[term]=int(self.terms_in_query[term])
                    
            #self.tf_of_match_docs=dict()             # a dict to store term frequency values
            #for doc_num in self.doc_ID_matching_query:
             #   self.tf_of_match_docs[doc_num]={}
              #  for term in self.tf_of_query:
               #     #if term in self.index
                #    if doc_num in self.index[term]:
                 #       self.tf_of_match_docs[doc_num][term]=int(self.index[term][doc_num])
        
            tf_of_match_docs=dict()
            for doc_ID in self.doc_ID_matching_query:
                tf_of_match_docs[doc_ID]={}
                for word in self.all_docs[doc_ID]:
                    tf_of_match_docs[doc_ID][word]=int(self.index[word][doc_ID])
        
            tf_match_docs_normalised=normalise(tf_of_match_docs)
        
            cosine_score=dict()
            for doc_id in self.doc_ID_matching_query:
                cosine_score[doc_id]=calc_cosine_sim_score(tf_of_query,tf_match_docs_normalised[doc_id])
        
        elif self.term_weighting=='binary':
            
            binary_of_query=dict()
            for term in self.terms_in_query:
                binary_of_query[term]=1
                    
            binary_of_docs=dict()
            for doc_id in self.doc_ID_matching_query:
                binary_of_docs[doc_id]={}
                for word in self.all_docs[doc_id]:
                    binary_of_docs[doc_id][word]=1
                        
            binary_of_docs_normalised=normalise(binary_of_docs)
            
            cosine_score=dict()
            for doc_id in self.doc_ID_matching_query:
                cosine_score[doc_id]=calc_cosine_sim_score(binary_of_query,binary_of_docs_normalised[doc_id])
        
        else:
            return list(range(10))
        
        sorted_dict=dict(sorted(cosine_score.items(), key=lambda item: item[1], reverse=True))
        self.t2=time.time()
        print(self.t2-self.t1)
        return list(sorted_dict.keys())