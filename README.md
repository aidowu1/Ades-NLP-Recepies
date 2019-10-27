# Ades-NLP-Recepies
__Exploration of Document Similarity Models__

_Abstract_: 
The need to develop robust document/text similarity measure solutions is an essential step for building applications such as Recommendation Systems, Search Engines, Information Retrieval Systems including other ML/AI applications such as News Aggregators  (that require document clustering) or automated recruitment systems used to match CVs to job specification and so on. In general, text similarity is the measure of how words/tokens, tweets, phrases, sentences, paragraphs and entire documents are lexically and semantically close to each other. Texts/words are lexically similar if they have similar character sequence or structure and, are semantically similar if they have the same meaning, describe similar concepts and they are used in the same context.  
This work will demonstrate a number of strategies for feature extraction i.e. transforming documents to numeric feature vectors. This transformation step is a prerequisite for computing the similarity between documents. Typically each strategy will involve 3 steps, namely: 1) the use of standard natural language pre-processing techniques to prepare/clean the documents, 2) the transformation of the document text into a vector of vectors. 3), calculation of document feature vectors using the Cosine Similarity metric.

Strategies and associated ML/NLP libraries that will be presented are:
  - document pre-processing using [NLTK library](https://www.nltk.org/)
  - feature extraction using Term frequency – Inverse Document Frequency (TF-IDF) with the aid of the [Scikit-Learn library](https://www.nltk.org/)
  - feature extraction using a pre-trained GloVe embedding with the aid of the [Spacy NLP Library](https://spacy.io/usage/vectors-similarity)
  - feature extraction using a trained [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) embedding from scratch with the aid of the [Gensim NLP library](https://radimrehurek.com/gensim/)
  - feature extraction using a trained [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) embedding from scratch also with the aid of the Gensim library
  
Models developed from these strategies will be compared against each other for the Kaggle competition problem of identifying duplicate questions using the [Quora dataset](https://www.kaggle.com/currie32/predicting-similarity-tfidfvectorizer-doc2vec/data). These models will also be used for computing the document similarity of the popular [NPL 20 News Group problem.](https://www.kaggle.com/irfanalidv/suggectedjob) 



## Update Notes:
Currently note that this project is work-in-progress. 
Completed parts of this this project are:
- TFIDF Feature vector Extractor component 
- Cosine Similarity Measure
- Visualization component including options to reduce the feature matrix dimensions to 2D using PCA, MDS, T-SNE and UMAP techniques
- Plots of the Feature Matrix and the Similarity Matrix Heatmap

Demos of the above features to solve 2 problems, namely:
- A toy/contrived problem which demonstates how to compute the similarity between documents in a corpus which contains 24 documents (Book titles) using data provided in this [blog](https://shravan-kuchkula.github.io/nlp/document_similarity/#plot-a-heatmap-of-cosine-similarity-values)
- The popular NLP 20 newsgroups problem with dataset which comprises around 18000 newsgroups posts on 20 topics

Pending items for this project include:
- build a document feature extarction model using a pre-trained GloVe embedding
- build a document feature extarction model using using a trained word2Vec embedding from stratch
- build a document feature extarction model using using a trained doc2Vec embedding from stratch
- compare the performance of models
- add the Quora duplicate problem [see this Kaggle link](https://www.kaggle.com/currie32/predicting-similarity-tfidfvectorizer-doc2vec/data) and use the above models to the problem
    
Although this project is currently work-in-progress, some of the completed componets of the project include:
  - __GenericDataSerializerComponent.py__: used for getting serialised Problem corous data
  - __ProblemSpecificationInterface.py__: used as an interface abstraction to specify a document similarity problem
  - __NLPEngineComponent.py__: used as a NLP pre-processing module for cleaning the raw corpus of text
  - __TFIDFDocmentVectorExtractor.py__: used to build a TF-IDF based feature vector extration model
  - __DocumentFeatureVisualization.py__: used to reduce the dimensions of the feature matrix based on techniques such as PCA, T-SNE, MDS and UMAP.It allows provides the visualization infrastructure to plot similarity Heatmap  and the visualizations of documents in 2D space

You can futher explore these details of the project by running [this Jupyter notebook](https://github.com/aidowu1/Ades-NLP-Recepies/blob/master/Exploration%20of%20Document%20Similarity%20Models/Exploration%20of%20Document%20Similarity%20Models.ipynb)



