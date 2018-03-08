# A Novel Sentence Similarity Model with Word Embedding based on Convolutional Neural Network
The paper was published on [Concurrency and Computation:Practice and Experience](http://onlinelibrary.wiley.com/doi/10.1002/cpe.4415/full).  
This is the experimental source codes in this paper.  
## How to use
Download pre-trained word embeddings from [GloVe](https://nlp.stanford.edu/projects/glove/).  
Then you need to change the directory of the word embeddings and self.len_of_word_vector (word vector dimension) in the model.py.   

You can run the model by:
```
python train.py
```

We used two kind of corpuses to test our model--SICK and MSRP, and we created two folder ForSICK and ForMSRP for these two kind of corpuses. And .txt files of these corpuses has been contained in the dataset file.  
If you want to download SICK and MSRP data set, please click here([SICK](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools)&[MSRP](https://research.microsoft.com/en-us/downloads/607d14d9-20cd-47e3-85bc-a2f65cd28042/)).

**Attention：** in the ForSICK, txt file SICK_tt.txt was merged by SICK_train.txt and SICK_trial.txt.
