# Utilizing ML/DL NLP to Classify Authorship 

We trained machine learning models and neural nets to predict author, author's sex, and author's literary period given small snippet of text from that author’s work using NLTK, Gensim, Doc2Vec, Polygot, and Stanford NER. With minimal tuning, our best predictions as of September 2018 were: 90.4% on sex; 63.47% on literary period (7 periods - 14% baseline); 56.91% on author (14 authors - 7% baseline). 
<br>

## Project Aims

This project had several aims. In addition to giving us a valuable opportunity to skill up on NLP skills, we hoped to explore whether semantic information would be captured and demonstratable through analogous relationships between authors, sexes, and literary periods, akin to the analogous relationships detailed by Tomas Mikolov, <i> et. al. </i>, and discussed in [“Distributed Representations of Words and Phrases and their Compositionality”](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). We also wished to explore generally the relationships between authors, sex, and literary period and whether we could build machine learning models that could discern an individual author's "style" for use in potential future work with RNNs (capturing the style to later train a model to reproduce it). 

## Project Walkthrough

If you’re looking to reproduce this project, please take a look at “Deployment Notes” first to save yourself a headache! Otherwise, without further ado, the project: <br>

### Data Gathering

We began by pulling two works from each author that we pre-selected. We grabbed these texts from [Project Gutenberg](https://www.gutenberg.org/), a free online resource for public domain texts, by using the Gutenberg API package described under the Libraries header. The code we used to pull these texts can be found in the gutenberg_downloads.py file. <br>

Our author and book distribution is displayed before:

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/nlp_authors_and_books.png)

This process of manually identifying - and later in preprocessing, labeling - each author as well as their sex and literary period wasn’t overly time-consuming because of the modest size of our dataset, but could present challenges at scale. That being said, we were able to fashion a balanced, fully labeled datasets of 3,500, 150-word ‘paragraphs’ (totaling 525,000 words) divided equally between 14 authors and 7 literary periods evenly split by sex with relatively little trouble. 

That makes the CSVs in this repo custom labeled datasets that you are <b> both free and encouraged to use or expand upon </b>. 

### Preprocessing

After we gathered and named our texts, we custom-coded a function to clean and tokenize each text and added the tokenized corpus into a dictionary with author names as keys. We took the tokenized corpora and tested three part of speech taggers (Stanford NER, Polyglot, and NLTK) to clean named entities. <br>

These steps can be time-consuming, but we deemed it an important step in preventing data leakage. While not perfect, the removal of many of the named entities (proper nouns) avoids text-specific named entites like "Frankenstein" or period-specific/work-specific named entities like 'King George' from unduly influencing the model (instead of 'style', which is admittedly a hard concept to strike at). 

#### The POS Taggers 

We tried three: 
1) StanfordNER's 7class tagger, 
2) NLTK's built-in POS tagging (based on the Penn Treebank Project model), and 
3) Polyglot's POS tagger. 

As a quick note, read the license section of this README if considering using the StanfordNER library, as its licensed under GNU GPL and therefore has some caveats. <br>

<b> StanfordNER </b>
NLTK has a built in python wrapper for StanfordNER, which is written in Java, called StanfordNERTagger. Stanford NER has three options, but we used the full 7class version, which handles people, places, organizations, dates, times, monetary units, and percentages. <br>

[Installation with instructions and docs here.](https://nlp.stanford.edu/software/CRF-NER.shtml) <br>

The POS taggers from NLTK and Polyglot taggers also worked well, but for the purposes of this (already long) README, we'll proceed with the StanfordNER only. All librarires found under the header of the same name below. <br>

<i>Quick note:</i> We removed named entities at the pre-processing stage, but we removed stopwords and set max/min word appearance requirements at the vectorization stage. <br>

An example of the NER tags that were removed, in this example from, as you probably could have guessed (which is exactly why they needed to be removed) from <i>Connecticut Yankee in King Arthur's Court</i> by Mark Twain.<br>

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/king_arthur.png)
<br>

We were careful to get a sense of just how much data we were losing by culling named entities from the text, and the word loss caused by this process is absolutely fascinating in it of itself. Locke and Wollstonecraft had by far the least word loss. Both of their works are enlightenment era political treatises. <br>

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/stanford_ner_word_loss_percentage.png)
<br>

Word loss was otherwise varied, and might itself give a sense of an author's style. This suggests that while we may have solved certain data leakage problems (Frankenstein), we may have inadvertently taken out something vital: that different periods/styles of writing may vary in how often they use named entities. <br>

It's precisely for that reason that we opted not to stem or lemme here. An author’s choice to use “He had felt happy before” versus “He had achieved happiness before” versus “He had lived happily before” is part of the style of the author.

We then sliced up these cleaned bodies of texts into semi-randomly selected 150-word 'snippets' (an equal number for each author, sex, and period). We ended up with 3,500 snippets, evenly distributed. 

At the end of pre-processing, the snippets looked like the below: 

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/paragraph_example.png)



### Modeling

In the process of modeling, we employed multiple vectorization strategies. For our machine learning model training, we used Scikit-Learn’s bag- of-words (BOW) and TFIDF vectorizations to create both unigram and bigram-based vectorizations in order to experiment with which best represented our data. In general, BOW bigrams performed best across models (although not uniformly). <br> 

And as mentioned earlier, we also experimented with Gensim’s Doc2Vec module to explore whether we could capture semantic relationships between authors and demonstrate analogous relationships between those authors akin to the analogous relationships discovered by Tomas Mikolov, et. al., with Word2Vec. These vectors were used with a feed-forward neural net, although they did not produce results capable of matching the simpler machine learning models. We suspect this is due, at least in part, to our dataset being undersized for deep learning training. 

#### Doc2Vec

Doc2Vec is absolutely fascinating. The paper introducing the 'paragraph vector' that forms the basis for the Gensim Doc2Vec functionality is fairly readable and comes recommended. You can check out the full paper: [Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) 

Our Doc2Vec results are best shown instead of told. <br>

We'll start with the final version (and a nice animation) to keep you intrigued and reading on as we explain.

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/3d_display.gif)

Pretty cool, huh? So here's what's going on. 

To create the images in the gif, we first inferred vectors for each text snippets using a trained Doc2Vec model. Then, we used Principal Component Analysis (PCA) to reduce the dimensonality (we were operating with 20 dimensional vectors for our purposes). The remaining three dimensions only account for around 35-40% of explained variance, but the visualization and interpretability are well worth it. 

Snippets were 'tagged' with an author, author's sex, and text's literary period. We suspected this would cause overfitting, so we used non-tagged vectors when we build our feed-forward neural net, but for the purposes of examining analogous relationships between snippets and the overall outcome of Doc2Vec, we tried both tagged and untagged. You can see an example of the snippet in question, along with its tags. We then infer its spot in vector space and get the cosine similarity for the 'tag' vectors, which can be thought of as the centroid points for each individual cluster of texts tagged with that term, be it 'female' or 'Victorian' or 'Jane Austen'. In this case, the inferred vector is in fact close to 'male' in the sex comparison, 'realism' in the period comparison, and 'Mark Twain' in the author comparison, which is fascinating. 

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/MarkTwain_InferredVector2.png)

The tags in question can be seen a little more clearly in the following example. Note, not all of our inferred vectors were this accurate, nor were our 'untagged' vectors quite as accurate, but we still saw impressive results. We've explored ways to compare how 'accurate' an inferrence is and work is ongoing on that point. 

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/JackLondon_InferredVector_Strong.png)

Finally, exploring the images themselves, as well as toggling some tags on and off, yields some interesting visualizations of the clusters involved. 

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/partial_author_pca.png)


#### Machine Learning Models

We built two versions of Naive Bayes (Bernoulli and Multinomial), given that Naive Bayes tends to perform well with linguistic data, requires no tuning, and is computationally cheap. We only ran the NB models for the binary sex classification. It was a Bernoulli Naive Bayes model that delivered our best test accuracy for classifying by sex: 90.4%. <br>
We also built a ‘test_classifiers’ function that ran, tuned, and returned results for all four vector version (BOW-unigram, BOW-bigrams, TFIDF-unigrams, TFIDF-bigrams) for each classifier we used. The function could also have taken additional vectorized data, so long as they were added to the vector dictionary before it was passed through. <br> <br>

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/test_classifiers_function.png)
<br>
We trained a bagging model, Random Forest, and a boosting model, AdaBoost for author, author's sex, and literary period. We also trained a K-Nearest-Neighbors on author and period. <br>

Our functions returned clear test/train accuracy results for each vectorization method we tried so we could clearly see which model hyperparameters (except in the case of Naive Bayes) tuned best and which vectorizations were proving most successful. 

<b> Naive Bayes Results Example </b>
![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/bernoulli_nb_results.png)

<b> Random Forest Results Example </b> 
![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/rf_author_results.png)

<b> Full Table of Best Results Across Models </b>

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/ml_results_table.png)

#### Neural Nets 
We built a feed-forward neural network models to make predictions based upon sex only. Not only was a binary classification model simpler, but the lack of unique data points for each author or literary period made using a neural net for multi-class classification a bit like using a sledgehammer on a finishing nail. As such, our multi-layer perception (MLP) was used only for classification on sex (binary). If we were to expand our data set, we would utilize softmax and multi-class classfication methodology. 

<b> Neural Net Architecture </b> 

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/MLP%20architecture.png)

Utilizing relu activation and a sigmoid for classification, we found the following results. 

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/mlp.png)


We also tried early stopping:

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/early_stopping.png)


And Dropout, as well as L1 & L2 regularization: 

![Image](https://github.com/mattymecks/nlp-authorship-vectorization/blob/master/images/dropout_l1_l2.png)


The best thing we could do to improve the performance of our neural net is expand the size of our dataset, which is an aim for future iterations of this project. 

### Visualization 

Our primarily method of visualization was through reducing dimensionality using PCA in order to create 3D data visualization of vectors and improve interpretability, as well as in graphing results. 

### Conclusion 

This project was an excellent example of knowing when to use the right tool for the job. There can be an urge after learning about the awesomeness of deep learning techniques to simply apply them to everything. But here, the relatively simplistic NB performed far better than the more advanced neural net, given the relatively limited size of the data and the penchanct of Naive Bayes model assumptions to fare well when dealing with linguistic data. 

Additionally, to the extent that this project was an exploration of the viability of more complicated projects evolving more experimental areas of NLP, we've deemed both Doc2Vec level analogous reasoning and author style reproductions with RNNs worth exploring. 

## Next Steps

Increasing the size of the data set, and finding a way to access more modern books (which would also allow us to add a 'genre' tag to each text) is probably the primary next step. There's also some refactoring that we'd like to do given the time, particulaly with functionalizing repetitive code. And we'd like to test different methodologies and develop a way of keeping a very clear record to ensure we can compare results across both hyperparameters and "super-hyper parameters" like which vectorization method to use. <br>

One of our next steps would also be to build out pipelines for additional classifiers like XGBoost. 

And finally, an additional next step that we were in the process of implementing, but have not yet finished, is to use extra validation data from authors and texts the model had not seen before to test the level of overfitting and see if the models have truly "learned" anything about sex, period, or author style. 

## Deployment Notes

If you want to deploy this project yourself, there a few installation steps you’ll need to follow to get things working.

First install libraries. Everything listed under the libraries header is a pre-req to completely explore the Juypter notebooks. It’s not an exhaustive list because we left off common libraries like Pandas, so there’s a chance there’s a few other dependencies you’ll need. 

Additionally, and this is the only challenging part, you’ll notice an empty folder called “PlaceNERFilesHere.” The Stanford NER library, which you can find (with complete installation instructions below) is sizable (180 MBs). In order to not break GitHub, we had to gitignore out this library. Furthermore, StanfordNER is licensed under the GNU GPL license, which is discussed further under the 'license' section, but presents some challenges. Because we recognized that this presented problems if someone wanted to use our project out-of-box, we've left code in to easily implement Polyglot or NLTK methods instead. There are two TK notebooks that reference the Stanford NER library. If you chose to use it yourself, make your you commented out our paths, and add paths that point to the <i> PlaceNERFilesHere </i> directory, where you should have placed your downloaded and unpacked StanfordNER files.  

## Additional Project Details

### Libraries 

We used the following libaries to complete our project and owe our thanks to their authors! 


<b>Gutenberg</b> <br>
Awesome library for downloading texts directly from Project Gutenberg. Comes with great tools for stripping metadata and pulling by either author or book id. <br>
https://pypi.org/project/Gutenberg/
<br>

<b>Gensim</b> <br>
Gensim is an incredibly library for a variety of NLP tasks, but perhaps its greatest application in our work was its out-of-box implemntation of word2vec and doc2vec. Highly, highly recommended. <br>
https://radimrehurek.com/gensim/index.html
<br>

<b>NLTK</b> <br>
One of the most versatile and oft-utilized of all natural language packages in Python, NLTK needs little in the way of introduction. Docs below. <br>
https://www.nltk.org/data.html
<br>

<b>Polyglot</b> <br>
Polyglot is a “is a natural language pipeline that supports massive multilingual applications.” It really is an incredibly powerful and accessible NLP library with tools in a multitude of languages. <br>
Quick note that Polyglot requires the following to get up and running once installed (it’s in the docs but it can be a bit hard to find): <br>

%%bash
polyglot download embeddings2.en pos2.en

https://pypi.org/project/polyglot/


<b>Stanford NER</b> <br>
You can download the StanfordNER yourself at the link below as well as explore the documentation. I will mention again that StanfordNER is licensed under a GNU GPL license. <br>
https://nlp.stanford.edu/software/CRF-NER.shtml#Download


### Authors
<b>Matt Mecoli</b> - [Linkedin](https://www.linkedin.com/in/mattmecoli/) ; [GitHub](https://github.com/mattymecks) ; [Medium](https://medium.com/@mattymecks)

Matt is a recent graduate of the data science immersive at the Flatiron School. He is a self-proclaimed data nerd and science geek, and is always happy to talk about interesting data science and machine learning projects. He comes from a background in law, but thinks this is way cooler. 

<b>Naoko Suga</b> - [Linkedin](https://www.linkedin.com/in/naoko-suga/) ; [GitHub](https://github.com/NaokoSuga) ; [Medium](https://medium.com/@ns2586)

Naoko is also a recent graduate of the data science immersive at the Flatiron School. She has a background in physics research (Columbia University) and financial analysis.

### Helpful Resources

Here are a few other helpful resources you might want to check out to learn more about NLP or using the libraries mentioned here. 

NLTK has a book guide that we used quite a bit. 
http://www.nltk.org/book/

The RaRe Technologies repo and tutorials were exceedingly helpful (given that they build genesis and give life to word2vec and doc2vec, this is perhaps not surprising). We owe them our thanks and recommend these resources: 

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

https://rare-technologies.com/doc2vec-tutorial/

For help using the StanfordNER in your preferred language. 
https://textminingonline.com/how-to-use-stanford-named-entity-recognizer-ner-in-python-nltk-and-other-programming-languages

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details EXCEPT for the language noted below: 

IMPORTANT LICENSING NOTE: Stanford NER is licensed under the GNU GPL. This means that if you intend to use this project and license it under a permissive license like MIT or Apache, you must leave out the Stanford NER, as we have here. If you use the StanfordNER to build any part of your project, you *must* license it under the GNU GPL as well.

If none of the above paragraph made any sense to you, Matt wrote an article on understanding the law behind common open source licenses for data scientists that can help clear it up: LINK PENDING. 


### Acknowledgments

* The Flatiron School, particularly Forest Polchow and Jeff Katz, for their advice and support.<br>
* Tomas Mikolov, et. al., for the word2vec and doc2vec tools and white papers, which were fascinating and inspiring in pursuing this project.  
