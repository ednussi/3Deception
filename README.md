# 3Deception

## Deceptive Information Detection through Computer Vision Analysis of Facial Micro-Expressions from Depth Video Stream

By: Eran Nussinovitch, Gregory Pasternak

Under the supervision of Prof. Daphna Weinshall

Advisors:Prof. Gershon Ben-Shakhar Ms. Nathalie Klein-Selle

![](https://github.com/ednussi/3deception/blob/master/display/figure1.PNG)

## Introduction
### Background
Lying and deceiving is dated as early as the beginning of human communication itself. Our society recognizes and even embraces (Santa, White lies, Secret Agents) some parts of this ability to hide or manipulate the truth. In general, whether a person is bluntly lying or simply leaving out facts in order to deceive us - we want to know the whole picture. Knowing the truth gives us better understanding of the situation and the ability to make intelligent choices.

More than 75% of our communication is nonverbal, with 55% accounted for facial expressions [20]. Furthermore some of the communication we express is not under our control, and might correlates to our underlying true feeling on the matter. In addition studies show that it is cognitively harder to maintain a lie [21]. First you need to fabricate some story to conceal the truth. Than you must keep in line with your own fabrication and make accord for your lie in every following statement you make - to make sure you are not contradicting yourself. It is believed that in most part of the untrained population this cognitive hardship costs in involuntary expressions that leak out which make lie-detecting an achievable fit by experts.

Throughout the years numerous research works were conducted in order to better understand the human deception mechanism and find those giveaways. Some of the recent computer vision based works introduce eye-blinks and pupil dilation patterns, eulerian video magnification techniques and micro-expressions analysis. Continuing recent ground-breaking works with depth cameras, our project approaches the lie detection challenge from the side of facial motion analysis using modern computer vision and machine learning techniques.

Even with all the new work the most common &quot;accepted&quot; solution today is a polygraph which involves recording and analysing of skin conductance, body heat, blood pressure and respiration - all require physical connection to the subject which is problematic as we explained above. In recent years it has become more common to develop less intrusive, computer vision based solutions. This project follow in their footsteps.
###Previous work
Current computer vision techniques for lie detection are:
1. Eye blinks patterns - based upon the experimentally proven hypothesis derived from eye blink literature that liars experience more cognitive demand than truth tellers. Thus, their lies are associated with a significant decrease in eye blinks, directly followed by an increase in eye blinks when the cognitive demand ceases. [16]
2. Pupillary size response - more recent developments found that there is also a high correlation connection between deceptive response and changes in pupil size [17
3. Eulerian video magnification - using SVM and HOG descriptors fed into Neural Network with backPropagation in order to successfully identify lies. [18]
4. Pure image processing/supervised machine learning - using trained SVM to label face image histogram to decide if the face expresses one of predefined expressions or it is neutral. [19]

Even though lie detection is not a new field of research, It has never been tried before to recognize deception using video stream obtained by 3D cameras. Prof. Weinshall emotion detection team&#39;s results [1] suggests that just like the human perception, a depth dimension helps us to better model and understand the world in a more realistic manner, and in particular more precisely identify human facial expressions. Together with their experience we implemented an algorithm pipeline for lie detection.

In essence, we built a computer system that is able to produce a classifier which decides if a subject responds to truthful information, by analyzing a video stream of the subject passing CIT-like [3], [11], [12], [14] questionnaire acquired with depth camera. We introduce a novel technique for learning subject-tailored classifier. By this we eliminate human bias and the need in physical contact with subject at time of the test, which raises extra consciousness in the subject. We developed a simple, portable and quickly deployed solution.

Our project's main purpose is to predict if an individual responds to a truthful information. We evaluate our system through traditional machine learning techniques.

We validate all trained classifiers with previously unseen data from validation set and choose the one that gives lowest error rates and then provide a test dataset (also previously unseen) to the chosen classifier in order to ensure its reliability and to improve its continuous learning.

## Methods and Materials
### Terminology

* Buffer item - research shows that we tend to have an initial strong responses to anything new. Thus, In accordance with previous works [2] we introduced a buffer answer designed to absorb the initial orienting response at each new question to the subject in order to reduce noise.
* Critical and control items - these are the real truthful (facts) and untruthful answers accordingly.
* Action Unit - a muscle or a number of muscles on the subject&#39;s face that is being tracked through the recording process. We used the commercial software Faceshift [5] to extract quantitative measures of over 50 such AUs from recordings of facial activity.
### Overview
Our solution paradigm strategy consists of 2 main parts:

* Data Acquisition
  * Create a psychologically correct questionnaire.
This part required research and consideration of known psychological effects which take place in any human interaction and interview. We built the questionnaire to target and isolate the effect of the lying mechanism and reduce unwanted noise.
  * Acquire the data.
Record a 3D video stream of the interview process, using Apple PrimeSense Carmine 1.09.

* Data Analysis
Apply statistical and signal processing methods on the acquired data to extract relevant features and produce a classifier which given a video record will decide if the received response corresponds to deception or not. An overview of the predictive model is presented in figure 2

![](https://github.com/ednussi/3deception/blob/master/display/figure2.PNG)

Figure 2: Algorithm components overview

### Interview
Since our project involves 3D camera recordings in order to build personal classifier for detecting deception, it was up to us to obtain appropriate dataset due to lack of such open sources. We had to plan the process of acquiring the data to details and take into account various psychological aspects we human are subjects to.
### Participants
Nine Hebrew University of Jerusalem undergraduate and graduate students (4 female and 5 males) participated in the experiment voluntarily. Participants were in the ages of 24-29 with mean age of 27 (std=1.41). All participants signed a consent form indicating that participation was voluntary and that they could withdraw from the experiment at any time without penalty.
### Questionnaire construction
Our questionnaire is based on CIT test, which proved its validity in previous research throughout the years [3]. The questionnaire is composed of 5 types of critical questions:

1. What is your name?
2. What is your surname?
3. What is your mother&#39;s name?
4. What is your birth country?
5. What is your birth month?

Each question is designed to contain some part of constant information which was true throughout the subject entire life. Thus the information is presumably not ambiguous to the subject. Individuals that the information has changed in their lifespan were disqualified from participating in the experiment.

Studies show that we have strong reactions to things we hold dear, are close to or in general have feelings towards it, whatever those feelings may be. To remove such biased answers we introduced an online registration form which included questions regarding subjects&#39; personal information. Subjects were given a list of 30 names of women, men and countries for them to mark all of the possibilities which appeared close, related or important in any particular way to them. The possible answers that were later presented to each subject in the interview were produced from the remaining answers which were not considered by themselves to be important. Since there are only 12 possibles months in the year, people which marked more than 9 months as critical to them were disqualified to continue their participation in the experiment.

After each question there were 3 answered that followed - a buffer item followed by random ordered critical and control answers. The questions appeared in sessions. Each session was composed of all of the 5 types of questions repeated 5 times in random order [6]. At the beginning of every session the subject were instructed to tell only truth or only lie for that entire session. There were 4 sessions, 2 for truth-telling and 2 for lie-telling summing up for a total of 200 samples tagged. As seen in figure 3, at the end we had for each question type 4 types of responds: &quot;Yes for truth&quot;, and &quot;No for truth&quot; that were produced from the truth sessions and &quot;Yes for lie&quot; and &quot;No for lie&quot; that were produced from the lie session. Meaning that 200 samples were divided into 10 samples of each kind of answer.

![](https://github.com/ednussi/3deception/blob/master/display/figure3.PNG)

Figure 3

### Interview design
All interviews were hold in the same conference room, lighten up with both natural and artificial light. Apart from the table, two chairs, a laptop, a camera and an additional screen, there was nothing in the room, that is to reduce subject&#39;s possible distraction.

Mean length of the interview was around 30 minutes: 4 sessions with 25 questions each (5 types of question repeated 5 times). There was a one second break between each answer and 5 seconds before the next question. A minimum of 2 minutes break were taken between the sessions, depending on subject&#39;s fatigue and was resumed when the subject was ready to carry on.

Every 5-10 questions, depending on the subject concentration, the subject was requested to answer a simple mathematical question. This was both to shift one&#39;s thinking away from the questionnaire in order to reduce boredom and fatigue in the subjects.

### Facial expression recording

Throughout the interview process, the subject&#39;s face was recorded to a single video movie (including breaks, math questions etc.), and after the interview ends, timeseries of all Action Units were exported to a comma-separated values table.

### Predictive models

In this step we use raw facial expression time series recorded during the interview and their according context-aware per-frame metadata to extract per-answer features, split resulting dataset to train, validation and test parts using 2 different methods, and finally train a linear predictive model (binary Support Vector Classifier). Given facial expression features of a recorded answer from a subject&#39;s interview (represented as vector in R^638), it aims to predict whether this answer is truthful or not.

Since in our experiment the number of answers in questionnaire interview was only 200 per subject, full feature vector representation would lead to overfit and result in very poor prediction power. Therefore we first selected most significant features using Spearman correlation of each feature to the target label in train set, and then reduced its dimension using principal component analysis (PCA).

In order to optimize the hyperparameters of the pipeline, such as number of top correlated features to use; PCA dimension to reduce the feature vector to; whether or not to perform PCA dimensionality reduction on the feature vector as a whole or on distinct groups of features; and SVC penalty parameter (_&quot;C&quot;_), we applied cross-validation on different splits of train and validation sets, and then selected the parameter which gave best mean validation accuracy over all such splits.

### Data Preprocessing and Feature Extraction
#### Frame grouping, quantization and normalization
First, each Action Unit time series in the whole video record were quantized over time using K-Means (K=4) (quantized series is needed later for feature extraction). Then, using the metadata which was recorded along with each frame, both raw frames and quantized ones were grouped by the answer they belong to, forming 200 raw and 200 quantized sets, with mean number of frames for each answer equals 64.28, and standard deviation of 17.54. Such std is due to the diversity of response time of the subject depending on question, presented answer, and session type.
##### Features
Using the intensity level representation of each action unit as a vector in R^51, 4 types of features were computed:

1. **Moments:** first 4 moments (mean, variance, skewness and kurtosis) were calculated for each Action Unit in each answer set.
2. **Discrete States Features:** for each quantized Action Unit signal, the following characteristics of facial activity were computed for each answer:
  * Activation Ratio: proportion of frames with intensity level greater than zero (the action unit was active)
  * Activation Length: mean number of frames for which there was continuous action unit activation
  * Activation Level: mean intensity of action unit activation (quantized)
  * Activation Average Volume: mean activation level of all action units, was computed once for each answer.
3. **Dynamic Features:** a transition matrix _M_ was generated from quantized signals, measuring the number of transitions between the 4 levels, and three features were calculated for each action unit based on it:
  * Slow Change Ratio: proportion of transitions with small level changes (difference of 1 quanta)
  * Fast Change Ratio: proportion of transitions with large level changes (difference of 2 or more quanta)
  * Change Ratio: overall proportion of transitions (sum of previous two)
4. **Miscellaneous:** including maximal and minimal intensity of each action unit, and number of smiles for left and right lip corners and for both, and number of blinks for each eye separately and for both in each answer. Number of smiles was calculated by taking number of peaks in lip corner signals, where peak is a local maximum which is higher than its surrounding points by at least 0.75.
#### Cross-validation k-fold split
In this part, question type refers to one of five types of questions that were presented in the questionnaire (_name_, _surname_, _mother name_, _birth month_, _birth country_); session type refers to one of two types of sessions that were recorded during the experiment (_truth_, _lies_).

In order to perform cross-validation, starting with single subject data set of 200 samples and 638 features, we used 2 split 
:

1. Question left-out (SP):
  * Choose question type for test set (1 out of 5)
  * Choose question type for validation set (1 out of 4 remaining)
  * Use remaining question types as train set (3)

This was repeated for all combinations of train/validation/test question types to get 20 folds train/validation folds for all test question types

2. Question/Session left-out (QSP):
  * Split 4 recorded sessions to two pairs (train and test) such that in each pair there are sessions of both types
  * In the test pair, choose question type for test set (1 out of 5)
  * In the train pair, choose question type for validation set (1 out of 4 remaining, excluding the test one)
  * Use remaining question types as training set (3)

Naturally, there is a possible bias of the algorithm based on the position of session in the interview process, as the person may tire in later sessions, and this state is consistent for all answers in that session. QSP split method is aimed to eliminate this bias, by only testing the answers from sessions that were never seen in the training process.

In all split methods, the target label is chosen such that our classifier determines if the current answer that is being shown to the subject is the critical item (truthful answer).

For example, If the subject is being asked for his name, and his real name is David, the classifier tells, regardless of the subject&#39;s answer, if his response is to the &quot;Truth&quot; - meaning is the truthful answer is now being shown to him.

Hence, if successful, we derive which of the presented untagged answers is the critical one.
### Classifier training
In all experiments we trained linear support vector machine classifiers, using the [implementation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)from scikit-learn package with kernel=&quot;linear&quot;, all other parameters were left default. In order to find optimal penalty parameter (&quot;c&quot;) we ran the training for each train/validation split n=50 iterations, while sampling c parameter from continuous exponential distribution (scale=100).
###Hyperparameters optimization
There are a number of different parameters that are input to the system, which can be divided to two categories:

1. Questionnaire parameters:
  * Number of control answers for each question (answers that are assumed irrelevant for the subject and this are expected to not raise any reaction). Using the previous CIT experiments, the empirically discovered fact that some subjects succeed to recognize the pattern of the relation between control-critical answers number, and to keep the dataset balanced, this parameter was set to 3: first answer is a buffer item, its function is to dump the initial probably irrelevant signal from the question itself (and not from the answers), and it is not considered in the training phase as a valid sample; second and third are the critical item, and single control item randomly selected from the pool of all control items for given question. The order of critical and control items is random.
  * Number of sessions; in order to keep the dataset balanced, should be even (equal number of &quot;truth&quot; and &quot;lies&quot; sessions). The tradeoff is between the dataset size and the level of adaptation of the subject to experiment questions (as every session contains the same questions, the more sessions there are the less signal the subject raises). Empirically was set to 4. Session types alternate.
  * Number of repetitions of every question in a session from the same considerations as previously, was set to 5.
2. Classifier parameters:
  * Number of Action Units to take into account; given a dataset, the training part of time series of all Action Units are first sorted by Pearson correlation to the target series, then the desired number of top AUs are passed to the next phase. Despite the intuition that more AUs can&#39;t do any harm, the reality is that the features calculated from uncorrelated AUs contain no useful information about the true distribution of the samples, thus adding noise to the dataset. Optimal value was found using grid search (min=10, max=51, selected value=44).
  * Number of features; as previously, all features extracted from selected AUs are sorted by Pearson correlation to the target, and then optimal value was found using grid search (min=20, max=462, selected value=74).
  * PCA method: either global or grouped. Given a dataset M where shape=(n, 462):
    * Global method uses regular PCA to obtain M&#39; where M&#39;.shape=(n, d).
    * Grouped method first splits M to 4 sub-matrices such that each contains only the features from according feature group as defined previously, then uses PCA on each to obtain 4 sub-matrices M\_i where M\_i.shape=(n, round(d/4)), and finally stacks them back horizontally.
  * PCA dimension to reduce feature vectors to. Optimal value was found using grid search (min=7, max=min(30, feature\_number), selected value=18).
  * Penalty parameter of linear SVM (&quot;c&quot; parameter) is found using random search as explained previously.

All of the hyperparameters are evaluated with accuracy of correct classification.
## Results
Our dataset consists of 9 recorded interviews. In _Figure 1_, test accuracy of predicting if the subject was shown the critical item (truthful answer) for each question type. We observe consistent accuracy of the intra-subject classifier over most of test questions. Since our classifier is custom built per individual, it is natural that some classifiers tend to be better than others. For example, _Subject 2_ was consistently easier to identify whatever he was shown a deceptive answer; on the other hand, _Subject 9_ proved to be a difficult challenge. We assume that given more test subjects the prediction accuracy will approximate normal distribution.

Some question types answers are consistently harder to predict (e.g. month of birth) over all subjects than others (e.g. surname). Interestingly, country of birth prediction accuracy is highly variated over the subjects. We propose that this is due to difference in personal relation of the subject to the question.

![](https://github.com/ednussi/3deception/blob/master/display/figure4.PNG)

Figure 1: Test accuracy on question type per subject when trained on 4 other types

Overall success could be measured by the mean accuracies over all question types as shown in  _Figure 2_. Mean accuracy over all subjections is 70% (standard deviation = 2.75%). ROC curves of final classifier were produced and displayed on _Figure 3_.

![](https://github.com/ednussi/3deception/blob/master/display/figure5.PNG)

Figure 2: Average accuracy per subject

![](https://github.com/ednussi/3deception/blob/master/display/figure6.PNG)

Figure 3: Receiver Operator Characteristics

Additionally, we present prediction accuracies for two halves of the dataset: in _Figure 4_ the classifier was trained and tested only on answers where the subject says &quot;Yes&quot;, while in _Figure 5_ - only on those where the subject says &quot;No&quot;. It may be noticed that in both cases the average accuracy is higher than on the combined &quot;Yes&quot;/&quot;No&quot; task in previous figures, which is intuitive, as these tasks involve only one type of response and we reduced the problem and trained the classifier from a smaller hypothesis space. In _Figure 6 the_ average accuracy over all question types for &quot;Yes&quot; part is shown: 77% (std=7%), for &quot;Not&quot; part is 73% (std=6%).

These types of classifiers give us the ability to predict whether a &quot;Yes&quot; or &quot;No&quot; answer is a response to being exposed to the truth. Despite being less general, they might have more practical value.

![](https://github.com/ednussi/3deception/blob/master/display/figure7.PNG)

Figure 4: Test accuracy for &quot;Yes&quot; answers

![](https://github.com/ednussi/3deception/blob/master/display/figure8.PNG)

Figure 5: Test accuracy for &quot;No&quot; answers

![](https://github.com/ednussi/3deception/blob/master/display/figure9.PNG)

Figure 6: Average accuracy over all question types

for &quot;Yes&quot; and &quot;No&quot; parts of the dataset

In addition, we held a user study to evaluate human performance for a similar task.

Human evaluators were first presented 160 short video cuts (~2 seconds each) of each answer instance of control and critical items of both &quot;Yes&quot; and &quot;No&quot; responses for training questions and their labels (&quot;true item&quot;/&quot;irrelevant item&quot;), in random order, repeated on request. Then they were presented with 40 test videos to be labeled themselves. Mean accuracy of correct labels from 9 evaluators reached only 54%. No human evaluator succeeded in beating our classifier.

## Conclusions and future work

First we conclude that for an individual based classifier given the restrictions presented in our work it is possible to train to some extent a &quot;Lie detector&quot;. Even though our sample size of 9 subjects is small to make a general statement, yet, there is evidence that further studies might map connections of our natural response for the truth related to subject&#39;s personal information.

Like in most machine learning problems, we came to the realization that the more specific your problem is - the easier it is to train a classifier to solve your problem. For the time being, if we could divide the &quot;Lie detection&quot; problem into smaller sub-pieces we will be able to achieve a robust solution to utilize in real life problems

In our work we set ourselves the goal of proving the connections between facial expressions and truth-telling, without using other strong features from earlier works, e.g. such as time of response [4].

In future work we will perform additional recordings to expand our sample size. We will include and combine our findings with additional factors (e.g. response time, skin conductance) in order to improve overall result. We also wish to create the first open 3D benchmark for lie detection. With the help of police and field agents we want to adapt the questionnaire to practical applications.

## References

[1] Daniel Hadar, Daphna Weinshall, Implicit Media Tagging and Affect Prediction from video of spontaneous facial expressions, recorded with depth camera, December 2016, [https://arxiv.org/pdf/1701.05248.pdf](https://arxiv.org/pdf/1701.05248.pdf)

[2] Breska, Assaf; Zaidenberg, Daphna; Gronau, Nurit; Ben-Shakhar, Gershon, Psychophysiological detection of concealed information shared by groups: An empirical study of the searching CIT.
Journal of Experimental Psychology: Applied, Vol 20(2), Jun 2014, 136-146.
 [http://dx.doi.org/10.1037/xap0000015](http://psycnet.apa.org/doi/10.1037/xap0000015)

[3] Nahari, T., Breska, A., Elber, L., Klein Selle, N., and Ben-Shakhar, G. (2017) The External Validity of the Concealed Information Test: The Effect of Choosing to Commit a Mock Crime. Appl. Cognit. Psychol., 31: 81–90. doi: [10.1002/acp.3304](http://dx.doi.org/10.1002/acp.3304).

[4] Suchotzki, K., Verschuere, B., Van Bockstaele, B., Ben-Shakhar, G., &amp; Crombez, G. (2017). Lying takes time: A meta-analysis on reaction time measures of deception. _Psychological Bulletin, 143_(4), 428-453. [http://dx.doi.org/10.1037/bul0000087](http://psycnet.apa.org/doi/10.1037/bul0000087)

[5] [http://www.faceshift.com](http://www.faceshift.com)

[6] Eitan Elaad, Gershon Ben-Shakhar, Effects of questions&#39; repetition and variation on the efficiency of the guilty knowledge test: A reexamination, Journal of Applied Psychology, Vol 87(5), Oct 2002.

[7] Y. I. Tian, T. Kanade and J. F. Cohn, Recognizing action units for facial expression analysis,  in _IEEE Transactions on Pattern Analysis and Machine Intelligence_, vol. 23, no. 2, pp. 97-115, Feb 2001.

[8] Ekman, Paul; Friesen, Wallace V.; O&#39;Sullivan, Maureen, Smiles when lying, Journal of Personality and Social Psychology, Vol 54(3), Mar 1988, 414-420. [http://dx.doi.org/10.1037/0022-3514.54.3.414](http://dx.doi.org/10.1037/0022-3514.54.3.414)

[9] Gershon Ben-shakhar , Eitan Elaad, The guilty knowledge test (GKT) as an application of psychophysiology: future prospects and obstacles (2002) [http://www.openu.ac.il/personal\_sites/gershon-ben-shakhar/GKTCHAP3.pdf](http://www.openu.ac.il/personal_sites/gershon-ben-shakhar/GKTCHAP3.pdf)

[10] Ekman, Paul; Friesen, Wallace V, Nonverbal leakage and clues to deception, Psychiatry: Journal for the Study of Interpersonal Processes, Vol 32(1), 1969, 88-106.

[11] Ekman, Paul; Friesen, Wallace V, Detecting deception from the body or face, Journal of Personality and Social Psychology, Vol 29(3), Mar 1974, 288-298. [http://dx.doi.org/10.1037/h0036006](http://dx.doi.org/10.1037/h0036006)

[12] Donald J. Krapohl, Concealed Information test (2012)
 [http://www.americanassociationofpolicepolygraphists.org/sites/default/files/downloads/Dale-Austin/Krapohl&#39;s%20CIT.pdf](http://www.americanassociationofpolicepolygraphists.org/sites/default/files/downloads/Dale-Austin/Krapohl%27s%20CIT.pdf)

[13] George Visu-Petra, Mihai Varga, Mircea Miclea, Laura Visu-Petra, When inference helps: increasing executive load to facilitate deception detection in the concealed information test (2012)
 [https://books.google.co.il/books?hl=en&amp;lr=&amp;id=rn9EBAAAQBAJ&amp;oi=fnd&amp;pg=PA148&amp;dq=concealed+information+test&amp;ots=Z5saPztGcn&amp;sig=ptaU0bU3VPjM8sK6KwLR1tPE1lk&amp;redir\_esc=y#v=onepage&amp;q=concealed%20information%20test&amp;f=false](https://books.google.co.il/books?hl=en&amp;lr=&amp;id=rn9EBAAAQBAJ&amp;oi=fnd&amp;pg=PA148&amp;dq=concealed+information+test&amp;ots=Z5saPztGcn&amp;sig=ptaU0bU3VPjM8sK6KwLR1tPE1lk&amp;redir_esc=y#v=onepage&amp;q=concealed%20information%20test&amp;f=false)

[14] Gershon Ben-Shakhar, Current research and potential applications of the Concealed Information Test: an overview [https://books.google.co.il/books?hl=en&amp;lr=&amp;id=rn9EBAAAQBAJ&amp;oi=fnd&amp;pg=PA9&amp;dq=concealed+information+test&amp;ots=Z5saPztGcn&amp;sig=uD8Vp8ExOadZ68e7Lwe3vBcRIPU&amp;redir\_esc=y#v=onepage&amp;q=concealed%20information%20test&amp;f=false](https://books.google.co.il/books?hl=en&amp;lr=&amp;id=rn9EBAAAQBAJ&amp;oi=fnd&amp;pg=PA9&amp;dq=concealed+information+test&amp;ots=Z5saPztGcn&amp;sig=uD8Vp8ExOadZ68e7Lwe3vBcRIPU&amp;redir_esc=y#v=onepage&amp;q=concealed%20information%20test&amp;f=false)

[15] Ewout H. Meijer, Nathalie Klein Selle, Lotem Elber, Gershon Ben-Shakhar, Memory detection with the Concealed Information Test: A meta analysis of skin conductance, respiration, heart rate, and P300 data (2014) [http://onlinelibrary.wiley.com/doi/10.1111/psyp.12239/full](http://onlinelibrary.wiley.com/doi/10.1111/psyp.12239/full)

[16] Birender Singh, Pooshkar Rajiv, Mahesh Chandra, Lie detection using image processing
 [h](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&amp;arnumber=7324092&amp;url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D7324092) [ttp://ieeexplore.ieee.org/xpl/login.jsp?tp=&amp;arnumber=7324092&amp;url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs\_all.jsp%3Farnumber%3D7324092](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&amp;arnumber=7324092&amp;url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D7324092)

[17] Lubow, R. E., &amp; Fein, O. (1996). Pupillary size in response to a visual guilty knowledge test: New technique for the detection of deception. _Journal of Experimental Psychology: Applied, 2_(2), 164-177. [http://dx.doi.org/10.1037/1076-898X.2.2.164](http://psycnet.apa.org/doi/10.1037/1076-898X.2.2.164)

[18] G. K. Chavali, S. K. N. V. Bhavaraju, T. Adusumilli, and V. Puripanda, &#39;Micro-Expression Extraction For Lie Detection Using Eulerian Video (Motion and Color) Magnication&#39;, Dissertation, 2014. [http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A830774&amp;dswid=-9104](http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A830774&amp;dswid=-9104)

[19] [http://www.cs.uwc.ac.za/~ncruz/](http://www.cs.uwc.ac.za/~ncruz/)

[20] Albert Mehrabian Ph.D, [http://www.kaaj.com/psych/smorder.html](http://www.kaaj.com/psych/smorder.html)

[21] Altebrando, Geena, The cognitive effort of lying, [http://hdl.handle.net/1951/65656](http://hdl.handle.net/1951/65656)
