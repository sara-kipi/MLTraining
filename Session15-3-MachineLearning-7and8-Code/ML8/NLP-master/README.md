# NLP
Natural Language processing

https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275

https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/

https://gumroad.com/l/nlpforhackers?utm_source=blog&utm_medium=side_banner

http://www.shivambansal.com/blog/text-classification-guide/

https://medium.com/vickdata/detecting-hate-speech-in-tweets-natural-language-processing-in-python-for-beginners-4e591952223

https://medium.com/emergent-future/spam-detection-using-neural-networks-in-python-9b2b2a062272

Upsampling code:-
from sklearn.utils import resample

train_majority = train[train['label'] == 0]
train_minority = train[train['label'] == 1]
print(train_majority.shape,train_minority.shape)

#
train_minority_upsampled = resample(train_minority, 
                                 replace=True,    
                                 n_samples=len(train_majority),   
                                 random_state=123)
train_upsampled = pd.concat([train_minority_upsampled, train_majority])
train_upsampled['label'].value_counts()
