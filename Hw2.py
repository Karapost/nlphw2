
# coding: utf-8

# In[5]:


import tensorflow as tf
import os
import numpy as np
from math import ceil
import xml.etree.ElementTree as ET
from util import *


# # Loading Data
# Given a word in a sentence, we transform it in a feature vector which includes the embedding vector and the POS tag. Thus, a sentence is a sequence of features vectors of the same size. The input space of our network will then be all the possibile features vector.
# 
# The output space will be the union of the lemmas of unambiguous words and the possibile senses FOUND in the training data. 
# 
# 
# 
# 

# In[6]:


embedding_dict, hidden_size = load_embeddings('glove.6B.300d.txt')

senses_dict, truesenses_dict, wf_lemmas_dict,senses_per_lemma_dict, out_size = compute_output_space('semcor.data.xml','semcor.gold.key.bnids.txt')

print('The output space size is: ' + str(out_size),flush=True)

n_features = hidden_size + 1 
print('The features vectors have size: ' + str(n_features))

#Window size for the padding
window_size = 30
window_overlap = 5

keep_prob = 0.8


# ### Training Data
# We load the training data with a xml_parser function which uses the previously generated data structures to get a list of sentences, labels, sentences' lengths and possible choices for ambiguous lemmas

# In[7]:


tree = ET.parse('semcor.data.xml')
root = tree.getroot()

tr_sentences,tr_labels,tr_lens,tr_amb_lemmas = xml_parser(root.getchildren(),window_size,window_overlap,embedding_dict,senses_dict,truesenses_dict,wf_lemmas_dict,senses_per_lemma_dict)


# ### Validation Data
# Being the instance_id in the validation data different from the ones found in training data, truesenese_dict needs to be updated

# In[10]:


with open('ALL.gold.key.bnids.txt','r') as f:
    for line in f.readlines():
        instance_id, sense = line.split()
        truesenses_dict[instance_id] = sense


# We then load the validation data but splitting by dataset. The first part just reads the names of dataset in the data. The second part loads the validation data creating dev_data which is organized as follow:
# 
# dev_data = [D_1,D_2,..] where D_x is a dataset
# 
# D_x = [sentences,labels,lens,amb_lemmas] for that dataset
# 
# sentences,labels,lens,amb_lemmas are structed in the same way as before
# 
# 

# In[11]:


tree = ET.parse('ALL.data.xml')
root = tree.getroot()

dev_texts = []                                      #list of lists: text split by datasets
datasets_names = []                                 #list: names of the found datasets

dev_data = []      

#First part
for text in root.getchildren():
    dataset_name = text.attrib['id'].split('.')[0]
    
    if dataset_name not in datasets_names:  # add the new dataset
        datasets_names.append(dataset_name)
        dev_texts.append([])
        
    dev_texts[datasets_names.index(dataset_name)].append(text) #assign the (text) node to the right dataset

#Second part
for dataset in dev_texts:
    
    dv_sentences,dv_labels,dv_lens,dv_amb = xml_parser(dataset,window_size,window_overlap,embedding_dict,senses_dict,truesenses_dict,wf_lemmas_dict,senses_per_lemma_dict)
    
    dev_data.append([dv_sentences,dv_labels,dv_lens,dv_amb])

del dev_texts,dv_sentences,dv_labels,dv_lens, dv_amb,root, tree #removing useless data


# # Network
# The network is structured as following:
# 
# Sentences, labels and lengths are read trought placeholder.There is one layer of BiLSTM. Lastly there are two output layer, one used for the label prediction and the other for pos tag prediction.
# 
#     inputs (sentences) has shape (batch_size,window_size,n_features)
#     targets (labels) has shape (batch_size,window_size,2) (2 = word_label + pos_label)
#     seq_lens (lens) has shape (batch_size)

# In[12]:


tf.reset_default_graph()


# In[13]:


inputs = tf.placeholder(tf.float32, shape=[None,window_size,n_features])
targets = tf.placeholder(tf.int32, shape=[None,None,2])
seq_lens = tf.placeholder(tf.int32, shape=[None])


# In[14]:


lstm_units = 500

cell_fw = tf.contrib.rnn.LSTMCell(lstm_units)
cell_bw = tf.contrib.rnn.LSTMCell(lstm_units)

#Dropout
cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)

outputs,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,sequence_length=seq_lens, dtype=tf.float32)

#Concatenation of the two hidden states
full_output = tf.concat([outputs[0],outputs[1]],axis=-1)
#full_output has shape (batch_size,window_size,lstm_units*2)

#Generating a mask to ignore padding
mask = tf.sequence_mask(seq_lens)
#mask has shape (batch_size,window_size)

#Mask repated in time for the attention layer
mask_repeated = tf.reshape(tf.tile(mask,[1,window_size]),(-1,window_size,window_size))


# ### Attention Layer

# In[15]:


class AttentionWithContext:
    def __init__(self,input_shape, **kwargs):
        super(AttentionWithContext,self).__init__(**kwargs)
        self.build(input_shape)
        
    def build(self,input_shape):
        dense = tf.keras.layers.Dense(input_shape[1],activation=tf.nn.tanh,use_bias=True)
        self.td = tf.keras.layers.TimeDistributed(dense)
        
    def __call__(self,inputs,mask):
        focused_a = self.td(inputs)
        focused_a = tf.exp(focused_a)
        
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            foucsed_a = mask * focused_a
            
        focused_a /= tf.cast(tf.reduce_sum(focused_a,axis=1,keepdims=True) + tf.keras.backend.epsilon(),tf.float32)
        
        focused_features = tf.keras.backend.batch_dot(focused_a,inputs)
        return [focused_features,focused_a]


# In[16]:


att_output, _ = AttentionWithContext(full_output.get_shape().as_list())(full_output,mask_repeated)

output_concat = tf.concat([full_output,att_output],-1)

#Size of the last dimension after the concatenation
aug_features = output_concat.get_shape()[-1]

#Flattening
output_flat = tf.reshape(output_concat,[-1,aug_features])


# ### Fully connected layer

# In[17]:


fc_w = 1024

W_fc = tf.get_variable('W_fc',shape=[aug_features,fc_w],dtype=tf.float32)
b_fc = tf.get_variable('b_fc',shape=[fc_w],dtype=tf.float32) 

h_fc = tf.nn.relu(tf.matmul(output_flat,W_fc) + b_fc)


# Slicing the targets placeholder to extract the label for the word and the pos

# In[18]:


target_label = tf.squeeze(tf.slice(targets,(0,0,0),(tf.shape(targets)[0],tf.shape(targets)[1],1)),-1)
target_pos = tf.squeeze(tf.slice(targets,(0,0,1),(tf.shape(targets)[0],tf.shape(targets)[1],1)),-1)


# ### Loss the label for the word

# In[19]:


W_label = tf.get_variable('W_label',shape=[fc_w,out_size],dtype=tf.float32)
b_label = tf.get_variable('b_label',shape=[out_size],dtype=tf.float32) 

logits_label = tf.matmul(h_fc,W_label) + b_label
logits_label = tf.reshape(logits_label,[-1,window_size,out_size])

losses_label = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_label,labels=target_label) #UNSCALED LOGITS!
#losses_label has shape (batch_size,window_size). Computes the loss for each word

#Ignore the losses from padding
losses_label = tf.boolean_mask(losses_label,mask)
#losses_label has shape (n) where n is the sum of the lens

#Summing the losses
loss_label = tf.reduce_sum(losses_label)


# ### Loss the POS for the word

# In[20]:


W_pos = tf.get_variable('W_pos',shape=[aug_features,n_tags],dtype=tf.float32)
b_pos = tf.get_variable('b_pos',shape=[n_tags],dtype=tf.float32) 

logits_pos = tf.matmul(output_flat,W_pos) + b_pos
logits_pos = tf.reshape(logits_pos,[-1,window_size,n_tags])

losses_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_pos,labels=target_pos)
losses_pos = tf.boolean_mask(losses_pos,mask)
loss_pos = tf.reduce_sum(losses_pos)


# ### Total loss

# In[21]:


total_loss = loss_pos + loss_label

optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.5).minimize(total_loss)


# ### Predictions and Metrics

# In[22]:


preds_label_values = tf.nn.softmax(logits_label)

preds_pos = tf.nn.softmax(logits_pos)
preds_pos = tf.argmax(preds_pos,axis=-1)

preds_pos = tf.boolean_mask(preds_pos,mask)
true_pos = tf.boolean_mask(target_pos,mask)

#Accuracy on POS learning
correct_pos = tf.cast(tf.equal(tf.cast(preds_pos,tf.int32), true_pos),tf.float32)
accuracy_pos = tf.reduce_mean(correct_pos)


# ### Summary scalars

# In[23]:


tf.summary.scalar('loss_label',loss_label)
tf.summary.scalar('loss_pos',loss_pos)
tf.summary.scalar('total_loss',total_loss)

tf.summary.scalar('accuracy_pos',accuracy_pos)

summary = tf.summary.merge_all()


# In[24]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

summary_writer = tf.summary.FileWriter('./summary', sess.graph)


# In[25]:


saver = tf.train.Saver()

if not os.path.exists("model"):
    os.makedirs("model")
    
if tf.train.checkpoint_exists('./model/model.ckpt'):
    saver.restore(sess, './model/model.ckpt')
    print("Previous model restored.")


# In[27]:


epochs = 4

batch_size = 32
batch_index = 0
num_batches_per_epoch = ceil(len(tr_lens)/batch_size)

n_iterations = num_batches_per_epoch * epochs

for ite in range(n_iterations):
    
    if ite % 10 == 0:
        print('Iteration: ' + str(ite) )
        
    #Batch
    batch_input = tr_sentences[batch_index*batch_size:(batch_index+1)*batch_size]
    batch_label = tr_labels[batch_index*batch_size:(batch_index+1)*batch_size]
    batch_lens = tr_lens[batch_index*batch_size:(batch_index+1)*batch_size]
    batch_dv = tr_amb_lemmas[batch_index*batch_size:(batch_index+1)*batch_size]
    
    batch_index = (batch_index + 1 ) % num_batches_per_epoch
    
    if batch_index % 50 == 0:
        saver.save(sess, './model/model.ckpt')
        print('Batch scores:')
        predicted_values,target_values,acc = sess.run([preds_label_values,target_label,accuracy_pos],feed_dict={ inputs : batch_input, targets : batch_label, seq_lens : batch_lens})
        precision,_,f1_score = scores_senses(predicted_values,target_values,batch_dv)
        print('Training: precision on senses {} f1_score on senses {} accuracy on pos {}'.format(precision,f1_score,acc),flush=True)
        
        print('Development set scores:')
        for dataset,name in zip(dev_data,datasets_names):
            predicted_values,target_values, acc = sess.run([preds_label_values,target_label,accuracy_pos],feed_dict={ inputs : dataset[0],targets : dataset[1], seq_lens : dataset[2]})
            precision,recall,f1_score = scores_senses(predicted_values,target_values,dataset[3])
            print('{} : precision on senses {} recall on senses {} f1_score on senses {} accuracy on pos {}'.format(name,precision,recall,f1_score,acc),flush=True)
        
    sess.run(optimizer,feed_dict={ inputs : batch_input, targets : batch_label, seq_lens : batch_lens})
    


# # Test Data

# In[ ]:


test_sentences = []

with open('test_data.txt','r') as f:
    for line_sentence in f.readlines():
        words = line_sentence.split()
        test_sentence = []
        for word in words:
            word_parts = word.split('|')
            
            word_vector = generate_word_vector_pos(word_parts[1],word_parts[2],embedding_dict)
            test_sentence.append(word_vector)
        test_sentences.append(test_sentence)

        

