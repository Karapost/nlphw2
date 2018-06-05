#Util functions

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder


pos_tags = ['.','ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X']
n_tags = len(pos_tags)
le = LabelEncoder().fit(pos_tags)

## load_embeddings ##
# Load previously generated embeddings to build a dictionary label->embedding
# -input
# path_embeddings: path to the .txt file with the embeddings. Format is label<-space->embedding.
# -output
# embedding_dict: dictionary label -> embedding
# hidden_size: size of the embeddings
def load_embeddings(path_embeddings):

	embedding_dict = dict()      

	with open(path_embeddings,'r') as f:

		for embedding_pair in f.readlines():

			label, embedding = embedding_pair.split(' ',1)
			embedding = np.array(embedding.split(),dtype=float)
			embedding_dict[label.strip()] = embedding

	hidden_size = len(embedding)

	return embedding_dict,hidden_size

## compute_output_space ##
# computes the necessary data structure for the output space.
# -input
# xml_file: xml file in input containing the sentences
# ground_truth: txt file containing the babelsids for each ambiguous lemma
# -output
# senses_dict : dictionary sense -> code
# truesenses_dict: dictionary instance_id -> sense
# wf_lemmas_dict: dictionary lemma (of unambiguos words) -> code
# senses_per_lemma_dict: dictionary lemma (of ambiguos word) -> list of sense codes
# out_size: size of the output space (+1 for padding)
def compute_output_space(xml_file,ground_truth):
    
	tree = ET.parse(xml_file)
	root = tree.getroot()

	senses_dict = dict()                        #dictionary: sense -> code
	truesenses_dict = dict()                    #dictionary: instance_id -> sense
	wf_lemmas_dict = dict()                     #dictionary: lemma (of unambiguos words) -> code
	senses_per_lemma_dict = dict()				#dictionary: lemma (of ambiguos word) -> list of sense codes

	#code 0 is used for padding
	code = 1
	#Senses
	with open(ground_truth,'r') as f:
		for line in f.readlines():
			instance_id , sense = line.split()

			truesenses_dict[instance_id] = sense
			if sense not in senses_dict:
				senses_dict[sense] = code
				code += 1

	#Lemmas
	for text in root.getchildren():
		for sentence in text:
			for word in sentence:
				lemma = word.attrib['lemma']
                
                #Unambiguous lemma
				if word.tag == 'wf' and lemma not in wf_lemmas_dict: #add the new lemma to the dictionary	
					wf_lemmas_dict[lemma] = code
					code += 1
                    
                #Ambiguous lemma
				elif word.tag == 'instance':
                    # Take a look at the sense
					sense = senses_dict[truesenses_dict[word.attrib['id']]]
                    # First time we found a sense for the lemma
					if lemma not in senses_per_lemma_dict:
						senses_per_lemma_dict[lemma] = []
                    # The new sense is added to the possible senses of the lemma
					if sense not in senses_per_lemma_dict[lemma]:
						senses_per_lemma_dict[lemma].append(sense)

    
	out_size = code

	return senses_dict,truesenses_dict,wf_lemmas_dict,senses_per_lemma_dict,out_size

## generate_word_vector ##
# returns the word vector using the lemma 
# input
# lemma: lemma of the word
# pos: pos of the word
# embedding_dict: dictionary label -> embedding
# output
# word_vector: the word embedding for the lemma + pos value
def generate_word_vector(lemma,pos,embedding_dict):
    
	pos_value = le.transform([pos])[0]
    
	if lemma in embedding_dict:
		wv = embedding_dict[lemma]
	else:
		wv = embedding_dict['unk']
    
	return np.append(wv,pos_value)

## generate_label ##
# Returns a pair of label for the word. The first label depends on the lemma. If the lemma in unambiguous then
# the first label is the lemma, otherwise it is the correct sense. The second label is the pos tag.
# -input
# word: child node of some sentence node
# senses_dict : dictionary sense -> code
# truesenses_dict: dictionary instance_id -> sense
# wf_lemmas_dict: dictionary lemma -> code
# -output
# label: code of the label + code for the pos

def generate_label(word,senses_dict,truesenses_dict,wf_lemmas_dict):
    
	pos = word.attrib['pos']
	code_pos = le.transform([pos])[0]

    #Unambiguous
	if word.tag == 'wf':
		lemma = word.attrib['lemma']
		if lemma in wf_lemmas_dict:
			code_label = wf_lemmas_dict[lemma]
		else:
			code_label = 0            #Predict 0 for the unk lemma (don't care)
            
    #Ambiguous
	elif word.tag == 'instance':
		sense = truesenses_dict[word.attrib['id']]
		if sense in senses_dict:
			code_label = senses_dict[sense]
		else:
			code_label = -1          #The sense is unknown by the system. Taken into account by the recall.
        
	return np.append(code_label,code_pos)

## generate_code_choices ##
# Returns all possible choices for a lemma (limiting the output space). If the word is a unambiguous lemma then 
# we don't care about the choice. If the lemma is ambiguous then we return all possible senses. If the system 
# doesn't recognize the sense of the lemma it returns -1 which will be taken into account by the recall.
# -input
# word: child node of some sentence node
# senses_per_lemma_dict : dictionary lemma (ambiguous) -> list of possible codes
# -output
# val: 0,-1 or a list
def generate_code_choices(child,senses_per_lemma_dict):
    #Unambiguous lemma
	if child.tag == 'wf':
		val = 0
	else:
    #Ambiguous lemma
		if child.attrib['lemma'] in senses_per_lemma_dict:   #Else if it has known senses, append the list of code
			val = senses_per_lemma_dict[child.attrib['lemma']]
		else:
			val = -1                                         #Otherwise cannot disambiguate
            
	return val
    
## pad_split ##
# Given in input a list of lists, pad_split pads each list with pad_token to a multiple of window_size
# to build a list of lists of fixed sizes.
# -input
# in_lists: list of lists to be padded
# pad_token: token used to pad the internal lists
# window_size: fixed size of the window
# window_overlap: overlap between two consecutive windows
# -output
# padded_lists: list of lists of padded & split lists
# original_lengths: the original lenghts of the lists
def pad_split(in_lists,pad_token,window_size,window_overlap):


	padded_lists = []
	original_lengths = []

	for in_list in in_lists:

		#Splitting
		list_parts = [in_list[i:i + window_size] for i in range(0, len(in_list), window_size) ] #-window_overlap
		len_parts = [len(piece) for piece in list_parts]

		#Padding could be required just for the last element
		diff = window_size - len_parts[-1]

		for i in range(diff):
			list_parts[-1].append(pad_token)

		padded_lists += list_parts
		original_lengths += len_parts


	return padded_lists, original_lengths

## xml_parser ##
# parse a xml file given the text nodes (padding applied). It uses all previously generated data structures
# to produce a list of (fixed size) sentences, labels and lens used to feed the netwok
# -input
# texts: list of text nodes in a xml file
# window_size: fixed size of the window over the sentence
# window_overlap: overlap between windows
# embedding_dict: dictionary label -> embedding
# senses_dict : dictionary sense -> code
# truesenses_dict: dictionary instance_id -> sense
# wf_lemmas_dict: dictionary lemma (of unambiguos words) -> code
# senses_per_lemma_dict: dictionary lemma (of ambiguos word) -> list of sense codes
# -output
# sentences: parsed sentences, each word is replaced by its feature vector. Each sentence has a fixed size.
# labels: vector of labels, same structure of sentences
# lens: list of the original lengths of the sentences
# code_choices: list of possible codes for each word in the sentence
def xml_parser(texts,window_size,window_overlap,embedding_dict,senses_dict,truesenses_dict,wf_lemmas_dict,senses_per_lemma_dict):
    
	sentences = []                                   
	labels = []                                      
	lens = []                                        
	code_choices = []                                 
    
	for text in texts:
		for sentence in text:
            
			sentence_words = []
			labels_words = []
			code_choices_words = []
        
			for child in sentence.getchildren():
                
				sentence_words.append(generate_word_vector(child.attrib['lemma'],child.attrib['pos'],embedding_dict))
				labels_words.append(generate_label(child,senses_dict,truesenses_dict,wf_lemmas_dict))
				code_choices_words.append(generate_code_choices(child,senses_per_lemma_dict))
    
			sentences.append(sentence_words)
			labels.append(labels_words)
			code_choices.append(code_choices_words)
    
    #Padding
	n_features = len(sentences[0][0])
	n_labels = len(labels[0][0]) 
    
	sentences,lens = pad_split(sentences,np.zeros(n_features),window_size,window_overlap)
	labels,_ = pad_split(labels,np.zeros(n_labels),window_size,window_overlap)
	code_choices,_ = pad_split(code_choices,0,window_size,window_overlap)
    
	return sentences,labels,lens,code_choices


## scores_senses ##
# computes the precision,recall and the f1 score taking into account only the senses
# -input
# prediction_values: prediction list of lists of probabilities over the output space
# true_labels: true labels (int codes)
# mask_values: list of values, 0 for ambiguous lemma, -1 for unknown senses, list of codes for known senses
# -output
# precision: correct_senses/disambiguated_senses
# recall: correct_sense/correct_senses+skipped_senses
# f1_score: f1 score
def scores_senses(prediction_values,true_labels,mask_values):
    
	correct_senses = 0
	disambiguated_senses = 0
	skipped_senses = 0
    
	for snt_pre, snt_tru, snt_msk in zip(prediction_values,true_labels,mask_values):  #Cycling the batch
		for vect_pre, val_tru, val_msk in zip(snt_pre,snt_tru,snt_msk):               #Cycling the sentence
            
			if val_msk == 0: #Either wf_lemma or padding, don't care
				continue
			if val_msk == -1: #Avoid to predict the sense
				skipped_senses += 1
				continue
                
            #Ambiguous lemma, choose only from the senses related to it
            
			disambiguated_senses += 1
            
			prediction_for_lemma = val_msk[np.argmax(vect_pre[val_msk])]
			if val_tru == prediction_for_lemma:
				correct_senses += 1
                
	precision = correct_senses/disambiguated_senses
	recall = correct_senses/(correct_senses+skipped_senses)
	f1_score = (2*precision*recall)/(precision + recall)
	return precision,recall,f1_score