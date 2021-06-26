import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Note for enterprising students:  There are myriad ways to split sentences for
    this algorithm.  For instance, you might want to exclude punctuation (unless
    it's organized in an email address format) or exclude numbers (unless they're
    organized in a zip code or phone number format).  Clearly this can become quite
    complex.  For our purposes, please split using the space character ONLY.  This
    is intended to balance your understanding with our ability to autograde the
    assignment.  Thanks and have fun with the rest of the assignment!

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    
    #Convert to lowercase
    message = message.lower()    
    
    #Split the message based on space delimiter and store values in a list
    x = message.split(" ")
    
    return x
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least *five messages*.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***

    #Initialize an empty list
    word_list = []
    
    #Loop over all the messages and append the words from each message in a list
    for i in range(0, len(messages)):
        
        #Convert the message string to lowercase and return a list of words 
        #in the input message string by using space separator
        temp_list = get_words(messages[i])
        
        #Remove duplicate words from the list
        temp_list = list(dict.fromkeys(temp_list))
        [word_list.append(x) for x in temp_list]
        
    
    #Create a dict which maps frequencies of words in a list
    freq = {}
    for item in word_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    

    #Select only those words which occur in at least 5 messages
    result = {key:value for (key, value) in freq.items() if value >= 5}
    
    return result
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to *a word of the vocabulary*.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***


    #Initialize a np.array 
    word_matrix = np.zeros((len(messages), len(word_dictionary)))
    

    
    #Loop over all the messages and append the words from each message in a list
    for i in range(0, len(messages)):
        
        #Convert the message string to lowercase and return a list of words in the input message string by using space separator
        temp_list = get_words(messages[i])
         
        dict_count = 0
        
        #Loop over the dictionary
        for x, y in word_dictionary.items():
            occurrences = temp_list.count(x)
            word_matrix[i, dict_count] = occurrences
            dict_count += 1
    
            
    return word_matrix
    #Row is for each message
    #Column is for each word in the dictionary
    #Entry i,j describes the frequency of a given word in the message
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    
    # *** START CODE HERE ***

    #Length of the vocabulary
    vocab_len = matrix.shape[1]
    
    #Boolean indexing for categorizing messages based on Labels
    mask = labels == 1
    
    #Fancy indexing to get Spam messages into a separate matrix
    spam = matrix[mask,:]
    
    #Fancy indexing to get Ham messages into a separate matrix
    ham = matrix[~mask,:]
    
    #Function to calculate phi for spam and ham messages
    def calc_phi(matrix, vocab_len):
        
        #This will take the frequency sum of each word across messages
        word_total = np.sum(matrix,axis=0)
        
        #Calculating the numerator
        num = (1+word_total)
        
        #Calculating the denominator
        den = vocab_len+np.sum(matrix)
        
        return np.divide(num, den)
    
    
    phi_k_y1 = calc_phi(spam, vocab_len)
    phi_k_y0 = calc_phi(ham, vocab_len)
    phi_y = spam.shape[0]/matrix.shape[0]

#     print("\n\n")  
#     print("This is phi_k_y1: ", phi_k_y1)
#     print("This is phi_k_y0: ", phi_k_y0)
#     print("This is phi_y: ", phi_y)
#     print("\n\n")  
    
    return phi_k_y0, phi_k_y1, phi_y
    
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containing the predictions from the model
    """
    # *** START CODE HERE ***
    
    output = np.zeros(matrix.shape[0])
    
#     print("This is matrix shape: ", matrix.shape)
    #558, 1717
    
    #This function returns the log probability for each given input model (spam or ham)
    def calc_prob(matrix, model):
        
        log_prob = np.multiply(matrix, np.log(model))
        
        #Take the sum of logs
        return np.sum(log_prob, axis = 1)
    
    
    phi_notspam = model[0]
    phi_spam = model[1]
    
    
    #We get the log probability for both p(y=0) and p(y=1)
    num = calc_prob(matrix, phi_notspam)
    den = calc_prob(matrix, phi_spam)
    
    ratio_calc = num - den + np.log(1-model[2]) - np.log(model[2])
    ratio_calc = np.exp(ratio_calc)
    
    final_prob = 1/(1+ratio_calc)
    
    output[final_prob > 0.5] = 1
    
    return output
    
    
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    
        
    #Calculate the result as per the formula in PS1(c)
    result = np.log(model[1]) - np.log(model[0])
    
    #Get the index location of top5_words
    top5_index = result.argsort()[-5:][::-1]
    
    #Array to store the final 5 words
    final = []
    
    #Get the words from dictionary in an arr to match their index
    word_arr = []
    
    for key, values in dictionary.items():
        word_arr.append(key) 
    
    for i in top5_index:
        final.append(word_arr[i])
    
    return final
    
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    
    
    svm_pred_list = list(map(lambda sel_radius: svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, sel_radius), radius_to_consider))
    

    svm_accuracy_list = list(map(lambda svm_pred: np.mean(svm_pred == val_labels), svm_pred_list))
    
    max_value = max(svm_accuracy_list)


    max_index = svm_accuracy_list.index(max_value)
    
    return radius_to_consider[max_index]
    
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    util.write_json('spam_dictionary_(soln)', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix_(soln)', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions_(soln)', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words_(soln)', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius_(soln)', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
