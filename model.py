import tensorflow as tf
import numpy as np
import data
import queue
import math

def make_model(lstm_units, embedding_size, max_caption_length, vocab_size, dropout_rate=0.0, stochastic_loss_lambda=0.005):
    vgg_model = tf.keras.applications.vgg16.VGG16()
    base_model = tf.keras.Model(inputs=vgg_model.input, outputs=tf.keras.layers.Reshape(target_shape=(196, 512))(vgg_model.get_layer('block5_conv3').output))
    base_model.trainable = False
    conv_features = base_model(vgg_model.input)
    h_init = tf.keras.layers.Dense(units=lstm_units, activation = 'relu')(tf.keras.layers.Flatten()(conv_features))
    h_init = tf.keras.layers.Dropout(rate=dropout_rate)(h_init)
    c_init = tf.keras.layers.Dense(units=lstm_units, activation = 'relu')(tf.keras.layers.Flatten()(conv_features))
    c_init = tf.keras.layers.Dropout(rate=dropout_rate)(c_init)
    lstm = tf.keras.layers.LSTM(units=lstm_units, return_state=True)
    attention = tf.keras.layers.Attention()
    token_inputs = [tf.keras.Input(shape=(1,)) for _ in range(max_caption_length)]
    input_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim = embedding_size)
    output_net = tf.keras.layers.Dense(units = vocab_size, activation = 'softmax')
    attention_projection = tf.keras.layers.Dense(units=512)
    output_symbols = []
    attention_scores_list = []
    for i in range(max_caption_length - 1):
        attention_query = attention_projection(h_init)
        attention_query = tf.keras.layers.Dropout(rate=dropout_rate)(attention_query)
        attention_input, attention_scores = attention([tf.keras.layers.Reshape(target_shape=(1, 512))(attention_query), conv_features], return_attention_scores=True)
        attention_scores_list.append(attention_scores)
        token_input = input_embedding(token_inputs[i])
        token_input  = tf.keras.layers.Reshape(target_shape=(1, embedding_size))(token_input)
        lstm_input = tf.keras.layers.concatenate([attention_input, token_input])
        output, h_init, c_init = lstm(inputs=lstm_input, initial_state=[h_init, c_init])
        output_symbols.append(tf.keras.layers.Reshape(target_shape=(1, vocab_size))(tf.keras.layers.Dropout(rate=dropout_rate)(output_net(output))))
    all_ones = tf.keras.initializers.Constant(1.)(shape=(1, 196))
    attention_sum = tf.keras.layers.Sum()(attention_scores_list)
    attention_diff = tf.keras.layers.Subtract()([all_ones, attention_sum])
    double_stochastic_loss = tf.keras.layers.Reshape(target_shape=(1,))(tf.keras.layers.dot([attention_diff, attention_diff], axis=-1))

    def custom_loss(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_cross_entropy(y_true=y_true, y_pred=y_pred) + stochastic_loss_lambda*double_stochastic_loss
   

    output_array = tf.keras.layers.concatenate(output_symbols, axis = -2)
    final_model = tf.keras.Model(inputs=[base_model.input] + token_inputs, outputs=output_array)
    final_model.compile(optimizer='Adam', loss=custom_loss)
    return final_model

def beam_search(model, image, max_caption_length, start_symbol, stop_symbol, vocab_size, beam_width):
    def update_candidate(q, candidate, added_symbol, candidate_prob):
        if q.qsize() < beam_width:
            ccopy = candidate.copy()
            if added_symbol is not None:
                ccopy.append(added_symbol)
            q.put((candidate_prob, ccopy))
        else:
            alternate_prob, alternate = q.get()
            if alternate_prob < candidate_prob:
                ccopy = candidate.copy()
                if added_symbol is not None:
                 ccopy.append(added_symbol)
                q.put((candidate_prob, ccopy))
            else:
                q.put((alternate_prob, alternate))

    candidates = queue.PriorityQueue()
    candidates.put((math.log(1.0), [start_symbol]))
    length_increment = 1
    while length_increment > 0:
        new_candidates = queue.PriorityQueue()
        max_candidate_length = max([len(l[1]) for l in candidates.queue])
        while candidates.qsize() > 0:
            candidate_prob, candidate = candidates.get()
            candidate_len = len(candidate)
            if candidate_len == max_caption_length or candidate[-1] == stop_symbol:
                update_candidate(new_candidates, candidate, None, candidate_prob)
            else:
                padded_candidate = [np.asarray([[c]]) for c in candidate]
                padded_candidate.extend([np.asarray([[stop_symbol]])] *(max_caption_length - len(candidate)))
                #print(padded_candidate)
                predict_out = model.predict([image, padded_candidate], batch_size=1)
                #print('OK')
                for i in range(vocab_size):
                    prob = predict_out[0][candidate_len - 1][i]
                    new_candidate_prob = math.log(prob) + candidate_prob
                    update_candidate(new_candidates, candidate, i, new_candidate_prob)
        #print('PROCESSED')
        candidates = new_candidates
        length_increment = max([len(l[1]) for l in candidates.queue]) - max_candidate_length
        #print(candidates.queue)
    results = [e for e in candidates.queue]
    results.sort(reverse=True, key=lambda x:x[0])
    return results
    
# Quick Test
def quick_test():
    def display_results(img_array, input_tokens, output_tokens, index):
        print([token_to_word[int(t[index])] for t in input_tokens])
        print([token_to_word[np.argmax(output_tokens[index][i])] for i in range(output_tokens.shape[1])])
    vocab, image_to_tokens = data.build_annotations_vocab('/datadrive/flickr8k/Flickr8k.token.txt')
    with open('/datadrive/flickr8k/Flickr8k.vocab.txt', 'w') as f:
        for w, index in vocab.items():
            f.write('{},{}\n'.format(w, index))
    with open('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', 'w') as f:
        for w, tokens in image_to_tokens.items():
            f.write('{}'.format(w))
            for t in tokens:
                f.write(',{}'.format(t))
            f.write('\n')
    token_to_word = data.load_annotations_vocab('/datadrive/flickr8k/Flickr8k.vocab.txt')
    vocab_size=len(token_to_word)
    stop_symbol=vocab_size - 1
    image_to_tokens = data.load_annotations_tokens('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', stop_symbol)
    max_caption_length=max([len(t) for t in image_to_tokens.values()])
    model = make_model(512, 8, max_caption_length, vocab_size)
    img_array1 = data.load_image('/datadrive/flickr8k/Flicker8k_Dataset/1001773457_577c3a7d70.jpg')
    img_array2 = data.load_image('/datadrive/flickr8k/Flicker8k_Dataset/760180310_3c6bd4fd1f.jpg')
    img_array = np.concatenate([img_array1, img_array2], axis=0)
    token_array1 = data.pad(image_to_tokens['1001773457_577c3a7d70.jpg'], max_caption_length, stop_symbol)
    token_array2 = data.pad(image_to_tokens['760180310_3c6bd4fd1f.jpg'], max_caption_length, stop_symbol)
    token_array = []
    for i in range(len(token_array1)):
        token_array.append(np.asarray([[token_array1[i]],[token_array2[i]]]))
    test_input = [img_array] + token_array
    test_output_list = token_array[1:]
    test_output = np.concatenate(test_output_list, axis = -1)
    model.fit(test_input, test_output, batch_size=2, epochs=500)
    predict_output = model.predict(test_input, batch_size=2)
    display_results(img_array, token_array, predict_output, 0)
    print('--------')
    r = beam_search(model, img_array1, max_caption_length, 0, stop_symbol, vocab_size, beam_width=2)
    for entry in r[0:2]:
        print([token_to_word[t] for t in entry[1]])
    print('--------')
    r = beam_search(model, img_array2, max_caption_length, 0, stop_symbol, vocab_size, beam_width=2)
    for entry in r[0:2]:
        print([token_to_word[t] for t in entry[1]])
    
if __name__ == '__main__':
    quick_test()
