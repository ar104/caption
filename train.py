import tensorflow as tf
import numpy as np
import model
import data


def make_vocab():
    vocab, image_to_tokens = data.build_annotations_vocab('/home/aroy/notebooks/experiments/Flickr8k_text/Flickr8k.token.txt')
    with open('/datadrive/flickr8k/Flickr8k.vocab.txt', 'w') as f:
        for w, index in vocab.items():
            f.write('{},{}\n'.format(w, index))
    with open('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', 'w') as f:
        for w, tokens in image_to_tokens.items():
            f.write('{}'.format(w))
            for t in tokens:
                f.write(',{}'.format(t))
            f.write('\n')

    print('Vocab built.')

def train():
    print('Executing eagerly:{}'.format(tf.executing_eagerly()))
    token_to_word = data.load_annotations_vocab('/datadrive/flickr8k/Flickr8k.vocab.txt')
    vocab_size=len(token_to_word)
    stop_symbol=vocab_size - 1
    image_to_tokens = data.load_annotations_tokens('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', stop_symbol)
    max_caption_length=max([len(t) for t in image_to_tokens.values()])
    caption_model = model.make_model(512, 8, max_caption_length, vocab_size)
    dataset = tf.data.Dataset.list_files('/datadrive/flickr8k/Flicker8k_Dataset/*.jpg')
    def load_image(fname):
        img_path = bytes.decode(fname.numpy())
        img_array = tf.convert_to_tensor(data.load_image(img_path))
        img_name = img_path.split('/')[-1]
        token_list = data.pad(image_to_tokens[img_name], max_caption_length, stop_symbol)
        ret = [img_array[0]]
        for t in token_list:
            ret.append(tf.convert_to_tensor(np.asarray([t])))
        return ret
    
    dataset = dataset.map(lambda x: tf.py_function(load_image, [x], [tf.float32] + [tf.int64]*max_caption_length))
    dataset = dataset.batch(4)
    dataset = dataset.map(lambda *x: (x, tf.concat(x[2:], axis=-1)))
    #for one in dataset:
    #    print(one)
    #    break
    # print(dataset.cardinality().numpy())
    val_dataset = dataset.take(200)
    train_dataset = dataset.skip(200)
    tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
    print(tf.python.client.device_lib.list_local_devices())
    for i in range(10):
        caption_model.fit(train_dataset, epochs=1, callbacks = [tb_callback], validation_data=val_dataset)
        caption_model.save('caption_model.{}.hd5'.format(i))
    
        

if __name__ == '__main__':
    #make_vocab()
    train()
