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

def preprocess(train_fname, test_fname):
    print('Preprocessing data. Executing eagerly:{}'.format(tf.executing_eagerly()))
    token_to_word = data.load_annotations_vocab('/datadrive/flickr8k/Flickr8k.vocab.txt')
    vocab_size=len(token_to_word)
    stop_symbol=vocab_size - 1
    image_to_tokens = data.load_annotations_tokens('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', stop_symbol)
    max_caption_length=max([len(t) for t in image_to_tokens.values()])
    caption_model = model.make_model(512, 8, max_caption_length, vocab_size)
    dataset = tf.data.Dataset.list_files('/datadrive/flickr8k/Flicker8k_Dataset/*.jpg')
    dataset = dataset.shuffle(96)
    def load_image(fname):
        img_path = bytes.decode(fname.numpy())
        img_array = tf.convert_to_tensor(data.load_image(img_path, include_batch=False))
        img_name = img_path.split('/')[-1]
        token_list = data.pad(image_to_tokens[img_name], max_caption_length, stop_symbol)
        ret = [img_array]
        for t in token_list:
            ret.append(tf.convert_to_tensor(np.asarray([t])))
        return ret
    def save_items(ds, fp_image, fp_caption):
        for one in ds:
            np.save(fp_image, one[0], allow_pickle=True)
            np.save(fp_caption, one[1], allow_pickle=True)
    
    dataset = dataset.map(lambda x: tf.py_function(load_image, [x], [tf.float32] + [tf.int64]*max_caption_length))
    dataset = dataset.map(lambda *x: (x[0], tf.concat(x[1:], axis=-1)))
    val_dataset = dataset.take(192)
    train_dataset = dataset.skip(192)
    train_image_fp = open(train_fname + '_image', "wb")
    train_caption_fp = open(train_fname + '_caption', "wb")
    test_image_fp = open(test_fname + '_image', "wb")
    test_caption_fp = open(test_fname + '_caption', "wb")
    save_items(val_dataset, test_image_fp, test_caption_fp)
    save_items(train_dataset, train_image_fp, train_caption_fp)
    # print(dataset.cardinality().numpy())

    
    
def train():
    token_to_word = data.load_annotations_vocab('/datadrive/flickr8k/Flickr8k.vocab.txt')
    vocab_size=len(token_to_word)
    stop_symbol=vocab_size - 1
    image_to_tokens = data.load_annotations_tokens('/datadrive/flickr8k/Flickr8k.image_to_tokens.txt', stop_symbol)
    max_caption_length=max([len(t) for t in image_to_tokens.values()])
    caption_model = model.make_model(512, 8, max_caption_length, vocab_size)
    print(tf.python.client.device_lib.list_local_devices())
    train_image_fp = open('blah_train_image', 'rb')
    train_caption_fp = open('blah_train_caption', 'rb')
    test_image_fp = open('blah_test_image', 'rb')
    test_caption_fp = open('blah_test_caption', 'rb')
    def ds_gen(fp_image, fp_caption):
        def generator():
            while True:
                try:
                    data_image = np.load(fp_image, allow_pickle=True)
                    data_caption = np.load(fp_caption, allow_pickle=True)
                except EOFerror as e:
                    return
                yield (data_image, data_caption)
        return generator
        
    train_dataset = tf.data.Dataset.from_generator(ds_gen(train_image_fp, train_caption_fp),
                                                   output_types=(tf.float32, tf.int64), output_shapes=((224, 224, 3), (max_caption_length,)))
                                                   #output_signature = (tf.TensorSpec(shape=(1, None), dtype=tf.float32), tf.TensorSpec(shape=(max_caption_length,))))
    val_dataset = tf.data.Dataset.from_generator(ds_gen(test_image_fp, test_caption_fp),
                                                   output_types=(tf.float32, tf.int64), output_shapes=((224, 224, 3), (max_caption_length,)))
                                                   #output_signature = (tf.TensorSpec(shape=(1, None), dtype=tf.float32), tf.TensorSpec(shape=(max_caption_length,))))
    train_dataset = train_dataset.map(lambda *x: tuple([tuple([x[0]] + tf.split(x[1], max_caption_length, axis=-1)), x[1][1:]]))
    #for v in train_dataset:
    #    print(v)
    #    exit(-1)
    val_dataset = val_dataset.map(lambda *x: tuple([tuple([x[0]] + tf.split(x[1], max_caption_length, axis=-1)), x[1][1:]]))
    train_dataset = train_dataset.batch(32)
    val_dataset = val_dataset.batch(32)
    train_dataset = train_dataset.prefetch(2)
    val_dataset = val_dataset.prefetch(2)

    caption_model.fit(train_dataset, epochs=2,
                      callbacks = [tf.keras.callbacks.TensorBoard('./logs', update_freq=1),
                                   tf.keras.callbacks.ModelCheckpoint(filepath='caption_model.{epoch:02d}-{val_loss:.2f}.h5')],
                      validation_data=val_dataset)

if __name__ == '__main__':
    #preprocess('blah_train', 'blah_test') 
    #make_vocab()
    train()
