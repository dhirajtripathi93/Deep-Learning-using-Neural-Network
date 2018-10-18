import numpy as np
from keras.preprocessing.sequence import pad_sequences
from src import config
from src.config import input_length
from src.neural_network_trainer import load_model, \
    load_vocab_tokenizer, load_encoded_sentence_from_string, all_languages

vocab_tokenizer = load_vocab_tokenizer(config.vocab_tokenizer_location)
model = load_model(config.model_file_location, config.weights_file_location)


def to_language(binary_list):
    i = np.argmax(binary_list)
    return all_languages[i]


def get_neural_network_input(code):
    encoded_sentence = load_encoded_sentence_from_string(code, vocab_tokenizer)
    return pad_sequences([encoded_sentence], maxlen=input_length)


def detect(code):
    y_proba = model.predict(get_neural_network_input(code))
    return to_language(y_proba)




code = """
   
int badblocks_set(struct badblocks *bb, sector_t s, int sectors,
			int acknowledged)
{
	u64 *p;
	int lo, hi;
	int rv = 0;
	unsigned long flags;

	if (bb->shift < 0)
		/* badblocks are disabled */
		return 1;

	if (bb->shift) {
		/* round the start down, and the end up */
		sector_t next = s + sectors;

		s >>= bb->shift;
		next += (1<<bb->shift) - 1;
		next >>= bb->shift;
		sectors = next - s;
	}

	write_seqlock_irqsave(&bb->lock, flags);

	p = bb->page;
	lo = 0;
	hi = bb->count;
	/* Find the last range that starts at-or-before 's' */
	while (hi - lo > 1) {
		int mid = (lo + hi) / 2;
		sector_t a = BB_OFFSET(p[mid]);

		if (a <= s)
			lo = mid;
		else
			hi = mid;
	}


"""
print(detect(code))

