from nltk.translate.bleu_score import corpus_bleu
from .predict import predict_sequence

# evaluate the skill of the model
def evaluate_accuracy(model, tokenizer, sources, raw_dataset):
    actual, predicted = [], []
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]'
                % (raw_src, raw_target, translation)
            )
        actual.append([raw_target.split()])
        predicted.append(translation.split())

    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted,
        weights=(1.0, 0, 0, 0))
    )
    print('BLEU-2: %f' % corpus_bleu(actual, predicted,
        weights=(0.5, 0.5, 0, 0))
    )
    print('BLEU-3: %f' % corpus_bleu(actual, predicted,
        weights=(0.3, 0.3, 0.3, 0))
    )
    print('BLEU-4: %f' % corpus_bleu(actual, predicted,
        weights=(0.25, 0.25, 0.25, 0.25))
    )
