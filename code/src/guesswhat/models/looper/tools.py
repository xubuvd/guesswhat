import numpy as np

def get_index(l, index, default=-1):
    try:
        return l.index(index)
    except ValueError:
        return default
"""
2019-04-09 23:00:24,276 :: INFO :: before-batch_id:0    url:310600.jpg  <start> is it a person ? <yes> is it the man ? <yes> <stop_dialogue> <no> is it the whole man ? <yes> <stop_dialogue> <no>
2019-04-09 23:00:24,276 :: INFO :: before-batch_id:1    url:496431.jpg  <start> is it the blender ? <no> is it the fridge ? <no> is it the fridge ? <no> is it the fridge ? <no> is it the fridge? <no>
2019-04-09 23:00:24,276 :: INFO :: before-batch_id:2    url:319706.jpg  <start> is it a cat ? <no> is it a keyboard ? <no> is it a keyboard ? <no> is it a keyboard ? <no> is it a keyboard ? <no>
2019-04-09 23:00:24,276 :: INFO :: before-batch_id:3    url:420308.jpg  <start> is it a person ? <yes> is it the person on the left ? <no> <stop_dialogue> <no> is it the person on the right ? <yes> <stop_dialogue> <no>
2019-04-09 23:00:24,276 :: INFO :: before-batch_id:4    url:379869.jpg  <start> is it a cat ? <yes> is it the one sitting on the left ? <yes> <stop_dialogue> <no> is it the one with the cat ? <yes> <stop_dialogue> <no>
2019-04-09 2:00:24,276 :: INFO :: before-batch_id:5    url:318185.jpg  <start> is it a computer ? <no> is it a keyboard ? <yes> is it the one on the left ? <no> <stop_dialogue> <no> is it the one on the right ? <no>
2019-04-09 23:00:24,276 :: INFO :: before-batch_id:6    url:507067.jpg  <start> is it a carrot ? <no> is it a knife ? <no> is it a knife ? <no> is it a table ? <no> is it a table ? <no>
2019-04-09 23:00:24,276 :: INFO :: before-batch_id:7    url:243650.jpg  <start> is it a person ? <yes> is it the one on the bike ? <no> is it the one on the left ? <yes> <stop_dialogue> <no> is it the one on the left ? <yes>

2019-04-09 23:00:24,294 :: INFO :: after-batch_id:0 url:310600.jpg  <start> is it a person ? <yes> is it the man ? <yes> <stop_dialogue>    Guesser: True
2019-04-09 23:00:24,294 :: INFO :: after-batch_id:1 url:496431.jpg  <start> is it the blender ? <no> is it the fridge ? <no> is it the fridge ? <no> is it the fridge ? <no> is it the fridge ? <no> Guesser: False
2019-04-09 23:00:24,294 :: INFO :: after-batch_id:2 url:319706.jpg  <start> is it a cat ? <no> is it a keyboard ? <no> is it a keyboard ? <no> is it a keyboard ? <no> is it a keyboard ? <no>  Guesser: True
2019-04-09 23:00:24,294 :: INFO :: after-batch_id:3 url:420308.jpg  <start> is it a person ? <yes> is it the person on the left ? <no> <stop_dialogue>  Guesser: False
2019-04-09 23:00:24,294 :: INFO :: after-batch_id:4 url:379869.jpg  <start> is it a cat ? <yes> is it the one sitting on the left ? <yes> <stop_dialogue>   Guesser: True
2019-04-09 23:00:24,294 :: INFO :: after-batch_id:5 url:318185.jpg  <start> is it a computer ? <no> is it a keyboard ? <yes> is it the one on the left ? <no> <stop_dialogue>   Guesser: True
2019-04-09 23:00:24,294 :: INFO :: after-batch_id:6 url:507067.jpg  <start> is it a carrot ? <no> is it a knife ? <no> is it a knife ? <no> is it a table ? <no> is it a table ? <no>   Guesser: True
2019-04-09 23:00:24,294 :: INFO :: after-batch_id:7 url:243650.jpg  <start> is it a person ? <yes> is it the one on the bike ? <no> is it the one on the left ? <yes> <stop_dialogue>   Guesser: False
"""
def clear_after_stop_dialogue(dialogues, tokenizer):
    stop_indices = []
    final_dialogues = []
    
    batch_size = len(dialogues)
    for i, dialogue in enumerate(dialogues):
        stop_dialogue_index = get_index(dialogue.tolist(), tokenizer.stop_dialogue, default=len(dialogue)-1)
        answers_index = [j for j,token in enumerate(dialogue[:stop_dialogue_index+1]) if token in tokenizer.answers]
        if answers_index:
            #dialog = list(dialogue[:stop_dialogue_index+1])
            #if dialog[-1] != tokenizer.stop_dialogue:
            #    dialog.append(tokenizer.stop_dialogue)
            #    stop_dialogue_index += 1
            final_dialogues.append(dialogue[:stop_dialogue_index+1])
            stop_indices.append(stop_dialogue_index)
        else:
            #<start> is it a person is it the <stop_dialogue> <no> is it a person ? <no> is it a <unk> ? <no> is it a car ? <yes> <stop_dialogue> <no>
            #<start> is it a person the is the object a <unk> <stop_dialogue> <no> is it a the it ? <no> is it a ? <no> is it a person ? <no> is it a person ? <no>
            #game = tokenizer.decode(dialogue)
            #print("clear_after_stop_dialogue_else-b[{}], dialogue:{}, game:{}".format(i,dialogue,game))
            final_dialogues.append([])
            stop_indices.append(0)
    #while len(final_dialogues) < batch_size:
    """
    permute_pairs = list()
    for i,dialogue in enumerate(final_dialogues):
        if len(dialogue) > 1:continue
        sample_count = 0
        idx = np.random.randint(low=0, high=len(final_dialogues), size=1)[0]
        while len(final_dialogues[idx]) < 1:
            idx = np.random.randint(low=0, high=len(final_dialogues), size=1)[0]
            sample_count += 1
            if sample_count > int(batch_size/2):break
        final_dialogues[i] = final_dialogues[idx]
        stop_indices[i] = stop_indices[idx]
        permute_pairs.append((i,idx))
    """
    return final_dialogues, stop_indices

def list_to_padded_tokens(dialogues, tokenizer):
    # compute the length of the dialogue
    seq_length = [len(d) for d in dialogues]

    # Get dialogue numpy max size
    batch_size = len(dialogues)
    max_seq_length = max(seq_length)

    # Initialize numpy array
    padded_tokens = np.full((batch_size, max_seq_length), tokenizer.padding_token, dtype=np.int32)

    # fill the padded array with word_id
    for i, (one_path, l) in enumerate(zip(dialogues, seq_length)):
       padded_tokens[i, 0:l] = one_path
    return padded_tokens, seq_length

