
"""
Due to the current lack of need for syllables and phonetic speech, for now whole words will be the "cortical neurons"


"""

# Convert each word in inputted sentence into its own cortical neuron


def process(text):
    sentences = text.split(".")
    sentenceList = []
    wordList = [[]]

    # for each word in the line:
    for sentence in sentences:
        # print the word
        sentenceList.append(sentence)
    for words in sentenceList:
        wordList.append(words.split())

    print(wordList)