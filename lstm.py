""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


KEYBOARD_INSTRUMENTS = ["KeyboardInstrument", "Piano", "Harpsichord", "Clavichord", "Celesta", ]
STRING_INSTRUMENTS = ["StringInstrument", "Violin", "Viola", "Violoncello", "Contrabass", "Harp", "Guitar",
                          "AcousticGuitar", "Acoustic Guitar", "ElectricGuitar", "Electric Guitar", "AcousticBass",
                          "Acoustic Bass", "ElectricBass", "Electric Bass", "FretlessBass", "Fretless Bass", "Mandolin",
                          "Ukulele", "Banjo", "Lute", "Sitar", "Shamisen", "Koto", ]
SONG_DIR_PATH = "/Users/konradsbuss/Documents/Uni/bak/dataset/preparedData/filenameLINUX2.txt"
SAVED_KEYB_NOTES = 'data/keyboard_notes'
SAVED_STR_NOTES = 'data/string_notes'


def train_network():
    """ Train a Neural Network to generate music """

    keyboard_notes, string_notes = get_all_notes()

    # get amount of pitch names
    n_vocab_k = len(set(keyboard_notes))
    n_vocab_s = len(set(string_notes))

    network_input = prepare_sequences(keyboard_notes, n_vocab_k)
    network_output = prepare_sequences(string_notes, n_vocab_s)

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_all_notes():
    complete_keyboard_notes = []
    complete_string_notes = []

    with open(str(SONG_DIR_PATH)) as f:
        # iterate trough songs and save notes
        for line in f:
            midi_file_path = line.rstrip()
            midi_file_path = midi_file_path.replace("/home/konrads/Documents/bakalaurs/", "/Users/konradsbuss/Documents/Uni/bak/dataset/")

            keyboard_notes = get_notes_chords_rests(KEYBOARD_INSTRUMENTS, midi_file_path)
            string_notes = get_notes_chords_rests(STRING_INSTRUMENTS, midi_file_path)

            keyboard_notes, string_notes = note_sanity_check(keyboard_notes, string_notes)

            complete_keyboard_notes.append(keyboard_notes)
            complete_string_notes.append(string_notes)

    return complete_keyboard_notes, complete_string_notes


def note_sanity_check(keyboard_notes, string_notes):
    """Check note length and normalize it"""
    if len(keyboard_notes) != len(string_notes):
        notes_max = max(len(keyboard_notes), len(string_notes))
        notes_min = min(len(keyboard_notes), len(string_notes))
        del notes_max[notes_min:]

    with open(str(SAVED_KEYB_NOTES), 'wb') as file_path:
        pickle.dump(keyboard_notes, file_path)
    with open(str(SAVED_STR_NOTES), 'wb') as file_path:
        pickle.dump(string_notes, file_path)

    return keyboard_notes, string_notes


def get_notes_chords_rests(instrument_type, path):
    """ Get all the notes, chords and rests from the midi files in the song directory """
    note_list = []
    try:
        midi_file = converter.parse(path)
        parts = instrument.partitionByInstrument(midi_file)
        for music_instrument in range(len(parts)):
            if parts.parts[music_instrument].id in instrument_type:
                for element_by_offset in stream.iterator.OffsetIterator(parts[music_instrument]):
                    for entry in element_by_offset:
                        if isinstance(entry, note.Note):
                            check_rest_amount(entry, note_list)
                            note_list.append(str(entry.pitch))
                        elif isinstance(entry, chord.Chord):
                            check_rest_amount(entry, note_list)
                            note_list.append('.'.join(str(n) for n in entry.normalOrder))
                        elif isinstance(entry, note.Rest):
                            check_rest_amount(entry, note_list)
                            note_list.append('Rest')
    except Exception as e:
        print("failed on ", path, e)
        pass
    return note_list


def check_rest_amount(element, note_list):
    """Check if there is not a missing rests which ensures silent part in melody"""
    if element.offset / 0.5 <= len(note_list):
        return
    else:
        while True:
            note_list.append('Rest')
            if element.offset / 0.5 <= len(note_list):
                return


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitch_names = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
