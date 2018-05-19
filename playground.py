import hashlib

import numpy
from keras.utils import np_utils
from music21 import *
from music21 import converter, instrument, note, chord

from music21.ext.six import StringIO
from music21.alpha.analysis import aligner
from music21.alpha.analysis import fixer
from music21.alpha.analysis import hasher
from music21.alpha.analysis import search
from music21.analysis import correlate
from music21.analysis import discrete
from music21.analysis import elements
from music21.analysis import enharmonics
from music21.analysis import floatingKey
from music21.analysis import metrical
from music21.analysis import neoRiemannian
from music21.analysis import patel
from music21.analysis import pitchAnalysis
from music21.analysis import reduceChords
from music21.analysis import reduceChordsOld
from music21.analysis import reduction
from music21.analysis import transposition
from music21.analysis import windowed
from music21 import articulations
from music21 import audioSearch
from music21.audioSearch import recording
from music21.audioSearch import scoreFollower
from music21.audioSearch import transcriber
from music21 import bar
from music21 import base
from music21 import beam
from music21.braille import basic
from music21.braille import examples
from music21.braille import noteGrouping
from music21.braille import objects
from music21.braille import runAllBrailleTests
from music21.braille import segment
from music21.braille import text
from music21.braille import translate
from music21 import chord
from music21.chord import tables
from music21 import clef
from music21.common import classTools
from music21.common import decorators
from music21.common import fileTools
from music21.common import formats
from music21.common import misc
from music21.common import numberTools
from music21.common import objects
from music21.common import parallel
from music21.common import pathTools
from music21.common import stringTools
from music21.common import weakrefTools
from music21 import converter
from music21.converter import qmConverter
from music21.converter import subConverters
from music21 import corpus
from music21.corpus import corpora
from music21.corpus import manager
from music21.corpus import virtual
from music21.corpus import work
from music21 import derivation
from music21 import duration
from music21 import dynamics
from music21 import editorial
from music21 import environment
from music21 import expressions
from music21.features import base
from music21.features import jSymbolic
from music21.features import native
from music21.features import outputFormats
from music21.figuredBass import checker
from music21.figuredBass import examples
from music21.figuredBass import notation
from music21.figuredBass import possibility
from music21.figuredBass import realizer
from music21.figuredBass import realizerScale
from music21.figuredBass import resolution
from music21.figuredBass import rules
from music21.figuredBass import segment
from music21 import freezeThaw
from music21 import graph
from music21.graph import axis
from music21.graph import findPlot
from music21.graph import plot
from music21.graph import primitives
from music21.graph import utilities
from music21 import harmony
from music21 import humdrum
from music21.humdrum import instruments
from music21.humdrum import spineParser
from music21 import instrument
from music21 import interval
from music21 import ipython21
from music21 import key
from music21.languageExcerpts import naturalLanguageObjects
from music21 import layout
from music21.lily import lilyObjects
from music21.lily import translate
from music21.mei import base
from music21 import metadata
from music21.metadata import bundles
from music21.metadata import caching
from music21.metadata import primitives
from music21 import meter
from music21 import midi
from music21.midi import percussion
from music21.midi import realtime
from music21.midi import translate
from music21 import musedata
from music21.musedata import base40
from music21.musedata import translate
from music21 import note
from music21.noteworthy import binaryTranslate
from music21.noteworthy import translate
from music21.omr import correctors
from music21.omr import evaluators
from music21 import pitch
from music21 import repeat
from music21 import roman
from music21 import scale
from music21.scale import intervalNetwork
from music21.scale import scala
from music21.search import base
from music21.search import lyrics
from music21.search import segment
from music21.search import serial
from music21 import serial
from music21 import sieve
from music21 import sites
from music21 import sorting
from music21 import spanner
from music21 import stream
from music21.stream import core
from music21.stream import filters
from music21.stream import iterator
from music21.stream import makeNotation
from music21.stream import streamStatus
from music21 import style
from music21 import tablature
from music21 import tempo
from music21 import text
from music21 import tie
from music21 import tinyNotation
from music21 import tree
from music21.tree import analysis
from music21.tree import core
from music21.tree import fromStream
from music21.tree import node
from music21.tree import spans
from music21.tree import timespanTree
from music21.tree import toStream
from music21.tree import trees
from music21.tree import verticality
from music21 import variant
from music21.vexflow import toMusic21j
from music21 import voiceLeading
from music21 import volpiano
from music21 import volume
import os
from pathlib import Path
import matplotlib as mpl
import pickle
mpl.use('TkAgg')



KEYBOARD_INSTRUMENTS = ["KeyboardInstrument", "Piano", "Harpsichord", "Clavichord", "Celesta", ]
STRING_INSTRUMENTS = ["StringInstrument", "Violin", "Viola", "Violoncello", "Contrabass", "Harp", "Guitar",
                          "AcousticGuitar", "Acoustic Guitar", "ElectricGuitar", "Electric Guitar", "AcousticBass",
                          "Acoustic Bass", "ElectricBass", "Electric Bass", "FretlessBass", "Fretless Bass", "Mandolin",
                          "Ukulele", "Banjo", "Lute", "Sitar", "Shamisen", "Koto", ]
SAVED_KEYB_NOTES = 'data/keyboard_notes'
SAVED_STR_NOTES = 'data/string_notes'

SONG_DIR_PATH = "/Users/konradsbuss/Documents/Uni/bak/dataset/preparedData/smalltest.txt"


def main():

    keyboard_notes, string_notes = get_all_notes()

    # get amount of pitch names
    n_vocab_k = len(set(keyboard_notes))
    n_vocab_s = len(set(string_notes))

    x = prepare_sequences(keyboard_notes, n_vocab_k)
    y = prepare_sequences(string_notes, n_vocab_s)

    print(n_vocab_k)
    print(n_vocab_s)


def get_all_notes():
    cmp_keyboard_notes = []
    cmp_string_notes = []

    with open(str(SONG_DIR_PATH)) as f:
        # iterate trough songs and save notes
        for line in f:
            midi_file_path = line.rstrip()
            midi_file_path = line.rstrip()
            midi_file_path = midi_file_path.replace("/home/konrads/Documents/bakalaurs/",
                                                    "/Users/konradsbuss/Documents/Uni/bak/dataset/")
            midi_file_path = midi_file_path.replace("\\", "/")
            midi_file_path = midi_file_path.replace("E:", "/Users/konradsbuss/Documents/Uni/bak/dataset")

            cmp_keyboard_notes = get_notes_chords_rests(KEYBOARD_INSTRUMENTS, midi_file_path, cmp_keyboard_notes)
            cmp_string_notes = get_notes_chords_rests(STRING_INSTRUMENTS, midi_file_path, cmp_string_notes)

            cmp_keyboard_notes, cmp_string_notes = note_sanity_check(cmp_keyboard_notes, cmp_string_notes)

    with open(str(SAVED_KEYB_NOTES), 'wb') as file_path:
        pickle.dump(cmp_keyboard_notes, file_path)
    with open(str(SAVED_STR_NOTES), 'wb') as file_path:
        pickle.dump(cmp_string_notes, file_path)

    return cmp_keyboard_notes, cmp_string_notes


def note_sanity_check(keyboard_notes, string_notes):
    """Check note length and normalize it"""
    if len(keyboard_notes) != len(string_notes):
        notes_max = max(len(keyboard_notes), len(string_notes))
        notes_min = min(len(keyboard_notes), len(string_notes))

        if notes_max == len(keyboard_notes):
            del keyboard_notes[notes_min:]
        else:
            del string_notes[notes_min:]

    return keyboard_notes, string_notes


def get_notes_chords_rests(instrument_type, path, note_list):
    """ Get all the notes, chords and rests from the midi files in the song directory """
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


def create_midi(prediction_output, string):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        elif pattern == 'Rest':
            new_note = note.Rest()
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    if string == "string":
        midi_stream.write('midi', fp='/Users/konradsbuss/Desktop/tmp/test_output_string.mid')
    else:
        midi_stream.write('midi', fp='/Users/konradsbuss/Desktop/tmp/test_output_piano.mid')


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

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

    return (network_input, network_output)


main()
