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
import matplotlib as mpl
mpl.use('TkAgg')


def main():
    keyboard_instruments = ["KeyboardInstrument", "Piano", "Harpsichord", "Clavichord", "Celesta", ]
    string_instruments = ["StringInstrument", "Violin", "Viola", "Violoncello", "Contrabass", "Harp", "Guitar",
                          "AcousticGuitar", "Acoustic Guitar", "ElectricGuitar", "Electric Guitar", "AcousticBass",
                          "Acoustic Bass", "ElectricBass", "Electric Bass", "FretlessBass", "Fretless Bass", "Mandolin",
                          "Ukulele", "Banjo", "Lute", "Sitar", "Shamisen", "Koto", ]

    # midi = converter.parse("/Users/konradsbuss/Documents/Uni/bak/dataset/lmd_full/7/7c5e7f395696f30f97b9091326882af6.mid")
    midi = converter.parse("/Users/konradsbuss/Documents/Uni/bak/dataset/lmd_full/9/9faf098947015f0ab65d4f87b9b6d41d.mid")



    piano_notes, string_notes = get_notes()
    print("piano_notes")
    print(piano_notes)
    print(len(piano_notes))
    print("string_notes")
    print(string_notes)
    print(len(string_notes))

    parts = instrument.partitionByInstrument(midi)

    notes0 = []
    notes1 = []
    counter = 0
    counter2 = 0
    print("loh")
    for thing in parts.parts[0]:
        # counter = counter + 1
        # if counter > 4:
        #     print(thing, thing.quarterLength, thing.offset)
            notes0.append(thing)

    print("loh")
    for thing in parts.parts[1]:
        # counter2 = counter2 + 1
        # if counter2 > 4:
        #     print(thing, thing.quarterLength, thing.offset)
            notes1.append(thing)


    print("len(notes0)")
    print(len(notes0))
    print("len(notes1)")
    print(len(notes1))

    for x in range(100):
        print("0 " + str(notes0[x]), str(notes0[x].offset))
        print("1 " + str(notes1[x]), str(notes1[x].offset))

    oiter = stream.iterator.OffsetIterator(parts[0])
    for groupedElements in oiter:
        if len(groupedElements) >= 2:
            print(groupedElements, groupedElements[0].offset, groupedElements[1].offset)
        else:
            print(groupedElements, groupedElements[0].offset)




def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    keyboard_instruments = ["KeyboardInstrument", "Piano", "Harpsichord", "Clavichord", "Celesta", ]
    string_instruments = ["StringInstrument", "Violin", "Viola", "Violoncello", "Contrabass", "Harp", "Guitar",
                          "AcousticGuitar", "Acoustic Guitar", "ElectricGuitar", "Electric Guitar", "AcousticBass",
                          "Acoustic Bass", "ElectricBass", "Electric Bass", "FretlessBass", "Fretless Bass", "Mandolin",
                          "Ukulele", "Banjo", "Lute", "Sitar", "Shamisen", "Koto", ]
    piano_notes = []
    string_notes = []

    midi = converter.parse(
        "/Users/konradsbuss/Documents/Uni/bak/dataset/lmd_full/7/7c5e7f395696f30f97b9091326882af6.mid")

    string_notes_to_parse = None
    piano_notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)

    for music_instrument in range(len(parts)):
        if parts.parts[music_instrument].id in keyboard_instruments:
            piano_notes_to_parse = parts.parts[music_instrument]
        if parts.parts[music_instrument].id in string_instruments:
            string_notes_to_parse = parts.parts[music_instrument]

    for element in piano_notes_to_parse:
        if isinstance(element, note.Note):
            piano_notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            piano_notes.append('.'.join(str(n) for n in element.normalOrder))

    for element in string_notes_to_parse:
        if isinstance(element, note.Note):
            string_notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            string_notes.append('.'.join(str(n) for n in element.normalOrder))

    return piano_notes, string_notes


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    print("note_to_int")
    print(note_to_int)


    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    print("network_output")
    print(len(network_output))
    n_patterns = len(network_input)
    print("n_patterns")
    print(n_patterns)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    print("network_input")
    print(network_input.shape)



    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (1, 2)
    # return (network_input, network_output)

main()