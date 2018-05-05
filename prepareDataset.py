import hashlib

import pretty_midi
import numpy as np
import re
import os
import glob
import mmap
from pathlib import Path
from music21 import *
import music21
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

import mido

dir_path = "/Users/konradsbuss/Documents/Uni/bak/dataset/references/MuseData"
output_file_path = "/Users/konradsbuss/Documents/Uni/bak/dataset/preparedData/filename_MuseData.txt"
output_file_path_same_note_length = "/Users/konradsbuss/Documents/Uni/bak/dataset/preparedData/filename_MuseData.txt"


def prepare_data():
    keyboard_instruments = ["KeyboardInstrument", "Piano", "Harpsichord", "Clavichord", "Celesta",]
    string_instruments = ["StringInstrument", "Violin", "Viola", "Violoncello", "Contrabass", "Harp", "Guitar",
                          "AcousticGuitar", "Acoustic Guitar", "ElectricGuitar", "Electric Guitar", "AcousticBass",
                          "Acoustic Bass", "ElectricBass", "Electric Bass", "FretlessBass", "Fretless Bass", "Mandolin",
                          "Ukulele", "Banjo", "Lute", "Sitar", "Shamisen", "Koto",]

    for midi_file_path in Path(dir_path).glob('**/*.mid'):
        # Some MIDI files will raise Exceptions on loading, if they are invalid.
        # We just skip those.
        try:
            song = converter.parse(str(midi_file_path))
            # Extract MIDI file with two instruments where one is piano
            parts = instrument.partitionByInstrument(song)
            if parts:
                if len(parts) == 2:
                    if (parts.parts[0].id in keyboard_instruments and parts.parts[1].id in string_instruments) or \
                            (parts.parts[1].id in keyboard_instruments and parts.parts[0].id in string_instruments):
                                    print(midi_file_path)
                                    text_file = open(str(output_file_path), "a+")
                                    text_file.write(str(midi_file_path) + "\n")
                                    text_file.close()

                                    keyboard_notes = get_notes_chords_rests(keyboard_instruments, midi_file_path)
                                    string_notes = get_notes_chords_rests(string_instruments, midi_file_path)

                                    if len(keyboard_notes) == len(string_notes):
                                        text_file = open(str(output_file_path_same_note_length), "a+")
                                        text_file.write(str(midi_file_path) + "\n")
                                        text_file.close()
        except Exception as e:
            print("Exception ", e)
            pass


def get_notes_chords_rests(instrument_type, path):
    """ Get all the notes, chords and rests from the midi files in the ./midi_songs directory """
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
    """Check if there is not missing rests which ensures silent part in melody"""
    if element.offset / 0.5 <= len(note_list):
        return
    else:
        while True:
            note_list.append('Rest')
            if element.offset / 0.5 <= len(note_list):
                return

prepare_data()