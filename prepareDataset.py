from music21 import chord, converter, instrument, note, stream, common
from pathlib import Path
import csv


keyboard_instruments = ["KeyboardInstrument", "Piano", "Harpsichord", "Clavichord", "Celesta",]
string_instruments = ["StringInstrument", "Violin", "Viola", "Violoncello", "Contrabass", "Harp", "Guitar",
                          "AcousticGuitar", "Acoustic Guitar", "ElectricGuitar", "Electric Guitar", "AcousticBass",
                          "Acoustic Bass", "ElectricBass", "Electric Bass", "FretlessBass", "Fretless Bass", "Mandolin",
                          "Ukulele", "Banjo", "Lute", "Sitar", "Shamisen", "Koto",]

dir_path = "/Users/konradsbuss/Documents/Uni/bak/dataset/lmd_full/c"
output_file_path = "/Users/konradsbuss/Documents/Uni/bak/dataset/preparedData/filename_full.txt"
output_file_path_same_note_length01 = "/Users/konradsbuss/Documents/Uni/bak/dataset/preparedData/filename_same_note_len_01.txt"
output_file_path_same_note_exception = "/Users/konradsbuss/Documents/Uni/bak/dataset/preparedData/filename_exception.txt"
output_file_path_csv = "/Users/konradsbuss/Documents/Uni/bak/dataset/preparedData/filename_csv.csv"


def prepare_data():
    path_list = []
    for midi_file_path in Path(dir_path).glob('**/*.mid'):
        path_list.append(midi_file_path)
    common.runParallel(path_list, check_instruments_and_save_notes)


def check_instruments_and_save_notes(midi_file_path):
    try:
        # Some MIDI files will raise Exceptions on loading, if they are invalid.
        # We just skip those.
        song = converter.parse(midi_file_path)
        # Extract MIDI file with two instruments where one is piano
        parts = instrument.partitionByInstrument(song)
        if parts:
            if len(parts) == 2:
                if (parts.parts[0].id in keyboard_instruments and parts.parts[1].id in string_instruments) or \
                        (parts.parts[1].id in keyboard_instruments and parts.parts[0].id in string_instruments):
                    print(midi_file_path)
                    write_to_file(output_file_path, str(midi_file_path))

                    keyboard_notes = get_notes_chords_rests(keyboard_instruments, midi_file_path)
                    string_notes = get_notes_chords_rests(string_instruments, midi_file_path)
                    print(len(keyboard_notes), len(string_notes))
                    notes_max = max(len(keyboard_notes), len(string_notes))
                    notes_min = min(len(keyboard_notes), len(string_notes))
                    percentage = (notes_max - notes_min) / notes_max

                    write_to_csv(output_file_path_csv,
                                 percentage,
                                 len(keyboard_notes),
                                 len(string_notes),
                                 str(midi_file_path))

                    if notes_max - (notes_max * 0.1) <= notes_min:
                        write_to_file(output_file_path_same_note_length01, str(midi_file_path))

    except Exception as e:
        print("Exception ", midi_file_path, e)
        write_to_file(output_file_path_same_note_exception, str(midi_file_path))
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


def write_to_file(file_path, what_to_write):
    text_file = open(str(file_path), "a+")
    text_file.write(str(what_to_write) + "\n")
    text_file.close()


def write_to_csv(csv_file_path, percentage, note_p, note_s, midi_file_path):
    with open(str(csv_file_path), 'a+', newline='') as csvfile:
        fieldnames = ['Percentage', 'Note_P', 'Note_S', 'Path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'Percentage': str(percentage),
                         'Note_P': str(note_p),
                         'Note_S': str(note_s),
                         'Path': str(midi_file_path)})


def init_csv_file(csv_file_path):
    with open(str(csv_file_path), 'a+', newline='') as csvfile:
        fieldnames = ['Percentage', 'Note_P', 'Note_S', 'Path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


if __name__ == '__main__':
    init_csv_file(output_file_path_csv)
    prepare_data()
