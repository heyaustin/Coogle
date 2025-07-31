from pydub import AudioSegment

m4a_file = '6.m4a'
wav_filename = '6.wav'

sound = AudioSegment.from_file(m4a_file, format='m4a')
file_handle = sound.export(wav_filename, format='wav')
