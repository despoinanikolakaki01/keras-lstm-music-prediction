""" This module generates notes for a midi file using the
	trained neural network """
import pickle
import glob
import numpy
from music21 import instrument, note, stream, chord, converter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Bidirectional, LSTM, concatenate, Input
from tensorflow.keras.layers import BatchNormalization as BatchNorm
import tensorflow.keras.utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

def generate():
	""" Generate a piano midi file """
	#load the notes used to train the model
	with open('notes', 'rb') as filepath:
		notes = pickle.load(filepath)
	
	with open('durations', 'rb') as filepath:
		durations = pickle.load(filepath)
	
	with open('offsets', 'rb') as filepath:
		offsets = pickle.load(filepath)

	


	



	# Get all pitch names
	#pitchnames = sorted(set(item for item in notes))
	# Get all pitch names
	#n_vocab = len(set(notes))
	
	
	notenames = sorted(set(item for item in notes))
	n_vocab_notes = len(set(notes))
	network_input_notes, normalized_input_notes = prepare_sequences(notes, notenames, n_vocab_notes)
	
	offsetnames = sorted(set(item for item in offsets))
	n_vocab_offsets = len(set(offsets))
	network_input_offsets, normalized_input_offsets = prepare_sequences(offsets, offsetnames, n_vocab_offsets)
	
	durationames = sorted(set(item for item in durations))
	n_vocab_durations = len(set(durations))
	network_input_durations, normalized_input_durations = prepare_sequences(durations, durationames, n_vocab_durations)


    #PRIOR DATA PREPARING
	start, start2, start3 = get_notes()
	# print('This is start2')
	# print(start2)
	#dhmiourgw to prior gia na to prosthesw sto teliko apotelesma
	prior = []
	for i in range(101):
		row = [start[i], start2[i], start3[i]]
		print(row)
		prior.append(row)
	#print('THis is prior:')
	# print(prior)
	 	
    
	# print('This is start2:')
	# print(start2)
	
	
    #kanw mapping tis notes me integers gia na mporw na tis xrhsimopoihsw sthn generate_notes
	notestart = sorted(set(item for item in start))
	n_vocab_notestart = len(set(start))
	start_input_notes, normalizedstart_input_notes = prepare_sequences(start, notestart, n_vocab_notestart)
	# print("This is start input")
	# print(start_input_notes)

	durstart = sorted(set(item for item in start3))
	n_vocab_durstart = len(set(start3))
	start_input_durations, normalizedstart_input_durations = prepare_sequences(start3, durstart, n_vocab_durstart)

	offsetstart = sorted(set(item for item in start2))
	n_vocab_offsetstart = len(set(start2))
	start_input_offsets, normalizedstart_input_offsets = prepare_sequences(start2, offsetstart, n_vocab_offsetstart)
	# print("This is the input:")
	# print(start_input_offsets)

	#model = create_network(network_input_notes, n_vocab_notes, network_input_offsets, n_vocab_offsets, network_input_durations, n_vocab_durations)
	
	model = create_network(normalized_input_notes, n_vocab_notes, normalized_input_offsets, n_vocab_offsets, normalized_input_durations, n_vocab_durations)
	
	
	
	
	
	

	#network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
	#model = create_network(normalized_input, n_vocab)
	
	prediction_output = generate_notes(start_input_notes,start_input_offsets,start_input_durations,model, network_input_notes, network_input_offsets, network_input_durations, notenames, offsetnames, durationames, n_vocab_notes, n_vocab_offsets, n_vocab_durations)
	#print("This is prediction output:")
	#print(prediction_output) #to prediction einai 3 sthles me [notes,offsets,durations]
	final_output = prior + prediction_output
	print("Prior")
	print(len(prior))
	# print("Prediction_output")
	# print(len(prediction_output))
	# print("Final")
	# print(len(final_output))
	# print(final_output)
	create_midi(final_output)

def get_notes():
	""" Get all the notes and chords from the midi files in the ./midi_songs directory """
	notes = []
	offsets = []
	durations = []
	for file in glob.glob("midis/classical-piano-type0/ty_oktober.mid"):
		midi = converter.parse(file)

		#print("Parsing %s" % file)

		notes_to_parse = None

		# try: # file has instrument parts
		# 	s2 = instrument.partitionByInstrument(midi)
		# 	notes_to_parse = s2.parts[0].recurse() 
		# except: # file has notes in a flat structure
		# 	notes_to_parse = midi.flat.notes

		notes_to_parse = midi.flat.notes
		

		
		
		offsetBase = 0
		for element in notes_to_parse:
			isNoteOrChord = False
			
			if isinstance(element, note.Note): #elegxei an einai nota
				notes.append(str(element.pitch))
				isNoteOrChord = True
			elif isinstance(element, chord.Chord): #elegxei an einai sygxordia
				notes.append('.'.join(str(n) for n in element.normalOrder))
				isNoteOrChord = True
			
			if isNoteOrChord:
				offsets.append(str(element.offset - offsetBase)) #aferei to offset ths prohgoumenhs notas
				durations.append(str(element.duration.quarterLength))
				isNoteOrChord = False
				offsetBase = element.offset
		
		
	
	return notes,offsets,durations


def prepare_sequences(notes, pitchnames, n_vocab):
	""" Prepare the sequences used by the Neural Network """
	# map between notes and integers and back
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
	# print('this is note_to_int:')
	# print(note_to_int)

	sequence_length = 100
	network_input = []
	output = []
	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i + sequence_length]
		network_input.append([note_to_int[char] for char in sequence_in]) #gia 100 notes--> ena char
		#Converts each note in the extracted subsequence (sequence_in) 
		# to its corresponding integer representation using the note_to_int dictionary
		output.append(note_to_int[sequence_out]) #den to xrhsimopoiei
    #to network input einai 2D matrix me N rows kai 100 columns
	n_patterns = len(network_input)

	# reshape the input into a format compatible with LSTM layers
	normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
	# normalize input
	normalized_input = normalized_input / float(n_vocab)

	return (network_input, normalized_input)

def create_network(network_input_notes, n_vocab_notes, network_input_offsets, n_vocab_offsets, network_input_durations, n_vocab_durations):
	# Branch of the network that considers notes
	inputNotesLayer = Input(shape=(network_input_notes.shape[1], network_input_notes.shape[2]))
	inputNotes = LSTM(
		256,
		input_shape=(network_input_notes.shape[1], network_input_notes.shape[2]),
		return_sequences=True
	)(inputNotesLayer)
	inputNotes = Dropout(0.2)(inputNotes)
	
	# Branch of the network that considers note offset
	inputOffsetsLayer = Input(shape=(network_input_offsets.shape[1], network_input_offsets.shape[2]))
	inputOffsets = LSTM(
		256,
		input_shape=(network_input_offsets.shape[1], network_input_offsets.shape[2]),
		return_sequences=True
	)(inputOffsetsLayer)
	inputOffsets = Dropout(0.2)(inputOffsets)
	
	# Branch of the network that considers note duration
	inputDurationsLayer = Input(shape=(network_input_durations.shape[1], network_input_durations.shape[2]))
	inputDurations = LSTM(
		256,
		input_shape=(network_input_durations.shape[1], network_input_durations.shape[2]),
		return_sequences=True
	)(inputDurationsLayer)
	#inputDurations = Dropout(0.3)(inputDurations)
	inputDurations = Dropout(0.2)(inputDurations)
	
	#Concatentate the three input networks together into one branch now
	inputs = concatenate([inputNotes, inputOffsets, inputDurations])
	
	# A cheeky LSTM to consider everything learnt from the three separate branches
	x = LSTM(512, return_sequences=True)(inputs)
	x = Dropout(0.3)(x)
	x = LSTM(512)(x)
	x = BatchNorm()(x)
	x = Dropout(0.3)(x)
	x = Dense(256, activation='relu')(x)
	
	#Time to split into three branches again...
	
	# Branch of the network that classifies the note
	outputNotes = Dense(128, activation='relu')(x)
	outputNotes = BatchNorm()(outputNotes)
	outputNotes = Dropout(0.3)(outputNotes)
	outputNotes = Dense(n_vocab_notes, activation='softmax', name="Note")(outputNotes)
	
	# Branch of the network that classifies the note offset
	outputOffsets = Dense(128, activation='relu')(x)
	outputOffsets = BatchNorm()(outputOffsets)
	outputOffsets = Dropout(0.3)(outputOffsets)
	outputOffsets = Dense(n_vocab_offsets, activation='softmax', name="Offset")(outputOffsets)
	
	# Branch of the network that classifies the note duration
	outputDurations = Dense(128, activation='relu')(x)
	outputDurations = BatchNorm()(outputDurations)
	outputDurations = Dropout(0.3)(outputDurations)
	outputDurations = Dense(n_vocab_durations, activation='softmax', name="Duration")(outputDurations)
	
	# Tell Keras what our inputs and outputs are 
	model = Model(inputs=[inputNotesLayer, inputOffsetsLayer, inputDurationsLayer], outputs=[outputNotes, outputOffsets, outputDurations])
	
	#Adam seems to be faster than RMSProp and learns better too 
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.load_weights('weights-improvement-182-0.8113-bigger.hdf5')

	return model

def generate_notes(startt, startt2, startt3, model, network_input_notes, network_input_offsets, network_input_durations, notenames, offsetnames, durationames, n_vocab_notes, n_vocab_offsets, n_vocab_durations):
	""" Generate notes from the neural network based on a sequence of notes """
	# pick a random sequence from the input as a starting point for the prediction
	# 
	#start = numpy.random.randint(0, len(network_input_notes)-1)
	#start2 = numpy.random.randint(0, len(network_input_offsets)-1)
	#start3 = numpy.random.randint(0, len(network_input_durations)-1)

	int_to_note = dict((number, note) for number, note in enumerate(notenames)) #kanei to antistrofo apto prep_seq
	#print(int_to_note)
	int_to_offset = dict((number, note) for number, note in enumerate(offsetnames))
	int_to_duration = dict((number, note) for number, note in enumerate(durationames))
    # use specific pattern as a starting point
	pattern = startt[0]
	# print("THIS is pattern:")
	# print(pattern)
	pattern2 = startt2[0]
	pattern3 = startt3[0]
	prediction_output = []
	# for i in range(100):
	# 	row = [pattern[i], pattern2[i], pattern3[i]]
	# 	prediction_output.append(row)

	

	# generate notes or chords
	for note_index in range(400):
		note_prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
		predictedNote = note_prediction_input[-1][-1][-1]
		note_prediction_input = note_prediction_input / float(n_vocab_notes)
		
		offset_prediction_input = numpy.reshape(pattern2, (1, len(pattern2), 1))
		offset_prediction_input = offset_prediction_input / float(n_vocab_offsets)
		
		duration_prediction_input = numpy.reshape(pattern3, (1, len(pattern3), 1))
		duration_prediction_input = duration_prediction_input / float(n_vocab_durations)
        #uses pattern as an input for the prediction
		prediction = model.predict([note_prediction_input, offset_prediction_input, duration_prediction_input], verbose=0)
		print("PREDICTIONS:")
		print(prediction)
	    #returns the index of the maximum value in the first row of the array
		index = numpy.argmax(prediction[0]) #chooses prediction with highest probability
		#print("This is index:")
		#print(index)
		result = int_to_note[index] #vriskei to index sto leksiko kai thn antistoixh nota
		#print(result)
		
		offset = numpy.argmax(prediction[1])
		offset_result = int_to_offset[offset]
		#print("offset")
		#print(offset_result)
		
		duration = numpy.argmax(prediction[2])
		duration_result = int_to_duration[duration]
		#print("duration")
		#print(duration_result)
		
		#print("Next note: " + str(int_to_note[predictedNote]) + " - Duration: " + str(int_to_duration[duration]) + " - Offset: " + str(int_to_offset[offset]))
		
		
		#
		prediction_output.append([result, offset_result, duration_result])

		pattern.append(index)
		pattern2.append(offset)
		pattern3.append(duration)

		pattern = pattern[1:len(pattern)]
		pattern2 = pattern2[1:len(pattern2)]
		pattern3 = pattern3[1:len(pattern3)]

	return prediction_output

def create_midi(prediction_output_all):
	""" convert the output from the prediction to notes and create a midi file
		from the notes """
	offset = 0
	output_notes = []
	
	offsets = []
	durations = []
	notes = []
	
	for x in prediction_output_all:
		#print(x)
		notes = numpy.append(notes, x[0])
		try:
			offsets = numpy.append(offsets, float(x[1]))
		except:
			num, denom = x[1].split('/')
			x[1] = float(num)/float(denom)
			offsets = numpy.append(offsets, float(x[1]))
			
		durations = numpy.append(durations, x[2])
	
	print("---")
	# print("Offsets")
	# print(offsets)
	# print("Durations")
	# print(durations)
	print("Creating Midi File...")

	# create note and chord objects based on the values generated by the model
	x = 0 # this is the counter
	for pattern in notes:
		# pattern is a chord
		if ('.' in pattern) or pattern.isdigit():
			notes_in_chord = pattern.split('.')
			notes = []
			for current_note in notes_in_chord:
				new_note = note.Note(int(current_note))
				new_note.storedInstrument = instrument.Piano()
				notes.append(new_note)
			new_chord = chord.Chord(notes)
			
			try:
				new_chord.duration.quarterLength = float(durations[x])
			except:
				num, denom = durations[x].split('/')
				new_chord.duration.quarterLength = float(num)/float(denom)
			
			new_chord.offset = offset
			
			output_notes.append(new_chord)
		# pattern is a note
		else:
			new_note = note.Note(pattern)
			new_note.offset = offset
			new_note.storedInstrument = instrument.Piano()
			try:
				new_note.duration.quarterLength = float(durations[x])
			except: #ama einai klasma to duration to metatrepei se dekadiko
				num, denom = durations[x].split('/')
				new_note.duration.quarterLength = float(num)/float(denom)
			
			output_notes.append(new_note)

		# increase offset each iteration so that notes do not stack
		try:
			offset += offsets[x]
		except: 
			num, denom = offsets[x].split('/')
			offset += num/denom
				
		x = x+1
  
	midi_stream = stream.Stream(output_notes)
	print(new_note.duration.quarterLength)
	#print(new_n)
	#print(output_notes)

	midi_stream.write('midi', fp='2024_output.mid')
	
	print("Midi created!")

if __name__ == '__main__':
	generate()
