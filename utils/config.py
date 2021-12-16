sample_rate = 16000
classes_num = 88    # Number of notes of piano
pedal_classes_num = 3    # Number of pedals
pedal_map = {"64":0, "66":1, "67":2} # map pedal controller type to index
begin_note = 21     # MIDI note of A0, the lowest note of a piano.
segment_seconds = 10.	# Training segment duration
hop_seconds = 1.
frames_per_second = 100
velocity_scale = 128