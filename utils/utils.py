import re
import os


def extract_frame_number(filename):
	match = re.search(r'frame_(\d+)_', filename)
	if match:
		return int(match.group(1))
	return 0  #


def find_frames(frame_dir):
	valid_extensions = [".jpg", ".jpeg", ".png"]
	frame_names = [
		p for p in os.listdir(frame_dir)
		if os.path.splitext(p)[-1].lower() in valid_extensions
		   and "mask" not in p.lower()  # An special case to exclude files with "mask" in the name (case-insensitive)
	]

	# frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
	frame_names.sort(key=extract_frame_number)
	return frame_names