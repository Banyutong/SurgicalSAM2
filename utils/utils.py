import re
import os



def extract_frame_number(filename):
    # Find all sequences of digits in the filename
    matches = re.findall(r'\d+', filename)

    if matches:
        # Combine all sequences of digits into a single string
        combined_digits = ''.join(matches)
        # Convert the combined string to an integer
        return int(combined_digits)

    return 0  # Default value if no number is found


def find_frames(frame_dir):
	valid_extensions = [".jpg", ".jpeg", ".png"]
	frame_names = [
		p for p in os.listdir(frame_dir)
		if os.path.splitext(p)[-1].lower() in valid_extensions
		   and "mask" not in p.lower()  # An special case to exclude files with "mask" in the name (case-insensitive)
	]

	# frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
	frame_names.sort(key=extract_frame_number)

	# print(f"------------{frame_names[0]}")

	return frame_names