# Videospoof

## Prevent face recognition during meetings

This script tries to apply a facemask, generated by the Fawkes face protection system

Requirements: cv2, dlib, numpy, fawkes binary, face recognition model

Dlib 68 point recognition model cam be found here: http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

### Install

Download the face recognition model to the current directory and the fawkes binary (https://github.com/Shawn-Shan/fawkes/releases)

`pip3 install -r requirements.txt`

### Run

`python3 mesh.py <input_image> <original_image> <processed_fawkes_image>`

Note: webcam support and the main code is still WIP

Credits:
* https://github.com/Shawn-Shan/fawkes
* https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
