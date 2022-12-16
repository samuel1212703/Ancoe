import face_recognition
import cv2
import numpy as np
import glob
import os
import easyocr

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

format_sub_point = "- "
format_sub_sub_point = "-" + format_sub_point
known_faces_path = 'Known_People\\'
image_formats = [".png", ".jpg"]
standard_image_format = image_formats[0]
frame_saved_image_name = "frame" + standard_image_format
general_image_resize_smaller = 0.15


def known_people_configuration(known_faces_path):
    known_face_names = []
    known_face_encodings = []
    person_folders = []

    for person_folder in glob.glob(known_faces_path + "*"):
        person_folders.append(person_folder)
        person_name = str(person_folder).replace(known_faces_path, "")
        print(person_name + " -----------------------------------")
        for format in image_formats:
            for image_file in glob.glob(person_folder + "\*" + format):
                # Add name to lidt taken from folder names
                known_face_names.append(person_name)
                # Add encoding made from images
                person_image = face_recognition.load_image_file(image_file)
                print(format_sub_point + "Image from {}".format(image_file))
                try:
                    known_face_encodings.append(face_recognition.face_encodings(person_image)[0])
                    print(format_sub_sub_point + "Image succesfully configured")
                except:
                    print(format_sub_sub_point + "Could not find faces in image")

    return [known_face_names, known_face_encodings, person_folders]


if __name__ == "__main__":
    print("Getting Known_Faces data")
    known_face_names, known_face_encodings, person_folders = known_people_configuration(known_faces_path)
    print("\nRunning analysis")
    while True:
        # Process every second frame, remove for higher quality (at a cost)
        process_this_frame = not process_this_frame

        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=general_image_resize_smaller, fy=general_image_resize_smaller)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            unidentified_name = "Unidentified"
            unidentified_amount = len(person_folders)
            if unidentified_amount > 9:
                unidentified_name = unidentified_name + "0"
                unidentified_amount = unidentified_amount % 100
            elif unidentified_amount < 9:
                unidentified_name = unidentified_name + "00"
                unidentified_amount = unidentified_amount % 100

            unidentified_name = unidentified_name + str(unidentified_amount)

            if name == "Unknown":
                # If person is in the frame with someone, name it accordingly
                if len(face_locations) > 1:
                    name = face_names[0] + standard_image_format
                # Else setup folder for the apparantly unknown person
                else:
                    unidentified_person_path = known_faces_path + unidentified_name
                    if not os.path.exists(unidentified_person_path):
                        os.makedirs(unidentified_person_path)
                    cv2.imwrite(unidentified_person_path + '\\' + unidentified_name +
                                str(len(glob.glob(unidentified_person_path)) + 1) + standard_image_format, small_frame)
            face_names.append(name)

        # OCR
        small_frame = cv2.resize(frame, (0, 0), fx=general_image_resize_smaller, fy=general_image_resize_smaller)[:, :, ::-1]
        reader = easyocr.Reader(['da', 'en'])  # this needs to run only once to load the model into memory
        result = reader.readtext(small_frame)
        print("Result: ", result)

        # Graphics for faces
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)

            # Variable for relative size
            name_field_size = (bottom - top)/10

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, int(bottom - name_field_size)), (right, bottom), (0, 0, 0, 0.75), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, int(bottom - name_field_size/4)), font, name_field_size/50, (255, 255, 255), 1)

        # Graphics for translations

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
