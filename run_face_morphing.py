from collections import deque
from datetime import datetime
import os
import random
import shutil

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
# from picamera2 import Picamera2
import yaml

from image_processing_utils import crop_face_with_margin, get_delauney_triangles, \
    get_triangulation_indexes, get_face_landmarks, get_additional_landmarks, morph



# Initialize MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                               max_num_faces=1,
                                               refine_landmarks=True,
                                               min_detection_confidence=0.5)
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


DEBUG = False


def generate_continuous_morphs(image_path_1, image_path_2, num_frames=10):

    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

    # Read the images.
    image_1 = cv2.imread(image_path_1)
    image_2 = cv2.imread(image_path_2)

    # Detect each face.
    detection_1 = mp_face_detection.process(image_1).detections[0]
    detection_2 = mp_face_detection.process(image_2).detections[0]

    # Draw a bounding box around each face.
    bounding_box_1 = detection_1.location_data.relative_bounding_box
    bounding_box_2 = detection_2.location_data.relative_bounding_box

    # Crop the face (relative to the bounding box)
    cropped_face_1 = crop_face_with_margin(image_1, bounding_box_1, margin=2, bb_type="mediapipe")
    cropped_face_2 = crop_face_with_margin(image_2, bounding_box_2, margin=2, bb_type="mediapipe")

    if DEBUG == True:
        cv2.imshow("Cropped face", cropped_face_1)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

    # Resize each face. TODO: resize while keeping aspect ratio (resize and crop)
    cropped_resized_face_1 = cv2.resize(cropped_face_1, (600, 600), interpolation=cv2.INTER_AREA)
    cropped_resized_face_2 = cv2.resize(cropped_face_2, (600, 600), interpolation=cv2.INTER_AREA)

    # Get the face landmarks and additional landmarks
    face_landmarks_1 = get_face_landmarks(cropped_resized_face_1)
    face_landmarks_2 = get_face_landmarks(cropped_resized_face_2)

    # Get additional landmarks on the edges of the image.
    additional_landmarks_1 = get_additional_landmarks(cropped_resized_face_1)
    additional_landmarks_2 = get_additional_landmarks(cropped_resized_face_2)

    # Use all the landmarks.
    all_landmarks_1 = face_landmarks_1 + additional_landmarks_1
    all_landmarks_2 = face_landmarks_2 + additional_landmarks_2


    if DEBUG == True:
        for (x, y) in all_landmarks_1:
            display_copy = cropped_resized_face_1.copy()
            # Draw a small circle at each landmark point
            cv2.circle(display_copy, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.imshow("face with landmarks", display_copy)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

    # Get the Delauney Triangles for one of the images.
    delauney_triangles = get_delauney_triangles(cropped_resized_face_1, all_landmarks_1)

    if DEBUG:
        display_copy = cropped_resized_face_1.copy()
        for t in delauney_triangles:
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))

            cv2.line(display_copy, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(display_copy, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(display_copy, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("delauney triangles", display_copy)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

    # Convert these triangles to indexes, which are generic.
    triangulation_indexes = get_triangulation_indexes(all_landmarks_1, delauney_triangles)

    # Collect the partial morphs
    partial_morphs_1 = []
    partial_morphs_2 = []

    # Use various values of alpha
    alphas = np.linspace(0, 1, 15).tolist()
    for alpha in alphas:
        # Morph-align the face to the target face.
        morphed_face_1 = morph(cropped_resized_face_2,
                               cropped_resized_face_1,
                               all_landmarks_2,
                               all_landmarks_1,
                               triangulation_indexes,
                               alpha=alpha)
        
        morphed_face_2 = morph(cropped_resized_face_1, # Swapped order of faces.
                               cropped_resized_face_2,
                               all_landmarks_1,
                               all_landmarks_2,
                               triangulation_indexes,
                               alpha=alpha)

        partial_morphs_1.append(morphed_face_1)
        partial_morphs_2.append(morphed_face_2)

    # Reverse this list for morphing.
    partial_morphs_1 = list(reversed(partial_morphs_1))

    if DEBUG == True:
        for i, partial_morph in enumerate(partial_morphs_1):
            cv2.imshow(f"Partial morph 1: {i}", partial_morph)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                pass

    if DEBUG == True:
        for i, partial_morph in enumerate(partial_morphs_2):
            cv2.imshow(f"Partial morph 2: {i}", partial_morph)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                pass

    # Collect the blended faces.
    blended_faces = []

    # Reverse the alphas for blending.
    alphas = list(reversed(alphas))

    # Blend together all the faces.
    for i, alpha in enumerate(alphas):
        blended_face = cv2.addWeighted(partial_morphs_1[i],
                                       alpha,
                                       partial_morphs_2[i],
                                       1 - alpha, 0)
        blended_faces.append(blended_face)

    if DEBUG == True:
        for i, blended_face in enumerate(blended_faces):
            cv2.imshow(f"Blended morphs: {i}", blended_face)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                pass
    
    return blended_faces


def is_face_looking_forward(face_image):
    """
    Returns true if the face is forward.
    """
    # Process the image.
    results = mp_face_mesh.process(face_image)

    # Convert back to the right colors.
    image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

    # Get image size information.
    img_h , img_w, img_c = image.shape

    # Collect the 2D and 3D landmarks.
    face_2d = []
    face_3d = []

    # If there are landmarks (face detected), continue
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                    if idx ==1:
                        nose_2d = (lm.x * img_w,lm.y * img_h)
                        nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                    x,y = int(lm.x * img_w),int(lm.y * img_h)

                    face_2d.append([x,y])
                    face_3d.append(([x,y,lm.z]))


            # Get 2D coordinates
            face_2d = np.array(face_2d, dtype=np.float64)

            # Get 3D coordinates
            face_3d = np.array(face_3d,dtype=np.float64)

            # Calculate the orientation of the face.
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length,0,img_h/2],
                                [0,focal_length,img_w/2],
                                [0,0,1]])
            distortion_matrix = np.zeros((4,1),dtype=np.float64)

            success, rotation_vec, translation_vec = cv2.solvePnP(face_3d,
                                                                  face_2d,
                                                                  cam_matrix,
                                                                  distortion_matrix)

            # Get the rotational vector of the face.
            rmat, jac = cv2.Rodrigues(rotation_vec)

            angles, mtxR, mtxQ ,Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Check which way the face is oriented.
            # Previously -3 3 -3 7
            if y < -5: # Looking Left
                return False
            elif y > 5: # Looking Right
                return False
            elif x < -5: # Looking Down
                return False
            elif x > 7: # Looking Up
                return False
            else: # Looking Forward
                return True
    
    # No face was detected!
    else:
        return None



if __name__ == "__main__":
    # # Set the display
    # os.environ["DISPLAY"] = ':0'

    # # Rotate the screen
    # os.system("WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 270")

    # # Hide the cursor
    # os.system("unclutter -idle 0 &")

    # Load the config.yaml file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Hold the most recent N faces (as a deque data structure).
    # This is a list of lists, where each sub-list is `[encoding, path_to_face]`
    face_dataset = deque(maxlen=config["face_memory"])

    # Where the faces will be saved.
    face_save_directory = config["save_directory"]

    # Delete the folder and its contents, if it exists, and recreate it.
    if os.path.exists(face_save_directory):
        shutil.rmtree(face_save_directory)
    os.makedirs(face_save_directory)

    # Make the display fullscreen
    cv2.namedWindow("Morphs", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Morphs", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Set the current face variable
    current_face = None

    # Main event loop
    while True:
        # Get a picture from the webcam.
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()

        # Reduce the frame size to speed up face recognition.
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR (OpenCV format) to RGB (face_recognition format).
        rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])

        # Detect faces and get encodings for the current frame.
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Collect all the face images.
        faces = []
        if face_locations:
            for location in face_locations:
                cropped_face = crop_face_with_margin(frame,
                                                     location,
                                                     config["margin"],
                                                     bb_type="face_recognition")
                faces.append(cropped_face)

        # Get the encodings for previously observed faces, for comparison.
        known_face_encodings = [encoding for encoding, _ in face_dataset]

        # Collect all the faces that have not yet been observed.
        previously_unobserved_faces = []

        if face_encodings:
            for face, face_encoding in zip(faces, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings,
                                                        face_encoding,
                                                        tolerance=config["tolerance"])
                # If the face hasn't previously been seen, continue.
                if not any(matches):
                    # Flip the image for selfie view.
                    face = cv2.cvtColor(cv2.flip(face, 1), cv2.COLOR_BGR2RGB)
                    previously_unobserved_faces.append([face, face_encoding])

        # Collect all the forward looking faces.
        forward_looking_faces = [[face, face_encoding ] for face, face_encoding \
                                 in previously_unobserved_faces \
                                 if is_face_looking_forward(face_image=face)]

        # Save all the faces, and update the database.
        new_face_paths = []
        if forward_looking_faces:
            for face, face_encoding in forward_looking_faces:
                # Get the filename.
                filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
                path_to_face_image = os.path.join(face_save_directory,
                                                filename + ".jpg")

                # Fix the colors
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

                # Save the file.
                cv2.imwrite(path_to_face_image, face)

                # This is a list of lists, where each sub-list is `[encoding, path_to_face]`
                face_dataset.append([face_encoding, path_to_face_image])

                # Write the the `new_face_paths`, to be displayed.
                new_face_paths.append(path_to_face_image)


        # Display the new faces.
        if new_face_paths:
            for new_face_path in new_face_paths:

                # If it is the first time going through the loop, morph the image with itself.
                if current_face == None:
                    current_face = new_face_path

                # Get all the morph images.
                images = generate_continuous_morphs(current_face,
                                                    new_face_path,
                                                    num_frames=10)

                # Display the morphs
                for image in images:
                    cv2.imshow("Blended morphs", image)
                    if cv2.waitKey(100) & 0xFF == ord("q"):
                        pass

                # Set the current face to the one just looked at, for morph chaining.
                current_face = new_face_path

        # If no new faces were detected, choose one at random from the dataset.
        else:
            # If it is the first pass through a no face is detected, keep trying.
            if current_face == None:
                pass
    
            else:
                # Randomly choose a new face.
                new_face_path = random.choice(face_dataset)[1]

                # Get all the morph images.
                images = generate_continuous_morphs(current_face,
                                                    new_face_path,
                                                    num_frames=10)

                # Display the morphs
                for image in images:
                    cv2.imshow("Blended morphs", image)
                    if cv2.waitKey(100) & 0xFF == ord("q"):
                        pass

                # Set the current face to the one just looked at, for morph chaining.
                current_face = new_face_path
