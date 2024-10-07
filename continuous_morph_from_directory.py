import os
import cv2
import numpy as np
import random
import mediapipe as mp
from image_processing_utils import get_delauney_triangles, get_triangulation_indexes, crop_face_with_margin, get_additional_landmarks, get_face_landmarks, morph, get_triangulation_indexes_for_basis_image



# Initialize MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
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
        blended_face = cv2.addWeighted(partial_morphs_1[i], alpha, partial_morphs_2[i], 1 - alpha, 0)
        blended_faces.append(blended_face)

    if DEBUG == True:
        for i, blended_face in enumerate(blended_faces):
            cv2.imshow(f"Blended morphs: {i}", blended_face)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                pass
    
    return blended_faces



if __name__ == "__main__":
    # Directory with all the saved images.
    image_files_dir = "_1_INPUT_IMAGES/Congress"

    # Paths to all the saved images.
    all_image_files = [os.path.join(image_files_dir, file) for file in os.listdir(image_files_dir)]

    # Get the path to the first image.
    image_path_1 = random.choice(all_image_files)
    
    while True:
        try:
            image_path_2 = random.choice(all_image_files)

            images = generate_continuous_morphs(image_path_1, image_path_2, num_frames=10)

            # Display the morphs
            for image in images:
                cv2.imshow("Blended morphs", image)
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    pass

            image_path_1 = image_path_2
        except Exception as e:
            print(e)
            pass
