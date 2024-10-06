from typing import List, Tuple
import numpy as np
import cv2
import mediapipe as mp



# Initialize MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def crop_face_with_margin(image : np.ndarray,
                          face_bbox,
                          margin : float,
                          bb_type: str) -> np.ndarray:
    """
    Crops a face from the image to the specified margin. If the margin extends
    beyond the image boundaries, those areas are filled with black pixels.
    Keeps the face centered in the output image.
    
    Parameters
    ----------
    image : numpy.ndarray
        The input image.
    face_bbox : mediapipe NormalizedRect object that acts like a NamedTuple
        A list with attributes xmin, ymin, width, height (values between 0 and 1).
    margin : float
        The scaling factor for the margin. Default is 1.0 (no margin).
        
    Returns
    -------
    numpy.ndarray
        The cropped face image with margin.
    """
    img_height, img_width = image.shape[:2]
    
    if bb_type == "face_recognition":
        # bb is (top, right, bottom, left)
        top, right, bottom, left = face_bbox
        x = int(left)
        y = (top)
        w = (right - left)
        h = (bottom - top)

    else:
        # Convert normalized bbox coordinates to pixel values
        x = int(face_bbox.xmin * img_width)
        y = int(face_bbox.ymin * img_height)
        w = int(face_bbox.width * img_width)
        h = int(face_bbox.height * img_height)
        
    # Calculate margin sizes
    margin_w = int(w * (margin - 1) / 2)
    margin_h = int(h * (margin - 1) / 2)
    
    # Calculate coordinates for the face region including margins
    x1 = x - margin_w
    y1 = y - margin_h
    x2 = x + w + margin_w
    y2 = y + h + margin_h
    
    # Calculate sizes for the output image
    output_w = x2 - x1
    output_h = y2 - y1
    
    # Create a black canvas for the output
    output_image = np.zeros((output_h, output_w, image.shape[2]), dtype=image.dtype)
    
    # Calculate the region to be copied from the original image
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(img_width, x2)
    src_y2 = min(img_height, y2)
    
    # Calculate the region where the face will be pasted on the black canvas
    dst_x1 = max(0, -x1)
    dst_y1 = max(0, -y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    
    # Ensure the region dimensions match exactly
    output_image[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    
    return output_image


def get_face_landmarks(image : np.ndarray) -> List[Tuple[float, float]]:
    """
    Accepts an image and returns the landmarks for a face in the image.
    Ideallly, the image should contain a face, cropped with a margin

    Parameters
    ----------
    image : np.ndarray
        An image, hopefully containing a face.
    
    Returns
    -------
    List[Tuple[float, float]]
        A list of the tuples of all the landmark locations, [(34.1, 16.6), ...]
    """
    # Process the image to get the landmarks
    results = mp_face_mesh.process(image)

    # Extract the facial landmarks
    height, width, _ = image.shape
    facial_landmarks = []
    res = results.multi_face_landmarks

    # If there are landmarks, collect their coordinates.
    if res:
        for face_landmarks in res:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                facial_landmarks.append([x, y])

    # Else return None
    else:
        return None

    return facial_landmarks


def get_additional_landmarks(image : np.ndarray) -> List[List[int]]:
    """
    Adds additional landmarks to an image. These landmarks are around the edges of
    the image. This helps with morphing so that the entire image can be tiled
    with delauney triangles.

    Parameters
    ----------
    image : np.ndarray
        An image, hopefully containing a face.

    Returns
    -------
    List[List[int, int]]
        A list of lists, where each sub-list is an additional landmark.
    """
    # subdiv.insert() cannot handle max values for edges, so add a small offset.
    # TODO: does this have to be an int? Or can it be a small float like 0.001 ?
    offset = 1

    # New coordinates to add to the landmarks
    new_coords = [
        # Corners of the image
        [0, 0],
        [image.shape[1] - offset, 0],
        [image.shape[1] - offset, image.shape[0] - offset],
        [0, image.shape[0] - offset],

        # Middle of the top, bottom, left, right sides
        [(image.shape[1] - offset) / 2, 0],
        [(image.shape[1] - offset) / 2, image.shape[0] - offset],
        [0, (image.shape[0] - offset) / 2],
        [image.shape[1] - offset, image.shape[0] / 2],
    ]

    int_coords = [(int(x), int(y)) for (x, y) in new_coords]

    return int_coords


def get_delauney_triangles(image : np.ndarray, landmark_coordinates : List[List[int]]) -> List[List[float]]:
    """
    TODO: the landmark coordinates might be of different types!
        List[List[int, int]] vs List[Tuple[float, float]] - but it still works!

    Accepts an image along with landmark coordinates, which are a list of tuples.
    The landmarks can be just the face landmarks or all the landmarks, which will
    include points along the edge of the image, not just the face.

    Returns a list of lists, where every element of the list is 6 long and contains
    the three coordinate pairs of every delauney triangle:
        [ [[x1, x2, y1, y2, z1, z2], ... ]

    NOTE: there will be more delauney triangles than points.

    Parameters
    ----------
    image : np.ndarray
        An image, hopefully containing a face.
    landmark_coordinates : List[List[int, int]]
        A list of all the landmark coordiantes. TODO: confirm this is true!
    
    Returns
    -------
    List[List[float, float, float, float, float, float]]
        A list of lists, where the sub-lists are the 6 coordinates of the triangle points
    """
    # Rectangle to be used with Subdiv2D
    size = image.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in landmark_coordinates:
        subdiv.insert(p)

    return subdiv.getTriangleList()


def get_triangulation_indexes(landmarks : List[List[float]], triangulation_points : List[List[float]]) -> List[List[int]]:
    """
    Accepts a list of landmarks, which is a list of tuples. Also accepts a list
    of triangulation points, which is a list of lists, where each sub-list is len 6
    and contains 3 of the landmark points.

    Returns a list of lists, where each sub-list if 3 elements long and contains the
    indexes of the triangulation points.

    For example, if:
        landmarks[0] = [130.4, 190.9]
        landmarks[1] = [156.0, 220.5]
        landmarks[2] = [222.2, 905.1]
    There might be a triangulation point like:
        [ 130.4, 190.9,   222.2, 905.1,   156.0, 220.5 ]
    And the indexes of these points, relative to `landmarks`, is:
        [0, 2, 1]

    NOTE: This function can be used to produce a canonical set of triangle indexes,
    which can be used for any face.

    TODO: data should be represented as floats instead of ints, but floats are of
    different lengths depending on whether they are in a list or a numpy array.

    Parameters
    ----------
    landmarks : List[List[float, float]] TODO: confirm this is true
        The face landmark coordinates.
    triangulation_points : List[List[float, float, float, float, float, float]]
        The 6 points that define a triangle
    
    Returns
    -------
    List[List[int, int, int]]
        A triangle is formed from 3 points. These are the indexes of those points.
    """
    # Combine the coordinates of each landmark into a string.
    # The strings are the keys and indexes of each key is the value in the dict.
    # Example: {'[123.566, 190.034]': 0}
    enumerated_rows = {}
    for index, row in enumerate(landmarks):
        enumerated_rows[str(list(row))] = index

    triangulation_indexes = []

    for x1, x2, y1, y2, z1, z2 in triangulation_points:
        x = str(list([int(x1), int(x2)]))
        y = str(list([int(y1), int(y2)]))
        z = str(list([int(z1), int(z2)]))

        index_x = enumerated_rows[x]
        index_y = enumerated_rows[y]
        index_z = enumerated_rows[z]

        triangulation_indexes.append([index_x, index_y, index_z])

    return triangulation_indexes


def get_triangulation_indexes_for_basis_image(basis_image_path : str,
                                              total_face_width : int,
                                              total_face_height : int,
                                              margin : float,
                                              debug : bool) -> List:
    """
    Process the basis image and return the indexes, landmarks, and cropped face.

    Parameters
    ----------
    basis_image_path : str
        The path to the basis image. This is what other faces will be mutated to look like.
    total_face_width : int
        How wide the image should ultimately be.
    total_face_height : int
        How tall the image should ultimately be.
    margin : float
        How much relative margin to add.
    debug : bool
        Show the debug images?
    
    Returns
    -------
    A list that contains:
        List[List[int, int, int]]
            The triangulation indexes.
        List[List[float, float]]
            All the landmarks.
        np.ndarray
            The basis image, cropped and processed.
    """
    # Read the basis image.
    basis_image = cv2.imread(basis_image_path)

    if debug:
        cv2.imshow("basis image", basis_image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

    # Crop the image to the face
    detection = mp_face_detection.process(basis_image).detections[0]
    bounding_box = detection.location_data.relative_bounding_box
    cropped_face = crop_face_with_margin(basis_image, bounding_box, margin)

    if debug:
        cv2.imshow("cropped face", cropped_face)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

    # Resize the face to the same size that all the output images will be.
    cropped_resized_face = cv2.resize(cropped_face, (total_face_width, total_face_height), interpolation=cv2.INTER_AREA)

    if debug:
        cv2.imshow("cropped resized face", cropped_resized_face)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

    # Get the face landmarks and additional landmarks
    face_landmarks = get_face_landmarks(cropped_resized_face)
    additional_landmarks = get_additional_landmarks(cropped_resized_face)
    all_landmarks = face_landmarks + additional_landmarks

    if debug:
        for (x, y) in all_landmarks:
            # Draw a small circle at each landmark point
            cv2.circle(cropped_resized_face, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.imshow("face with landmarks", cropped_resized_face)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

    # Get the delauney triangles
    delauney_triangles = get_delauney_triangles(cropped_resized_face, all_landmarks)
    
    if debug:
        for t in delauney_triangles:
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))

            cv2.line(cropped_resized_face, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(cropped_resized_face, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(cropped_resized_face, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("delauney triangles", cropped_resized_face)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

    # Convert these points into indexes.
    indexes = get_triangulation_indexes(all_landmarks, delauney_triangles)

    return indexes, all_landmarks, cropped_resized_face


def compute_average_coordinates(landmarks_1, landmarks_2, alpha=0):
    """
    NOTE: all current applications have alpha=0, meaning this function
    simply returns `landmarks_1`


    Computes the weighted average coordinates of 2 equal-length lists of tuples,
    `landmarks_1`, `landmarks_2`.

    Returns a list of lists, where each sub-list are the average coordinates.

    Alpha determines the amount of blending. For example:
        - alpha = 0 : returns `landmarks_1`
        - alpha = 0.25 : more similar to `landmarks_1`
        - alpha = 0.5 : equally `landmarks_1` and `landmarks_2`
        - alpha = 1 : returns `landmarks_2`

    TODO: this is not necessary for full morph!!!!
    """
    average_points = []

    if alpha == 0:
        return landmarks_1
    elif alpha == 1:
        return landmarks_2

    # Compute weighted average point coordinates
    for i in range(0, len(landmarks_1)):  # edit to index list within list
        x = (1 - alpha) * landmarks_1[i][0] + alpha * landmarks_2[i][0]
        y = (1 - alpha) * landmarks_1[i][1] + alpha * landmarks_2[i][1]
        average_points.append([x, y])

    return average_points


def applyAffineTransform(src, srcTri, dstTri, size):
    """
    Applies an affine transformation.

    Apply affine transform calculated using srcTri and dstTri to src

    TODO: properly document this function.
    """
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


def morphTriangle(img1, img2, img, t1, t2, t, margin, alpha=1):
    """
    Warps and alpha blends triangular regions from img1 and img2 to img

    If alpha=1, keep the skin of img2

    TODO: properly document this function.
    """
    alpha = 1
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(src=img1Rect, srcTri=t1Rect, dstTri=tRect, size=size)
    warpImage2 = applyAffineTransform(src=img2Rect, srcTri=t2Rect, dstTri=tRect, size=size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


def morph(image_1, image_2, landmarks_1, landmarks_2, triangulation_indexes, alpha):
    """
    Morph the image.

    TODO: properly document this function.
    TODO: same alpha for both warping and blending... dangerous!
    """
    # Allocate space for final output
    imgMorph = np.zeros(image_1.shape, dtype=image_1.dtype)

    # Get the average points between the two faces.
    average_points = compute_average_coordinates(landmarks_1, landmarks_2, alpha)

    # Read the canonical triangulation
    for line in triangulation_indexes:
        # ID's of the triangulation points
        x = line[0]
        y = line[1]
        z = line[2]

        # Coordinate pairs
        t1 = [landmarks_1[x], landmarks_1[y], landmarks_1[z]]
        t2 = [landmarks_2[x], landmarks_2[y], landmarks_2[z]]
        t = [average_points[x], average_points[y], average_points[z]]

        # Morph one triangle at a time.
        morphTriangle(image_1, image_2, imgMorph, t1, t2, t, alpha)

    return imgMorph
