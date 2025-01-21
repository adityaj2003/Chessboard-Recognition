import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as clstr
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input


def canny_lines(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def filter_close_lines(lines, threshold=150):
    lines = sorted(lines, key=lambda x: x[0])

    filtered_lines = []
    current_group = [lines[0]]

    for line in lines[1:]:
        if abs(line[0] - current_group[-1][0]) <= threshold:
            current_group.append(line)
        else:
            # Choose 1 line from many close lines
            representative_line = sorted(current_group, key=lambda x: x[0])[len(current_group) // 2]
            filtered_lines.append(representative_line)
            current_group = [line]

    # Add the selected line from the last group
    if current_group:
        representative_line = sorted(current_group, key=lambda x: x[0])[len(current_group) // 2]
        filtered_lines.append(representative_line)

    return filtered_lines

def get_lines(lines):
    h, v = [], []
    for distance, angle in lines:
        if angle < np.pi / 4 or angle > np.pi - np.pi / 4:
            v.append([distance, angle])
        else:
            h.append([distance, angle])

    # Filter close horizontal and vertical lines
    h = filter_close_lines(h)
    v = filter_close_lines(v)

    return h, v


def get_points(h, v):
    points = []
    for d1, a1 in h:
        for d2, a2 in v:
            A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
            b = np.array([d1, d2])
            point = np.linalg.solve(A, b)
            points.append(point)
    return np.array(points)


def intersection_points(points, image, max_dist=50):
    Y = spatial.distance.pdist(points)
    Z = clstr.hierarchy.single(Y)
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    
    clustered_points = []
    for cluster in clusters.values():
        mean_point = np.mean(np.array(cluster), axis=0)
        clustered_points.append(mean_point)
    
    clustered_points = np.array(clustered_points)
    
    # Convert image to BGR format for OpenCV visualization
    if len(image.shape) == 2:  # Grayscale to BGR
        image = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
    
    # Draw original points in yellow
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), radius=5, color=(0, 255, 255), thickness=-1)
    
    # Draw clustered points in red
    for point in clustered_points:
        cv2.circle(image, (int(point[0]), int(point[1])), radius=7, color=(0, 0, 255), thickness=-1)
    
    # Display the image using OpenCV
    cv2.imshow("Intersection Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return clustered_points


def process_image(filename):
    img = cv2.imread(filename)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = canny_lines(gray)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        return None
    lines = np.reshape(lines, (-1, 2))
    
    # Visualize the detected lines
    img_with_lines = img.copy()
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    h, v = get_lines(lines)
    points = get_points(h, v)
    points = intersection_points(points, img)

    cv2.destroyAllWindows()
    return points



def filter_equidistant_points(points, error_margin=0.4):
    if len(points) == 0 or len(points[0]) != 2:
        raise ValueError("Points must be a 2-dimensional array with shape (n, 2).")

    points = np.array(points)

    # Calculate average distance of the three nearest neighbors for each point
    avg_distances = []
    for point in points:
        distances = np.array([spatial.distance.euclidean(point, other_point) for other_point in points if not np.array_equal(point, other_point)])
        three_nearest = np.partition(distances, 3)[:3]
        avg_distances.append(np.mean(three_nearest))

    # Compute the overall average of these average distances
    overall_avg_distance = np.mean(avg_distances)

    # Filter points based on deviation from overall average distance
    filtered_points = []
    for i, point in enumerate(points):
        if abs(avg_distances[i] - overall_avg_distance) <= error_margin * overall_avg_distance:
            filtered_points.append(point)

    return np.array(filtered_points)



def group_points_into_rows(points, y_threshold=100):
    rows = []
    used_points = set()

    for point in points:
        point_tuple = tuple(point)
        if point_tuple not in used_points:
            same_row = [p for p in points if abs(p[1] - point[1]) < y_threshold and tuple(p) not in used_points]
            if len(same_row) >= 9:
                same_row_sorted = sorted(same_row, key=lambda p: p[0])
                rows.append(same_row_sorted[:9]) 
                used_points.update(tuple(p) for p in same_row_sorted[:9])
    rows.sort(key=lambda row: row[0][1])

    return rows




def extract_subsquare(img, points, row_index, total_rows, col_index, total_cols):
    """
    Extract a subsquare from the image with perspective adjustment for the border squares.
    """
    tl, tr, bl, br = points

    expansion_amount = 100

    # Expand the first and last rows to account for perspective distortion
    if row_index == 0:  # First row
        tl[1] -= expansion_amount*2
        tr[1] -= expansion_amount*2
    elif row_index == total_rows - 2:
        bl[1] += expansion_amount*2
        br[1] += expansion_amount*2

    # Expand the first and last columns
    if col_index == 0:
        tl[0] -= expansion_amount
        bl[0] -= expansion_amount
    elif col_index == total_cols - 2:
        tr[0] += expansion_amount
        br[0] += expansion_amount

    width = int(max(spatial.distance.euclidean(tl, tr), spatial.distance.euclidean(bl, br)))
    height = int(max(spatial.distance.euclidean(tl, bl), spatial.distance.euclidean(tr, br)))
    dst = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype='float32')
    M = cv2.getPerspectiveTransform(np.array([tl, tr, bl, br], dtype='float32'), dst)
    subsquare = cv2.warpPerspective(img, M, (width, height))
    return subsquare




def form_subsquares_from_rows(rows, img):
    height, width, channels = img.shape
    subsquares = []
    total_rows = len(rows)
    total_cols = len(rows[0])

    for row_index in range(total_rows - 1):
        for col_index in range(total_cols - 1):
            # Determine the corner points for the current subsquare
            subsquare_points = [rows[row_index][col_index].copy(), 
                                rows[row_index][col_index + 1].copy(), 
                                rows[row_index + 1][col_index].copy(), 
                                rows[row_index + 1][col_index + 1].copy()]
            
            # Extract the subsquare
            subsquare = extract_subsquare(img, subsquare_points, row_index, total_rows, col_index, total_cols)
            subsquares.append(subsquare)

    return subsquares


def predict_pieces(subsquares, model):
    processed_subsquares = []
    for subsquare in subsquares:
        if len(subsquare.shape) == 2 or subsquare.shape[2] == 1:
            subsquare = cv2.cvtColor(subsquare, cv2.COLOR_GRAY2RGB)
        resized_subsquare = cv2.resize(subsquare, (299, 299))
        
        processed_subsquares.append(resized_subsquare/255)
    #Batch processing
    batch_subsquares = np.array(processed_subsquares)
    predictions = model.predict(batch_subsquares)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels



def form_fen_string(predicted_labels):
    if len(predicted_labels) != 64:
        raise ValueError("Expected 64 labels for an 8x8 chess board")

    fen_rows = []
    for i in range(0, len(predicted_labels), 8):
        fen_row = ''.join(str(label_to_fen(label)) for label in predicted_labels[i:i+8])
        fen_rows.append(fen_row)
    fen_string = '/'.join(fen_rows)
    return fen_string


#{'B': 0, 'K': 1, 'N': 2, 'P': 3, 'Q': 4, 'R': 5, '_': 6, 'b': 7, 'k': 8, 'n': 9, 'p': 10, 'q': 11, 'r': 12}
def label_to_fen(label):
    label_mapping = {
        0: 'B',
        1: 'K',
        2: 'N',
        3: 'P',
        4: 'Q',
        5: 'R',
        6: '1',
        7: 'b',
        8: 'k',
        9: 'n',
        10: 'p',
        11: 'q',
        12: 'r',
    }
    return label_mapping.get(label, '1')



def preprocess_and_display_subsquares(subsquares):
    processed_subsquares = []
    for subsquare in subsquares:
        if len(subsquare.shape) == 2 or subsquare.shape[2] == 1:
            subsquare = cv2.cvtColor(subsquare, cv2.COLOR_GRAY2RGB)
        resized_subsquare = cv2.resize(subsquare, (399, 399))
        normalized_subsquare = resized_subsquare / 255.0
        processed_subsquares.append(normalized_subsquare)

    cv2.destroyAllWindows()
    return np.array(processed_subsquares)

def calculate_accuracy(predicted_fen, ground_truth_fen):
    correct = sum(p == g for p, g in zip(predicted_fen, ground_truth_fen))
    return correct / len(ground_truth_fen) * 100

image_filenames = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg']
ground_truth_fens = [
    # Expanded FEN strings with individual '1's
    "rnbqkbnr/pppppppp/11111111/11111111/11111111/11111111/PPPPPPPP/RNBQKBNR",
    "rnbqkbnr/pppppppp/11111111/11111111/1111P111/11111111/PPPP1PPP/RNBQKBNR",
    "rnbqkbnr/pppp1ppp/11111111/1111p111/1111P111/11111111/PPPP1PPP/RNBQKBNR",
    "r1b1kbnr/1pp11ppp/p1p11111/11111111/111NP111/11111111/PPP11PPP/RNB1K11R"
]

model_resnet = load_model('chess_piece_classifier_resnet_model.h5')

for filename, gt_fen in zip(image_filenames, ground_truth_fens):
    img = cv2.imread(filename)
    points = process_image(filename)
    points = filter_equidistant_points(points)
    rows = group_points_into_rows(points)
    subsquares = form_subsquares_from_rows(rows, img)


    predicted_labels_resnet = predict_pieces(subsquares, model_resnet)
    predicted_fen_resnet = form_fen_string(predicted_labels_resnet)

    accuracy_resnet = calculate_accuracy(predicted_fen_resnet, gt_fen)

    print(f"Image: {filename}")
    print(f"Ground Truth FEN: {gt_fen}")
    print(f"ResNet Model FEN: {predicted_fen_resnet}, Accuracy: {accuracy_resnet}%")
    print("-" * 50)
