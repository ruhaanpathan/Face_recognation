import cv2
import face_recognition
import numpy as np
import os
import uuid


# Directory to store the captured photos
save_directory = "C:\python_face_rec\known_faces"
os.makedirs(save_directory, exist_ok=True)

# List of texts to display on each photo
texts = [
    "look in the camera and press s",
    "look at your left side and press s",
    "look at your right side and press s",
    "look upwards and press s",
    "look downwards and press s"
]
# Function to capture photos
def capture_photos():
    video_capture = cv2.VideoCapture(0)

    for i in range(5):
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            
            # Add text at the bottom center of the frame for display
            text = texts[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (255, 255, 255)  # White color
            thickness = 2
            
            # Get the width and height of the text box
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2  # Center the text
            text_y = frame.shape[0] - 10  # Position the text 10 pixels from the bottom
            
            # Create a copy of the frame to save without text
            frame_to_save = frame.copy()
            
            # Put the text on the frame for display
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow("Capture Photo", frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                random_filename = f"{uuid.uuid4()}.jpg"
                photo_path = os.path.join(save_directory, random_filename)
                cv2.imwrite(photo_path, frame_to_save)  # Save the frame without text
                print(f"Photo saved: {photo_path}")
                break
            elif key == ord('q'):
                print("Exiting without saving.")
                video_capture.release()
                cv2.destroyAllWindows()
                return None

    video_capture.release()
    cv2.destroyAllWindows()

# Main Function
def main():
    capture_photos()

if __name__ == "__main__":
    main()
    






# Function to load known faces from a directory
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            known_image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(known_image)
            
            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(filename.split('.')[0])  # Use filename (without extension) as name

    return known_face_encodings, known_face_names

# Load known faces
known_faces_dir = "C:\python_face_rec\known_faces"  # Directory containing known face images
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame
    ret, frame = video_capture.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Failed to capture image")
        break

    # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with all known faces
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # Find the best match
        best_match_index = np.argmin(distances)
        best_match_distance = distances[best_match_index]
        
        # Calculate similarity percentage
        similarity_percentage = (1 - best_match_distance) * 100

        # Check if the similarity is 65% or more
        if similarity_percentage >= 65:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            name = known_face_names[best_match_index]
            cv2.putText(frame, f"Recognised: {similarity_percentage:.2f}%", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            print(f"Similarity: {similarity_percentage:.2f}% - Recognised as {name}")
        else:
            print(f"Similarity: {similarity_percentage:.2f}% - Not Recognised")

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()