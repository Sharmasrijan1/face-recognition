import cv2
import sys
import os

def main():
    """Main application for real-time face detection"""
    print("Initializing Face Detection System...")
    
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise ValueError("Could not load face cascade classifier")
        print("Face detector initialized successfully!")
    except Exception as e:
        print(f"Error initializing face detector: {e}")
        sys.exit(1)
    
    print("Starting camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("err: could not open camera")
        print("please check if your camera is connected and turned on.")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Face Detection Started!")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press 'SPACE' to pause/resume")
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            frame_count += 1
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            frame_with_faces = frame.copy()
            
            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(frame_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                label = f"Face {i + 1}"
                cv2.putText(frame_with_faces, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_with_faces, f"Faces detected: {len(faces)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(frame_with_faces, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.putText(frame_with_faces, "Press 'q' to quit", 
                       (10, frame_with_faces.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Detection System', frame_with_faces)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s') and not paused:
            filename = f"face_detection_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame_with_faces)
            print(f"Frame saved as {filename}")
        elif key == ord(' '):
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"Detection {status}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection system stopped.")

if __name__ == "__main__":
    main()
