from pathlib import Path
import numpy as np
import cv2

class PortraitCapturer:
    def __init__(self, camera_id, portrait_folder):
        self.camera_id = camera_id
        self.portrait_folder = Path(portrait_folder)
        self.portrait_folder.mkdir(parents=True, exist_ok=True)

    def capture_portrait(self):
        print(f">>>> Capturing portrait")
        cv2.namedWindow('portrait', cv2.WINDOW_FULLSCREEN)

        # define video capture
        cap = cv2.VideoCapture(self.camera_id)
        while True:
            # capture video frame by frame
            ret, frame = cap.read()
            if ret:
                frame_viz = frame.copy()
                frame_viz = cv2.putText(frame_viz, "Press X to capture portrait", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
                frame_viz = cv2.putText(frame_viz, "Press Q to exit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('x'):
                    cv2.imwrite(str(self.portrait_folder / 'portrait.png'), frame)
                    print('Portrait saved!')
                elif key == ord('q'):
                    break
                cv2.imshow('portrait', frame_viz)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
