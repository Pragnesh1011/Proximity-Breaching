import cv2
import numpy as np
from tkinter import messagebox
import tkinter as tk
import torch
import time

class ProximityMonitor:
    def __init__(self):
        # Initialize YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = 0.5  # Confidence threshold
        
        # Initialize tkinter for alerts
        self.root = tk.Tk()
        self.root.withdraw()
        
        # Alert cooldown parameters
        self.last_alert_time = 0
        self.alert_cooldown = 3  # Seconds between alerts
        
        # Initialize breach count
        self.breach_count = 0
        
        # Colors for visualization
        self.ZONE_COLOR = (255, 0, 0)    # Blue
        self.ALERT_COLOR = (0, 0, 255)   # Red
        self.TEXT_COLOR = (255, 255, 255) # White

    def define_monitoring_zone(self, frame):
        """
        Define the monitoring zone as a rectangle in the middle of the frame
        """
        height, width = frame.shape[:2]
        
        # Define monitoring zone (center rectangle covering 40% of frame)
        zone_width = int(width * 0.4)
        zone_height = int(height * 0.4)
        x1 = (width - zone_width) // 2
        y1 = (height - zone_height) // 2
        x2 = x1 + zone_width
        y2 = y1 + zone_height
        
        return (x1, y1, x2, y2)

    def check_breach(self, detections, monitoring_zone):
        """
        Check if any detected object intersects with the monitoring zone
        """
        zone_x1, zone_y1, zone_x2, zone_y2 = monitoring_zone
        
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            
            # Check for intersection
            if (x1 < zone_x2 and x2 > zone_x1 and 
                y1 < zone_y2 and y2 > zone_y1):
                return True
        
        return False

    def show_alert(self):
        """
        Show alert message if cooldown period has passed
        """
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_cooldown:
            self.breach_count += 1
            messagebox.showwarning("Proximity Alert", 
                                 f"Breach detected! Total breaches: {self.breach_count}")
            self.last_alert_time = current_time

    def run(self):
        # Initialize webcam
        cap = cv2.VideoCapture(1)  # Use default webcam (0)
        
        if not cap.isOpened():
            print("Error: Could not access webcam")
            return

        print("Proximity monitoring started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Define monitoring zone
            monitoring_zone = self.define_monitoring_zone(frame)
            x1, y1, x2, y2 = monitoring_zone

            # Detect objects
            results = self.model(frame)
            
            # Check for breaches
            breach_detected = self.check_breach(results.xyxy[0], monitoring_zone)

            # Draw monitoring zone
            color = self.ALERT_COLOR if breach_detected else self.ZONE_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw detection boxes
            for *box, conf, cls in results.xyxy[0]:
                bx1, by1, bx2, by2 = map(int, box)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame, f'{results.names[int(cls)]}: {conf:.2f}',
                          (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show status and breach count
            cv2.putText(frame, f"Status: {'BREACH' if breach_detected else 'Secure'}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.TEXT_COLOR, 1)
            cv2.putText(frame, f"Total Breaches: {self.breach_count}", 
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, self.TEXT_COLOR, 1)

            # Show frame
            cv2.imshow('Proximity Monitor', frame)

            # Show alert if breach detected
            if breach_detected:
                self.show_alert()

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

def main():
    monitor = ProximityMonitor()
    monitor.run()

if __name__ == "__main__":
    main()