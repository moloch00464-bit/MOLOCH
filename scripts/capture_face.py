#!/usr/bin/env python3
import cv2, os, sys, time

RTSP = "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
FACES = "/home/molochzuhause/moloch/faces"

def main():
    name = sys.argv[1].lower() if len(sys.argv) > 1 else "markus"
    save_dir = os.path.join(FACES, name)
    os.makedirs(save_dir, exist_ok=True)
    
    print("Verbinde...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Kamera nicht erreichbar!")
        return
    
    print(f"5 Bilder fuer {name} - verschiedene Winkel!\n")
    
    for i in range(5):
        print(f"Bild {i+1}/5 in: 3...", end=" ", flush=True)
        time.sleep(1)
        print("2...", end=" ", flush=True)
        time.sleep(1)
        print("1...", end=" ", flush=True)
        time.sleep(1)
        
        for _ in range(30):
            cap.grab()
        
        ret, frame = cap.read()
        if ret:
            f = os.path.join(save_dir, f"{name}_{i+1}.jpg")
            cv2.imwrite(f, frame)
            print(f"OK -> {f}")
        
        if i < 4:
            print("Anderer Winkel!\n")
            time.sleep(2)
    
    cap.release()
    print(f"\nFertig! {len(os.listdir(save_dir))} Bilder")

if __name__ == "__main__":
    main()
