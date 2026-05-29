# final_pipeline_nomeroff.py
import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from lprnet_pruned import LPRNetPrunable

RU_CHARS = "0123456789АВЕКМНОРСТУХ"


class NomeroffANPR:
    def __init__(self, detector_weights='best.pt',
                 recognizer_weights='models/lprnet_nomeroff_pruned.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.detector = YOLO(detector_weights)
        self.detector.to(self.device)

        self.num_classes = len(RU_CHARS) + 1
        self.recognizer = LPRNetPrunable(num_classes=self.num_classes)
        self.recognizer.load_state_dict(torch.load(recognizer_weights, map_location=self.device))
        self.recognizer.to(self.device)
        self.recognizer.eval()

        self.idx_to_char = {i: c for i, c in enumerate(RU_CHARS)}
        self.fps_list = []

    def preprocess_plate(self, plate_img):
        resized = cv2.resize(plate_img, (94, 24))
        resized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def decode_ctc(self, preds):
        preds = preds.cpu().detach().numpy()
        pred_indices = np.argmax(preds, axis=2)

        prev_char = -1
        result_chars = []
        blank_idx = len(RU_CHARS)

        for idx in pred_indices[0]:
            if idx != blank_idx and idx != prev_char:
                if idx in self.idx_to_char:
                    result_chars.append(self.idx_to_char[idx])
            prev_char = idx

        return "".join(result_chars)

    def process_video(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error opening video")
            return

        print("Starting Nomeroff-based ANPR. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret: break

            t_start = time.time()

            results = self.detector(frame, verbose=False, conf=0.5)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_name = self.detector.names[int(box.cls[0])]
                    if cls_name == 'license_plate':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]

                        if crop.size == 0: continue

                        input_tensor = self.preprocess_plate(crop)
                        with torch.no_grad():
                            preds = self.recognizer(input_tensor)
                        text = self.decode_ctc(preds)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            t_end = time.time()
            fps = 1 / (t_end - t_start)
            self.fps_list.append(fps)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('ANPR Optimized', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.fps_list:
            avg_fps = sum(self.fps_list) / len(self.fps_list)
            print(f"\nMETRICS:")
            print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    pipeline = NomeroffANPR()
    pipeline.process_video(0)
