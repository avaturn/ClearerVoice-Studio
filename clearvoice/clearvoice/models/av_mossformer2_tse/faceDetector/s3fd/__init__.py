import time, os, sys, subprocess
import numpy as np
import cv2
import torch
from torchvision import transforms
from pathlib import Path

from .nets import S3FDNet
from .box_utils import nms_


img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
img_mean_torch = None  # Will be initialized lazily

class S3FD():

    def __init__(self, device='cuda'):

        tstamp = time.time()
        self.device = device

        PATH_WEIGHT = Path(__file__).parent / "sfd_face.pth"
        if os.path.isfile(PATH_WEIGHT) == False:
            Link = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
            cmd = "gdown --id %s -O %s"%(Link, PATH_WEIGHT)
            subprocess.call(cmd, shell=True, stdout=None)
        

        # print('[S3FD] loading with', self.device)
        self.net = S3FDNet(device=self.device).to(self.device)
        PATH = os.path.join(os.getcwd(), PATH_WEIGHT)
        state_dict = torch.load(PATH, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        # print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))
    
    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype('float32')
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h])

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            keep = nms_(bboxes, 0.1)
            bboxes = bboxes[keep]

        return bboxes

    def detect_faces_batch(self, images, conf_th=0.8, scales=[1]):
        """
        Detect faces using batched RGB torch.Tensor input.

        Args:
            images: RGB torch.Tensor of shape [B, C, H, W], uint8
            conf_th: Confidence threshold
            scales: List of scales for multi-scale detection

        Returns:
            batch_bboxes: list of numpy arrays, each of shape [N_i, 5] with [x1, y1, x2, y2, score]
        """
        global img_mean_torch
        if img_mean_torch is None:
            img_mean_torch = torch.from_numpy(img_mean).to(self.device)

        batch_size = images.shape[0]
        h, w = images.shape[2], images.shape[3]

        images = images.to(self.device).float()

        # Store bboxes for each image in the batch
        all_batch_bboxes = [[] for _ in range(batch_size)]

        with torch.no_grad():
            for s in scales:
                if s != 1.0:
                    # Resize using torch.nn.functional.interpolate
                    new_h, new_w = int(h * s), int(w * s)
                    scaled_imgs = torch.nn.functional.interpolate(
                        images,
                        size=(new_h, new_w),
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    # No resize needed
                    scaled_imgs = images  # [B, 3, H, W]

                # At this point: RGB [B, 3, H, W], float32
                # Subtract mean (applied to RGB channels)
                scaled_imgs -= img_mean_torch

                # Forward pass through network
                detections = self.net(scaled_imgs).cpu().numpy()  # [B, num_priors, num_classes, 5]

                scale = np.float32([w, h, w, h])

                # Process detections for each image in the batch
                for batch_idx in range(batch_size):
                    for i in range(detections.shape[1]):
                        j = 0
                        while detections[batch_idx, i, j, 0] > conf_th:
                            score = detections[batch_idx, i, j, 0]
                            pt = (detections[batch_idx, i, j, 1:] * scale)
                            bbox = [pt[0], pt[1], pt[2], pt[3], score]
                            all_batch_bboxes[batch_idx].append(bbox)
                            j += 1

        # Apply NMS to each image's detections separately
        result_bboxes = []
        for bboxes in all_batch_bboxes:
            if len(bboxes) == 0:
                result_bboxes.append(np.empty((0, 5), dtype=np.float32))
            else:
                bboxes = np.float32(bboxes)
                keep = nms_(bboxes, 0.1)
                result_bboxes.append(bboxes[keep])

        return result_bboxes
