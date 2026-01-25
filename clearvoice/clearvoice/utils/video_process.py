import torch
import torch.utils.data

import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, pdb, math, python_speech_features
import numpy as np
from scipy import signal
from shutil import rmtree
from pathlib import Path
import tempfile
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score
import soundfile as sf
import torchcodec

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from ..models.av_mossformer2_tse.faceDetector.s3fd import S3FD

from .decode import decode_one_audio_AV_MossFormer2_TSE_16K, MAX_WAV_VALUE

import time


class VideoFrameDataset(torch.utils.data.Dataset):
    """Dataset for loading video frames from torchcodec decoder."""
    def __init__(self, decoder, frame_indices=None):
        self.decoder = decoder
        if frame_indices is None:
            # Get all frames
            self.frame_indices = list(range(decoder.metadata.num_frames))
        else:
            self.frame_indices = frame_indices

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        frame_idx = self.frame_indices[idx]
        # Get single frame: returns tensor of shape [3, H, W]
        frame = self.decoder.get_frame_at(frame_idx).data
        return {'frame': frame, 'frame_idx': frame_idx}


def process_tse(args, model, device, data_reader, output_wave_dir):
    video_args = args_param()
    video_args.model = model
    video_args.device = device
    video_args.sampling_rate = args.sampling_rate
    args.device = device
    assert args.sampling_rate == 16000
    with torch.no_grad():
        for videoPath in data_reader:  # Loop over all video samples
            output_folder_name = Path(videoPath).with_suffix("").name
            video_args.savePath = str(Path(output_wave_dir) / output_folder_name)
            video_args.videoFilePath = videoPath
            main(video_args, args)



def args_param():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--nDataLoaderThread',     type=int,   default=int(os.cpu_count() * 0.8),   help='Number of workers')
    parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
    parser.add_argument('--minTrack',              type=int,   default=0,    help='Number of min frames for each shot')
    parser.add_argument('--numFailedDet',          type=int,   default=10**6,help='Number of missed detections allowed before tracking is stopped')
    parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
    parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')
    parser.add_argument('--start',                 type=int, default=0,      help='The start time of the video')
    parser.add_argument('--duration',              type=int, default=0,      help='The duration of the video, when set as 0, will extract the whole video')
    video_args = parser.parse_args()
    return video_args


# Main function
def main(video_args, args):
    print(f"Processing video {video_args.videoFilePath}")

    # Initialization
    video_args.pyaviPath = os.path.join(video_args.savePath, 'py_video')
    video_args.pyframesPath = os.path.join(video_args.savePath, 'pyframes')
    video_args.pyworkPath = os.path.join(video_args.savePath, 'pywork')
    video_args.pycropPath = os.path.join(video_args.savePath, 'py_faceTracks')
    if os.path.exists(video_args.savePath):
        rmtree(video_args.savePath)
    os.makedirs(video_args.pyaviPath, exist_ok=True)  # The path for the input video, input audio, output video
    os.makedirs(video_args.pyworkPath, exist_ok=True)  # Save the results in this process by the pckl method
    os.makedirs(video_args.pycropPath, exist_ok=True)  # Save the detected face clips (audio+video) in this process

    # If the video is too large, downscale first
    t1 = time.time()
    video_capture_tmp = cv2.VideoCapture(video_args.videoFilePath)
    video_original_h = video_capture_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_original_w = video_capture_tmp.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_capture_tmp.release()
    if video_original_w > 1280:
        downscaled_video_path = Path(tempfile.gettempdir()) / \
            Path(video_args.videoFilePath).with_suffix(".tmp.mp4").name
        command = \
            f"ffmpeg -y -hide_banner -i {video_args.videoFilePath} " \
            f"-threads {video_args.nDataLoaderThread} -c:a copy -vf scale=1280:-2 " \
            f"{downscaled_video_path} -loglevel warning"
        print(command)
        subprocess.call(command, shell=True, stdout=None)
        video_args.videoFilePath = str(downscaled_video_path)
    print(f'{time.time() - t1} seconds: downscaling to HD')

    # Load video into memory using torchcodec
    t1 = time.time()
    with open(video_args.videoFilePath, 'rb') as f:
        video_raw = f.read()
    decoder = torchcodec.decoders.VideoDecoder(video_raw)
    num_frames = decoder.metadata.num_frames
    print(f'{time.time() - t1} seconds: video loading into memory')

    # Extract audio
    t1 = time.time()
    video_args.audioFilePath = os.path.join(video_args.pyaviPath, 'audio.wav')
    command = ("ffmpeg -y -hide_banner -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel warning" % \
        (video_args.videoFilePath, video_args.nDataLoaderThread, video_args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)

    # Load audio into memory
    _, full_audio = wavfile.read(video_args.audioFilePath)

    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" % (video_args.audioFilePath))
    print(f'{time.time() - t1} seconds: audio extraction and loading')

    # Scene detection for the video frames
    t1 = time.time()
    scene = scene_detect(video_args)
    # scene = [(FrameTimecode(0, fps=25.0), FrameTimecode(num_frames - 1, fps=25.0))]
    print(f'{time.time() - t1} seconds: scene detection')
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" % (video_args.pyworkPath))

    # Face detection for the video frames with batched inference
    t1 = time.time()
    faces = detect_faces(video_args, decoder, batch_size=256)
    print(f'{time.time() - t1} seconds: face detection')
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" % (video_args.pyworkPath))

    # Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= video_args.minTrack:  # Discard the shot frames less than minTrack frames
            allTracks.extend(track_shot_theskindeep(video_args, faces[shot[0].frame_num:shot[1].frame_num], shot[0].frame_num))
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" % len(allTracks))

    # Face clips cropping - returns tensors in memory
    t1 = time.time()
    for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
        vidTracks.append(crop_video(video_args, track, decoder, full_audio, ii))
    savePath = os.path.join(video_args.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(vidTracks, fil)
    print(f'{time.time() - t1} seconds: cropping')
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop completed \r\n")

    # AVSE - pass tensors directly instead of reading from disk
    t1 = time.time()
    est_sources = evaluate_network(vidTracks, video_args, args)
    print(f'{time.time() - t1} seconds: speech separation forward')

    # Normalize the outputs by "max amplitude of speech" (without outliers)
    full_audio_fp32 = full_audio.astype(np.float32) / MAX_WAV_VALUE
    original_speech_max = np.percentile(np.abs(full_audio_fp32), 95)
    predicted_speech_max = np.percentile(np.abs(np.concatenate(est_sources)), 95)
    for audio in est_sources:
        audio *= original_speech_max / predicted_speech_max

    # Save estimated audio sources
    for idx, audio in enumerate(est_sources):
        sf.write(video_args.pycropPath + f"/est_{idx:04}.wav", audio, 16000)

    audio_left = np.concatenate(est_sources[::2])
    audio_right = np.concatenate(est_sources[1::2])
    sf.write(video_args.savePath + f"/audio_left.wav", audio_left, 16000)
    sf.write(video_args.savePath + f"/audio_right.wav", audio_right, 16000)

    # # Save cropped face videos to disk using torchcodec
    # t1 = time.time()
    # for idx, track in enumerate(vidTracks):
    #     video_tensor = track['video_tensor']  # [n_frames, 3, 224, 224], uint8
    #     encoder = torchcodec.encoders.VideoEncoder(video_tensor, frame_rate=25.0)
    #     orig_path = os.path.join(video_args.pycropPath, f'orig_{idx}.mp4')
    #     encoder.to_file(orig_path)

    #     # Combine with estimated audio
    #     est_audio_path = os.path.join(video_args.pycropPath, f'est_{idx:04}.wav')
    #     est_video_path = os.path.join(video_args.pycropPath, f'est_{idx:04}.mp4')
    #     command = f"ffmpeg -y -hide_banner -i {orig_path} -i {est_audio_path} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {est_video_path} -loglevel warning"
    #     subprocess.call(command, shell=True, stdout=None)

    #     # Clean up temporary files
    #     os.remove(orig_path)

    # print(f'{time.time() - t1} seconds: saving output videos')

    # Visualization (optional)
    t1 = time.time()
    # visualization(vidTracks, est_sources, video_args, decoder)
    print(f'{time.time() - t1} seconds: visualization')

    # Clean up
    rmtree(video_args.pyworkPath)




def scene_detect(video_args):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([video_args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source = videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(video_args.pyworkPath, 'scene.pckl')
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
    with open(savePath, 'wb') as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write('%s - scenes detected %d\n'%(video_args.videoFilePath, len(sceneList)))
    return sceneList

def detect_faces(video_args, decoder, batch_size=32):
    # GPU: Face detection with batched inference, output is the list contains the face location and score in this frame
    DET = S3FD(device=video_args.device)

    # Create dataset and dataloader for concurrent video decoding
    dataset = VideoFrameDataset(decoder)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=video_args.nDataLoaderThread,
        pin_memory=True
    )

    dets = []
    num_frames = len(dataset)

    # Initialize dets list with empty lists for each frame
    for _ in range(num_frames):
        dets.append([])

    # t0 = time.time()
    for batch_idx, batch in enumerate(dataloader):
        # print(f'{time.time() - t0} seconds: waiting on the batch')
        frames = batch['frame']  # [B, 3, H, W], torch tensors
        frame_indices = batch['frame_idx']  # [B]

        # Process entire batch through S3FD at once
        batch_bboxes = DET.detect_faces_batch(frames, conf_th=0.9, scales=[video_args.facedetScale])

        # Store results
        for i, (fidx, bboxes) in enumerate(zip(frame_indices, batch_bboxes)):
            fidx = fidx.item()
            for bbox in bboxes:
                dets[fidx].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})

        if batch_idx % 10 == 0:
            processed = min((batch_idx + 1) * batch_size, num_frames)
            print('%s-%05d/%05d' % (video_args.videoFilePath, processed, num_frames))
        # t0 = time.time()

    sys.stderr.write('\n')
    savePath = os.path.join(video_args.pyworkPath, 'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def track_shot_theskindeep(video_args, sceneFaces, first_frame_idx):
    w, h = 1280, 720

    tracks = [[], []]
    
    for frameIdx, frameFaces in enumerate(sceneFaces):
        while len(frameFaces) < 2:
            frameFaces.append({'frame': frameIdx, 'bbox': [w/2-100, h/2-100, w/2+100, h/2+100, 1.0]})

        x_center_fn = lambda face: (face['bbox'][0] + face['bbox'][2]) / 2
        face_left = min(frameFaces, key=x_center_fn)
        face_right = max(frameFaces, key=x_center_fn)

        tracks[0].append(face_left)
        tracks[1].append(face_right)

    def track_to_numpy(track):
        assert len(track) == len(sceneFaces)
        frameI      = np.arange(first_frame_idx, first_frame_idx + len(sceneFaces))
        bboxesI     = np.array([face['bbox'][:4] for face in track])
        return {'frame': frameI, 'bbox': bboxesI}

    tracks = [track_to_numpy(track) for track in tracks]
    # import pickle
    # with open('tmp.pkl', 'wb') as f: pickle.dump(tracks, f)
    return tracks

def track_shot(video_args, sceneFaces):
    # CPU: Face tracking
    iouThres  = 0.1     # Minimum IOU between consecutive face detections
    tracks    = []
    while True:
        track     = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= video_args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        # found the face in this frame that will continue the track
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > video_args.minTrack:
            frameNum    = np.array([ f['frame'] for f in track ])
            bboxes      = np.array([np.array(f['bbox']) for f in track])
            frameI      = np.arange(0, len(sceneFaces))
            bboxesI    = []
            for ij in range(0,4):
                interpfn  = interp1d(frameNum, bboxes[:,ij], fill_value=(track[0]['bbox'], track[1]['bbox']))
                bboxesI.append(interpfn(frameI))
            bboxesI  = np.stack(bboxesI, axis=1)
            if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > video_args.minFaceSize:
                tracks.append({'frame':frameI,'bbox':bboxesI})

    return tracks

def crop_video(video_args, track, decoder, full_audio, crop_idx):
    # CPU: crop the face clips and return as RGB tensors [n_frames, 3, 224, 224]
    dets = {'x':[], 'y':[], 's':[]}
    for det in track['bbox']: # Read the tracks
        dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2)
        dets['y'].append((det[1]+det[3])/2) # crop center y
        dets['x'].append((det[0]+det[2])/2) # crop center x
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

    # Extract and crop face frames
    cropped_frames = []
    for fidx, frame_idx in enumerate(track['frame']):
        cs  = video_args.cropScale
        bs  = dets['s'][fidx]   # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

        # Get frame from decoder
        frame = decoder.get_frame_at(int(frame_idx)).data  # [3, H, W], uint8, RGB
        frame = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

        # Pad and crop
        frame_padded = np.pad(frame, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my  = dets['y'][fidx] + bsi  # BBox center Y
        mx  = dets['x'][fidx] + bsi  # BBox center X
        face = frame_padded[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        face_resized = cv2.resize(face, (224, 224))  # [224, 224, 3]
        cropped_frames.append(face_resized)

    # Convert to tensor: [n_frames, 3, 224, 224]
    cropped_frames = np.stack(cropped_frames, axis=0)  # [n_frames, 224, 224, 3]
    cropped_frames_tensor = torch.from_numpy(cropped_frames).permute(0, 3, 1, 2)  # [n_frames, 3, 224, 224]

    # Extract audio segment by slicing the full audio array
    # Video is at 25 fps, audio is at 16000 Hz
    audio_sample_rate = 16000
    video_fps = 25
    start_sample = int((track['frame'][0] / video_fps) * audio_sample_rate)
    end_sample = int(((track['frame'][-1] + 1) / video_fps) * audio_sample_rate)
    audio = full_audio[start_sample:end_sample]

    return {
        'track': track,
        'proc_track': dets,
        'video_tensor': cropped_frames_tensor,  # [n_frames, 3, 224, 224], uint8, RGB
        'audio': audio
    }


def evaluate_network(vidTracks, video_args, args):
    # vidTracks: list of dicts with 'video_tensor', 'audio', 'track', 'proc_track'

    # this architecture only accepts paired videos
    if args.network == "AV_TFGridNet_ISAM_TSE_16K":
        assert len(vidTracks) % 2 == 0, \
            f"For TFGridNet videos have to come in pairs, but got {len(vidTracks)} videos"

        tracks_new = []
        for i in range(0, len(vidTracks), 2):
            tracks_new.append((vidTracks[i], vidTracks[i+1]))
            tracks_new.append((vidTracks[i+1], vidTracks[i]))
    else:
        tracks_new = [(track, None) for track in vidTracks]

    est_sources = []
    for track, track_second in tqdm.tqdm(tracks_new, total=len(tracks_new)):

        # Load audio
        audio = track['audio']

        # Process video tensor: [n_frames, 3, 224, 224], uint8, RGB
        video_tensor = track['video_tensor']  # torch.Tensor

        # Convert to grayscale and crop to center 112x112
        videoFeature = []
        for frame_idx in range(video_tensor.shape[0]):
            # Get frame: [3, 224, 224]
            frame = video_tensor[frame_idx]  # uint8, RGB
            # Convert to grayscale: 0.299*R + 0.587*G + 0.114*B
            frame_gray = (0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]).numpy()  # [224, 224]
            # Crop center 112x112
            face = frame_gray[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
            videoFeature.append(face)

        visual = np.array(videoFeature) / 255.0
        visual = (visual - 0.4161) / 0.1688

        length = int(audio.shape[0] / 16000 * 25)
        if visual.shape[0] < length:
            visual = np.pad(visual, ((0, int(length - visual.shape[0])), (0, 0), (0, 0)), mode='edge')

        visual = np.expand_dims(visual, axis=0)  # [1, T, 112, 112]

        # if architecture needs to process face tracks in pairs:
        if track_second is not None:
            # Process the second video tensor
            video_tensor_2 = track_second['video_tensor']

            videoFeature = []
            for frame_idx in range(video_tensor_2.shape[0]):
                frame = video_tensor_2[frame_idx]
                frame_gray = (0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]).numpy()
                face = frame_gray[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                videoFeature.append(face)

            videoFeature = np.array(videoFeature) / 255.0
            videoFeature = (videoFeature - 0.4161) / 0.1688

            length = int(audio.shape[0] / 16000 * 25)
            if videoFeature.shape[0] < length:
                videoFeature = np.pad(videoFeature, ((0, int(length - videoFeature.shape[0])), (0, 0), (0, 0)), mode='edge')

            videoFeature = np.expand_dims(videoFeature, axis=0)

            visual = np.concatenate([visual, videoFeature])[None]  # [1, 2, T, 112, 112]

        audio = audio.astype('float32') / MAX_WAV_VALUE
        audio = np.expand_dims(audio, axis=0)

        inputs = (audio, visual)

        est_source = decode_one_audio_AV_MossFormer2_TSE_16K(video_args.model, inputs, args)
        # print('Audio output in evaluate_network() for one track vs audio input:', est_source.shape, audio.shape)

        est_sources.append(est_source)

    return est_sources

def visualization(tracks, est_sources, video_args, decoder):
    # CPU: visualize the result for video format
    num_frames = decoder.metadata.num_frames

    for tidx, track in enumerate(tracks):
        faces = [[] for i in range(num_frames)]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            faces[frame].append({'track': tidx, 's': track['proc_track']['s'][fidx], 'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx]})

        # Get first frame to determine dimensions
        first_frame = decoder.get_frame_at(0).data  # [3, H, W], uint8, RGB
        fh, fw = first_frame.shape[1], first_frame.shape[2]

        vOut = cv2.VideoWriter(os.path.join(video_args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw, fh))
        for fidx in tqdm.tqdm(range(num_frames), total=num_frames):
            # Get frame from decoder
            frame = decoder.get_frame_at(fidx).data  # [3, H, W], uint8, RGB
            image = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, 3], RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

            for face in faces[fidx]:
                cv2.rectangle(image, (int(face['x'] - face['s']), int(face['y'] - face['s'])), (int(face['x'] + face['s']), int(face['y'] + face['s'])), (0, 255, 0), 10)
            vOut.write(image)
        vOut.release()

        command = ("ffmpeg -y -hide_banner -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel warning" % \
            (os.path.join(video_args.pyaviPath, 'video_only.avi'), (video_args.pycropPath + '/est_%s.wav' % tidx), \
            video_args.nDataLoaderThread, os.path.join(video_args.pyaviPath, 'video_out_%s.avi' % tidx)))
        output = subprocess.call(command, shell=True, stdout=None)

        command = "ffmpeg -y -hide_banner -i %s %s -loglevel warning;" % (
                    os.path.join(video_args.pyaviPath, 'video_out_%s.avi' % tidx),
                    os.path.join(video_args.pyaviPath, 'video_est_%s.mp4' % tidx)
                )
        output = subprocess.call(command, shell=True, stdout=None)

    command = f"rm {os.path.join(video_args.pyaviPath, 'audio.wav')} ;"
    output = subprocess.call(command, shell=True, stdout=None)
