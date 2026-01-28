import torch
import torch.utils.data

import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, pdb, math, python_speech_features
import numpy as np
from scipy import signal
from shutil import rmtree
from pathlib import Path
import tempfile
import copy
import traceback
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
            try:
                output_folder_name = Path(videoPath).with_suffix("").name
                video_args.savePath = str(Path(output_wave_dir) / output_folder_name)
                video_args.videoFilePath = videoPath
                main(video_args, args)
            except Exception as e:
                print(f"Failed to process {videoPath}. The exception was:")
                traceback.print_exc()



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
    full_audio = full_audio.astype(np.float32) / full_audio.max()

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
    allTracks = []  # list of [track1: dict, track2: dict] for each scene
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= video_args.minTrack:  # Discard the shot frames less than minTrack frames
            allTracks.append(track_shot_theskindeep(video_args, faces[shot[0].frame_num:shot[1].frame_num], shot[0].frame_num))
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" % sum(len(x) for x in allTracks))

    # Smooth tracks and convert to x_center/y_center/size format
    allTracksSmooth = smooth_tracks(allTracks)

    # Face clips cropping - returns tensors in memory
    t1 = time.time()
    vidTracks = crop_video(video_args, allTracksSmooth, decoder, full_audio)
    print(f'{time.time() - t1} seconds: cropping')
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop completed \r\n")

    # AVSE - pass tensors directly instead of reading from disk
    t1 = time.time()
    est_sources = evaluate_network(vidTracks, video_args, args)
    # list of np arrays, shape: [speaker_no, len(audio_segment)]
    est_sources = sum((list(x) for x in est_sources), start=[])
    # list of np arrays, shape: [len(audio_segment)]
    print(f'{time.time() - t1} seconds: speech separation forward')

    # Normalize the outputs by "max amplitude of speech" (without outliers)
    original_speech_max = np.percentile(np.abs(full_audio), 95)
    predicted_speech_max = np.percentile(np.abs(np.concatenate(est_sources)), 95)
    for audio in est_sources:
        audio *= original_speech_max / predicted_speech_max

    # Finally, save estimated separated audios
    audio_left = np.concatenate(est_sources[::2])
    audio_right = np.concatenate(est_sources[1::2])
    sf.write(video_args.savePath + f"/left.wav", audio_left, 16000)
    sf.write(video_args.savePath + f"/right.wav", audio_right, 16000)

    # Uncomment to save the detected face clips (audio+video) for each scene
    # t1 = time.time()
    # os.makedirs(video_args.pycropPath, exist_ok=True)

    # # Save estimated audio sources
    # for idx, audio in enumerate(est_sources):
    #     sf.write(video_args.pycropPath + f"/est_{idx:04}.wav", audio, 16000)

    # # Save cropped face videos to disk
    # for idx, track in enumerate(vidTracks):
    #     video_tensors = track['video_tensors']  # list of [n_frames, 1, 224, 224], torch.uint8
    #     for j, video_tensor in enumerate(video_tensors):
    #         out_idx = idx*2 + j
    #         video_tensor = torch.cat([video_tensor] * 3, dim=1)
    #         encoder = torchcodec.encoders.VideoEncoder(video_tensor, frame_rate=25.0)
    #         orig_path = os.path.join(video_args.pycropPath, f'orig_{out_idx:04}.mp4')
    #         encoder.to_file(orig_path)

    #         # Combine with estimated audio
    #         est_audio_path = os.path.join(video_args.pycropPath, f'est_{out_idx:04}.wav')
    #         est_video_path = os.path.join(video_args.pycropPath, f'est_{out_idx:04}.mp4')
    #         command = f"ffmpeg -y -hide_banner -i {orig_path} -i {est_audio_path} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {est_video_path} -loglevel warning"
    #         subprocess.call(command, shell=True, stdout=None)

    #         # Clean up temporary files
    #         os.remove(orig_path)

    # print(f'{time.time() - t1} seconds: saving output videos')

    # Visualization (optional)
    t1 = time.time()
    # visualization(vidTracks, est_sources, video_args, decoder)
    print(f'{time.time() - t1} seconds: visualization')

    # Clean up
    rmtree(video_args.pyworkPath)
    rmtree(video_args.pyaviPath)


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

def smooth_tracks(all_tracks):
    all_tracks_smooth = []

    for scene_tracks in all_tracks:
        # Smooth bboxes and convert them to a different convention
        scene_tracks_converted = []

        for track in scene_tracks:
            dets = {'x': [], 'y': [], 's': [], 'frame': track['frame']}

            for det in track['bbox']: # Read the tracks
                dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2)
                dets['y'].append((det[1]+det[3])/2) # crop center y
                dets['x'].append((det[0]+det[2])/2) # crop center x
            dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
            dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
            dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

            scene_tracks_converted.append(dets)

        all_tracks_smooth.append(scene_tracks_converted)

    return all_tracks_smooth

def crop_video(video_args, all_tracks, decoder, full_audio, batch_size=128):
    # Crop the face clips and return as RGB tensors [n_frames, 3, 224, 224]
    for scene_tracks in all_tracks:
        for i in range(0, len(scene_tracks)):
            assert np.all(scene_tracks[i]['frame'] == scene_tracks[0]['frame']), \
                f"This function only works with sets of face tracks defined on same frames " \
                f"in the scene"
            assert np.all(np.diff(scene_tracks[i]['frame']) == 1), \
                f"This function only works with continuous face tracks"
            assert all(len(scene_tracks) == len(all_tracks[0]) for scene_tracks in all_tracks), \
                f"This function only works with same number of face tracks in every scene"

    # Combine tracks for performance
    fields = 'x', 'y', 's'
    num_tracks = len(all_tracks[0])
    all_tracks_combined = [
        {
                field: np.concatenate([scene_tracks[track_idx][field] for scene_tracks in all_tracks])
                for field in fields
        } for track_idx in range(num_tracks)
    ]

    # Extract and crop face frames
    cropped_frames_all = [torch.empty((len(all_tracks_combined[0]['x']), 1, 224, 224), dtype=torch.uint8) \
        for _ in all_tracks_combined]

    # Create dataset and dataloader for concurrent video decoding
    dataset = VideoFrameDataset(decoder)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=video_args.nDataLoaderThread,
        pin_memory=True,
    )
    device = 'cuda'

    # Create sampling grid for 224x224 output
    out_size = 224
    grid_y_template, grid_x_template = torch.meshgrid(
        torch.linspace(0, 1, out_size, device=device),
        torch.linspace(0, 1, out_size, device=device),
        indexing='ij'
    )

    current_chunk_start = 0
    for frame_batch in dataloader:
        frame_batch = frame_batch['frame'].to(device).float()  # torch.float32, 0..1, [b, 3, h, w]
        # Convert to grayscale - [b, 1, h, w]
        frame_batch = \
            0.299 * frame_batch[..., 0, :, :] + \
            0.587 * frame_batch[..., 1, :, :] + \
            0.114 * frame_batch[..., 2, :, :]
        frame_batch = frame_batch.unsqueeze(1)

        # Get input dimensions
        curr_batch_size, _, H, W = frame_batch.shape
        batch_end = current_chunk_start + curr_batch_size

        for track_idx, dets in enumerate(all_tracks_combined):
            # Get batch size and extract crop parameters for this batch

            # Extract crop parameters (center x, center y, half-size)
            crop_x = torch.tensor(dets['x'][current_chunk_start:batch_end], dtype=torch.float32, device=device)
            crop_y = torch.tensor(dets['y'][current_chunk_start:batch_end], dtype=torch.float32, device=device)
            # Detection box half-size
            bs = torch.tensor(dets['s'][current_chunk_start:batch_end], dtype=torch.float32, device=device)
            cs = video_args.cropScale

            # Original crop logic:
            # Y: from (my - bs) to (my + bs*(1+2*cs)), size = 2*bs*(1+cs)
            # X: from (mx - bs*(1+cs)) to (mx + bs*(1+cs)), size = 2*bs*(1+cs)
            # The Y crop is asymmetric (more below center), X is symmetric

            # For symmetric grid_sample, we need:
            # - Half-size of crop region: bs*(1+cs)
            # - Adjusted Y center to account for asymmetry: my + bs*cs
            # - X center remains: mx

            crop_half_size = bs * (1 + cs)  # Half-size: bs*(1+cs)
            crop_y_center = crop_y + bs * cs  # Adjust Y center for asymmetry
            crop_x_center = crop_x  # X is symmetric

            # Expand to batch dimension [curr_batch_size, 224, 224]
            grid_x = grid_x_template.unsqueeze(0).expand(curr_batch_size, -1, -1)
            grid_y = grid_y_template.unsqueeze(0).expand(curr_batch_size, -1, -1)

            # Compute crop regions in pixel coordinates
            crop_x_min = (crop_x_center - crop_half_size).view(-1, 1, 1)
            crop_x_max = (crop_x_center + crop_half_size).view(-1, 1, 1)
            crop_y_min = (crop_y_center - crop_half_size).view(-1, 1, 1)
            crop_y_max = (crop_y_center + crop_half_size).view(-1, 1, 1)

            # Map from [0, 1] grid to crop region in pixel coordinates
            sample_x = crop_x_min + grid_x * (crop_x_max - crop_x_min)
            sample_y = crop_y_min + grid_y * (crop_y_max - crop_y_min)

            # Normalize to [-1, 1] for grid_sample
            sample_x_norm = (sample_x / (W - 1)) * 2 - 1
            sample_y_norm = (sample_y / (H - 1)) * 2 - 1

            # Create grid [curr_batch_size, 224, 224, 2] with [x, y] in last dimension
            grid = torch.stack([sample_x_norm, sample_y_norm], dim=-1)

            # Apply grid_sample (requires float input)
            batch_cropped = torch.nn.functional.grid_sample(
                frame_batch,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )

            batch_cropped = batch_cropped.clip(0, 255).round().byte()
            cropped_frames_all[track_idx][current_chunk_start:batch_end][:] = batch_cropped

        current_chunk_start += curr_batch_size

    # Re-combine frames to match scene cuts (i.e. splits in original all_tracks)
    retval = []

    for scene_tracks in all_tracks:
        start_frame_idx = scene_tracks[0]['frame'][0].item()
        num_frames = len(scene_tracks[0]['frame'])

        cropped_frames_scene = [
            cropped_frames_all_oneperson[start_frame_idx:start_frame_idx+num_frames]
            for cropped_frames_all_oneperson in cropped_frames_all
        ]

        # Extract audio segment by slicing the full audio array
        # Video is at 25 fps, audio is at 16000 Hz
        audio_sample_rate = 16000
        video_fps = 25
        start_sample = int((start_frame_idx / video_fps) * audio_sample_rate)
        end_sample = int(((start_frame_idx + num_frames) / video_fps) * audio_sample_rate)
        audio = full_audio[start_sample:end_sample]

        retval.append({
            'tracks': scene_tracks,
            'video_tensors': cropped_frames_scene,  # list of [num_frames, 1, 224, 224], float32, 0..1, grayscale
            'audio': audio
        })

    return retval

def evaluate_network(vidTracks, video_args, args):
    # vidTracks: list of dicts with 'video_tensors', 'audio', 'tracks', 'proc_tracks'

    # this architecture only accepts paired videos
    est_sources = []
    for track in tqdm.tqdm(vidTracks):
        # Load audio
        audio = track['audio']

        # Load video
        frames = track['video_tensors'] # list of torch.uint8, [num_frames, 1, 224, 224]

        frames = torch.stack(frames).float() / 255.0 # [num_speakers, num_frames, 1, 224, 224]
        # Crop center - # [num_speakers, num_frames, 1, 112, 112]
        frames = frames[..., int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
        # Normalize for neural net
        frames = (frames - 0.4161) / 0.1688
        # Remove channel dim
        frames = frames.squeeze(2)

        length = int(audio.shape[0] / 16000 * 25)
        if frames.shape[1] < length:
            frames = torch.nn.functional.pad(
                frames, (0, int(length - frames.shape[1]), 0, 0, 0, 0), mode='replicate')

        audio = np.expand_dims(audio, axis=0)  # [1, T]
        visual = np.expand_dims(frames, axis=0)  # [1, num_speakers, num_frames, 112, 112]

        inputs = (audio, visual)

        est_source = decode_one_audio_AV_MossFormer2_TSE_16K(video_args.model, inputs, args)
        # shape: [num_speakers, T]

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
