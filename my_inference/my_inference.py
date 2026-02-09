from clearvoice import ClearVoice
model_runner = ClearVoice(task='target_speaker_extraction', model_names=['AV_TFGridNet_ISAM_TSE_16K'])

import os

model_runner(
    input_path=os.environ.get("INPUT_PATH", "./inputs"),
    online_write=True,
    output_path="./outputs",
)

