from clearvoice import ClearVoice
model_runner = ClearVoice(task='target_speaker_extraction', model_names=['AV_TFGridNet_ISAM_TSE_16K'])

model_runner(
    input_path="./inputs",
    online_write=True,
    output_path="./outputs",
)

