import time
from read_wav import get_audio_data
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

# load dummy dataset and read audio files
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[0]["audio"]
sample = get_audio_data("./output.wav")
start_time = time.time()
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# generate token ids
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
# decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
print(f"Cost time: {time.time() - start_time}.")
