import whisper

import whisperx
import torch
from simple_diarizer.diarizer import Diarizer
import re
import pandas as pd

from tqdm import tqdm
import time
import datetime

sentence_ending_punctuations = ".?!;"

def transcribing(model_name="meidum", audiofile="cleared_audiofile.wav"):
    print("Transcription", datetime.datetime.now())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stt_model = whisper.load_model(model_name, device=device)

    result = stt_model.transcribe(audiofile,
                                  language="ru", 
                                  word_timestamps=True,
                                  condition_on_previous_text=False,
                                #   patience=2,
                                  beam_size=5,
                                #   temperature=0.1,
                                  verbose=True
                                  )
    
    # clear gpu vram
    del stt_model
    torch.cuda.empty_cache()

    return result #result['segments']


def aligning_transcribed_text(transcription_result):
    print("Aligning transcribed text", datetime.datetime.now())

    word_timestamps = []
    for segment in tqdm(transcription_result):
        for word in segment["words"]:
            word_timestamps.append({"text": word['word'], "start": word['start'], "end": word['end']})

    return word_timestamps


def diarization(word_timestamps, audiofile, audio_rttm="audio.rttm", num_speakers=None):
    start_time = time.time()
    print("Diarization", datetime.datetime.now())

    diar_model = Diarizer(
        embed_model="xvec",
        cluster_method="sc"
    )

    print(diar_model.run_opts)

    diar_model.diarize(
        audiofile,
        num_speakers=num_speakers,
        threshold=None if num_speakers is not None else 1e-1,
        outfile=audio_rttm
    )

    speaker_ts = []
    with open(audio_rttm, "r", encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line_list = line.split(" ")
            start = int(float(line_list[3]) * 1000)
            end = start + int(float(line_list[4]) * 1000)
            speaker_ts.append([start, end, int(line_list[7].split("_")[-1])])

    word_speaker_mapping = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    print("End of diarization", time.time() - start_time)
    return word_speaker_mapping, speaker_ts


def reailign_words(words_mapping, speaker_timestamps):
    print("Realigning words to their true timestamps", datetime.datetime.now())

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict in words_mapping:
        word = word_dict["word"].strip().rstrip()
        if (
            word
            and word[-1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

    words_mapping = get_realigned_ws_mapping_with_punctuation(words_mapping, max_words_in_sentence=50)
    sentence_mapping = get_sentences_speaker_mapping(words_mapping, speaker_timestamps)

    return sentence_mapping


def get_word_ts_anchor(start, end, option="start"):
    if option == "end":
        return end
    elif option == "mid":
        return (start + end) / 2
    return start


def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wrd = (
            int(wrd_dict["start"] * 1000),
            int(wrd_dict["end"] * 1000),
            wrd_dict["text"],
        )
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
            if turn_idx == len(spk_ts) - 1:
                e = get_word_ts_anchor(ws, we, option="end")
        wrd_spk_mapping.append(
            {"word": wrd, "start_time": ws, "end_time": we, "speaker": sp}
        )
    return wrd_spk_mapping


def get_realigned_ws_mapping_with_punctuation(word_speaker_mapping, max_words_in_sentence=100):

    is_word_sentence_end = (
        lambda x: x >= 0
        and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)

    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        if (
            k < wsp_len - 1
            and speaker_list[k] != speaker_list[k + 1]
            and not is_word_sentence_end(k)
        ):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence
            )
            right_idx = (
                get_last_word_idx_of_sentence(
                    k, words_list, max_words_in_sentence - k + left_idx - 1
                )
                if left_idx > -1
                else -1
            )
            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speaker_list[left_idx : right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue

            speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (
                right_idx - left_idx + 1
            )
            k = right_idx

        k += 1

    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy()
        line_dict["speaker"] = speaker_list[k]
        realigned_list.append(line_dict)
        k += 1

    return realigned_list

def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    s, e, spk = spk_ts[0]
    prev_spk = spk

    snts = []
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk:
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk

    snts.append(snt)
    return snts


def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    left_idx = word_idx
    while (
        left_idx > 0
        and word_idx - left_idx < max_words
        and speaker_list[left_idx - 1] == speaker_list[left_idx]
        and not is_word_sentence_end(left_idx - 1)
    ):
        left_idx -= 1

    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1

def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    right_idx = word_idx
    while (
        right_idx < len(word_list)
        and right_idx - word_idx < max_words
        and not is_word_sentence_end(right_idx)
    ):
        right_idx += 1

    return (
        right_idx
        if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
        else -1
    )
