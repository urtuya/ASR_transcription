import streamlit as st
import streamlit_ext as ste
st.set_page_config(layout="wide")

import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
import shutil

import pandas as pd
import ffmpeg
import demucs.separate
from asr_process import transcribing,\
                        aligning_transcribed_text,\
                        diarization,\
                        reailign_words


TEMP_DIR = "temp_out"
RESULT_DIR = os.path.join(TEMP_DIR, "result_dir")
CONVERTED = os.path.join(TEMP_DIR, "converted.wav")
AUDIO_CLEANED = os.path.join(TEMP_DIR, "htdemucs/converted", "vocals.wav")
AUDIO_RTTM = os.path.join(TEMP_DIR, "audio.rttm")

MODEL_NAME = "medium"


def save_diarization_results(data, filename:str, audio_name:str):

    speaker = [x['speaker'] for x in data]
    start = [x['start_time'] / 1000 for x in data]
    end = [x['end_time'] / 1000 for x in data]
    text = [x['text'] for x in data]

    result_df = \
    (
        pd.DataFrame()
        .assign(start = start,
                end = end,
                speaker = speaker,
                text = text)
    )
    result_df.to_excel(filename + '.xlsx', index=False)

    ste.download_button(label="Download excel",
                        data=result_df,
                        file_name=audio_name[:audio_name.rfind(".")]+".xlsx")
    return result_df


def save_tts_results(data, filename:str, audio_name:str):

    data_segments = data['segments']

    result_df = \
    (
        pd.DataFrame()
        .assign(start_segment = [round(x['start'], 2) for x in data_segments])
        .assign(end_segment = [round(x['end'], 2) for x in data_segments])
        .assign(text_segment = [x['text'] for x in data_segments])
    )

    result_df.to_excel(filename + '.xlsx', index=False)

    ste.download_button(label="Download text in timestamp segments",
                        data=result_df,
                        file_name=audio_name[:audio_name.rfind(".")]+".xlsx")
    ste.download_button(label="Download text in txt format",
                        data=data['text'],
                        file_name=audio_name[:audio_name.rfind(".")]+".txt")
    
    return result_df


def create_zip():
    dir_to_archive = os.getcwd()
    dir_to_archive = os.path.join(dir_to_archive, RESULT_DIR)

    archive_name = "result_archive"
    path_to_archive = os.path.join(TEMP_DIR, archive_name)

    shutil.make_archive(path_to_archive,
                        format='zip',
                        root_dir=dir_to_archive)

    with open(path_to_archive + '.zip', "rb") as fd:
        ste.download_button(label="Download archive",
                            data = fd,
                            file_name=archive_name+'.zip',
                            mime="application/zip")
    


def start_asr(path_to_files:list,
              num_of_speakers_lst:list,
              diarize=False,
              enable_stemming=False)->list:

    """
        Main function of asr processing:
        Parsing files
            1. cleaning file and saved cleaned
            2. stt-stranscription
            3. aligning text
            4. diarizing
            5. realligning
            6. saving result into dataframe and show 10 first rows in app
            7. save excel file to zip after all files finished
    """

    result_list = []

    for i in range(len(path_to_files)):

        start_time = time.time()

        audio_name = path_to_files[i]
        audiofile = os.path.join(TEMP_DIR, path_to_files[i])
        audio_path = AUDIO_CLEANED
        num_of_speakers = num_of_speakers_lst[i] if diarize else None
        path_to_save = os.path.join(RESULT_DIR, audiofile[:audiofile.rfind(".")])

        ## 1. Converting video into audio, file saved in CONVERTED

        with st.spinner(audio_name + ": Converting to audio"):
            out_tmp = ffmpeg.input(audiofile).output(
                                            CONVERTED,
                                            ar="16000",
                                            ac="1",
                                            acodec="pcm_s16le"
                                        )
            out_tmp.overwrite_output().run(quiet=True)
        st.success(audio_name + ": Converting to audio - Done")

        ## 2. Сleaning audio from extraneous sounds if needed

        if enable_stemming:
            with st.spinner("Cleaning audio"):
                demucs.separate.main(
                    ["--two-stems", "vocals", "-n", "htdemucs", CONVERTED, "-o", TEMP_DIR]
                )
            st.success("Cleaning audio - Done")
        else:
            audio_path = CONVERTED
        

        ## 3. STT
        with st.spinner("STT transcription"):
            transcription_result = transcribing(model_name=MODEL_NAME,
                                                audiofile=audio_path)
        st.success("STT transcription - Done")

        ## 4. Diarization if needed
        if diarize:

            ## 5. Aligning transcribed text
            with st.spinner("Aligning text"):
                word_timestamps = aligning_transcribed_text(
                    transcription_result=transcription_result['segments'])
            st.success("Aligning text - Done")

            with st.spinner("Diarization"):
                words_mapping, speaker_timestamps = diarization(
                                                        word_timestamps=word_timestamps,
                                                        audiofile=audio_path,
                                                        audio_rttm=AUDIO_RTTM,
                                                        num_speakers=num_of_speakers)

                ## 6. Realigning using punctuation by whisper
                sentence_mapping = reailign_words(
                                            words_mapping=words_mapping,
                                            speaker_timestamps=speaker_timestamps)
            st.success("Diarization - Done")

            ## Create result dataframe
            result_df = save_diarization_results(sentence_mapping, filename=path_to_save, audio_name=audiofile)
            st.text(audio_name + ": Processing time - " + str(round((time.time() - start_time) / 60, 2)) + " min")
            
            
            # result_list.append(df_tmp.copy())

        else:
            ## Save TTS result into .txt file
            result_df = save_tts_results(transcription_result, filename=path_to_save, audio_name=audiofile)
            st.text(audio_name + ": Processing time - " + str(round((time.time() - start_time) / 60, 2)) + " min") # in minutes
        
        result_list.append(result_df)

    return result_list


def clear_temp_folder():


    try:
        for root, _, files in os.walk(TEMP_DIR):
            for f in files:
                os.remove(os.path.join(root, f))
        
        for root, _, files in os.walk(RESULT_DIR):
            for f in files:
                os.unlink(os.path.join(root, f))
    except OSError:
        pass


def web_introductory_text():
    """
        Introduction and usage text
    """

    st.sidebar.title("General info")
    st.sidebar.info(
        """
        - TODO
        - add confluence
        - add whisper hithub
        - add diar model github
        - add link discussion of solution faster-whisper + nemo diar
        - add credentials (?)
        """
    )

    st.title("ASR Speech-to-text")
    st.divider()

    st.header("Whisper transcription & Diarization")
    st.subheader("Usage")
    st.markdown(
        """
            To get STT transcription:
            1. Upload a video/audio
            2. Slide to Yes diarization slider to get STT-transcription and diarization if needed
            3. Download the result (.txt without diarization, excel for diarization)
        """
    )

    st.button("Refresh ", on_click=st.rerun)

    st.divider()


def save_uploadedfile(uploadedfile):
    with open(os.path.join(TEMP_DIR, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
        # return st.success("Saved File:{} to Data".format(uploadedfile.name))
    

def user_artifacts():
    """
        uploaded files: downloading the audio/video files
        slider: True or False - flag if diarization needed
    """

    if "upload_button" not in st.session_state:
        st.session_state.upload_button = False

    uploaded_files = st.file_uploader(
        label="Upload a file: audio/video",
        key="download_button",
        type=['mp4', 'mov', 'mkv', 'wmv', 'avi', 'mp3', 'wav', 'flac'],
        accept_multiple_files=True,
        disabled=st.session_state.upload_button
    )
    if st.session_state.get("download_button"):
        st.session_state.slider = False
    else:
        st.session_state.slider = True
    
    uploaded_files_names = [file.name for file in uploaded_files]
    for file in uploaded_files:
        save_uploadedfile(file)

    slider_clearing = st.toggle("Enable audio cleaning (may take a long time)",
                                disabled=st.session_state.slider)
    slider_diarization = st.toggle("Enable diarization", key="enabled_diar",
                                   disabled=st.session_state.slider)


    num_of_speakers = []
    if st.session_state.get("enabled_diar"):
        for file in uploaded_files_names:
            num_of_speakers.append(
                st.number_input("Number of speakers in " + file + "",
                                value=None,
                                key=file,
                                placeholder="Type a number",
                                min_value=1, max_value=12)
            )

    return uploaded_files_names, num_of_speakers, slider_clearing, slider_diarization


# def main(files, diar_flag=False):
#     """
#         Main program exec
#         Parsing files in a circle
#         1. cleaning file and saved cleaned
#         2. stt-stranscription
#         3. aligning text
#         4. diarizing
#         5. realligning
#         6. saving result into dataframe and show 10 first rows in app
#         7. save excel file to zip after all files finished
#         8. add button to download zipped result
#     """
    
#     return pd.DataFrame()



if __name__ == "__main__":

    web_introductory_text()
    clear_temp_folder()

    files, num_speakers_lst, clearing_flag, diar_flag = user_artifacts()

    if st.session_state.get("download_button"):
        st.session_state.transc_button = False
    else:
        st.session_state.transc_button = True

    transcr_button = st.button("Transcribe", key="transcribe_button", type="primary",
                               disabled=st.session_state.transc_button)

    st.divider()
    st.subheader("Processing")

    if st.session_state.get("transcribe_button"):
        st.session_state.transc_button = True
        # st.session_state.upload_button = True

        result_df_list = start_asr(path_to_files=files,
                                   num_of_speakers_lst=num_speakers_lst,
                                   diarize=diar_flag,
                                   enable_stemming=clearing_flag)

        st.divider()
        st.subheader("Transcription results (first 5 rows)")

        for i in range(len(files)):
            st.caption(files[i])
            st.dataframe(result_df_list[i].head(5))

        create_zip()
        clear_temp_folder()


