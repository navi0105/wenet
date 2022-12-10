import re
import pandas as pd
import os
import argparse
from opencc import OpenCC
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("/hdd/dataset/commonvoice/cv-corpus-11.0-2022-09-21/zh-TW"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/commonvoice"))
    parser.add_argument("--split", type=str, required=True, nargs='+')

    args = parser.parse_args()

    return args

def data_prep(data_dir, clips_dir, output_dir, split):
    cc = OpenCC('t2s')

    for x in split:
        data = pd.read_csv(data_dir / f"{x}.tsv", sep='\t')

        audio_path_list = data['path'].values
        sentences_list = data['sentence'].values

        (output_dir / x).mkdir(parents=True, exist_ok=True)

        with open(output_dir / f"{x}/text", 'w') as f:
            for audio, text in zip(audio_path_list, sentences_list):
                utterance = os.path.splitext(audio)[0]
                text = re.sub(r'[^\w\s]', '', text)
                text = cc.convert(text)
                f.write(f"{utterance} {text}\n")

        with open(output_dir / f"{x}/wav.scp", 'w') as f:
            for audio in audio_path_list:
                audio = os.path.splitext(audio)
                utterance = audio[0]
                audio = clips_dir / (audio[0] + ".mp3")
                f.write(f"{utterance} {audio}\n")

def main():
    args = parse_args()
    data_dir = args.data_dir
    clips_dir = data_dir / "clips"
    output_dir = args.output_dir
    split = args.split

    data_prep(data_dir, clips_dir, output_dir, split)

if __name__ == "__main__":
    main()
