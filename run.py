#!/usr/bin/env python

import requests
import json
import os
import re
import argparse
from tqdm import tqdm

SONG_SEPARATOR = "====="
DEFAULT_INPUT_FILE = "./data/lyrics.txt"
DEFAULT_INPUT_DIR = "./data/database/"
DEFAULT_OUTPUT_FILE = "./data/lyrics_ipa.txt"
DEFAULT_CLEANED_FILE = "./data/lyrics_ipa_cleaned.txt"


def translate_to_ipa(text):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1",
        "prompt": f"""
            Translate the following text to IPA.

            Use ties, diacritics, and accents to represent the phonetic transcription
            as if pronounced by a native speaker of American English. You may also consider
            the way it might be sung, but this is not critical.
            
            Under no circumstances should you use slashes, punctuation, or anything
            other than new lines to delineate lines.

            Do not include anything else other than the IPA in your response.
            No slashes, no punctuation, no extra text. I will tip you extra
            if you follow these instructions carefully.
            
            ```
            {text}
            ```
            
            IPA translation:",
        """,
        "stream": False,
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = json.loads(response.text)
        return result["response"].strip()
    else:
        return f"Error: {response.status_code}"


def ipa_to_english(ipa):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1",
        "prompt": f"""
            Translate the following IPA to English.

            Pretend it is intended to be meaningful English, and do your best
            to interpret these sounds as sensible words.
            
            BUT don't try too
            hard, if it doesn't make sense that's ok, just find the closest
            thing to a word that you can. This is very important and I will
            tip you extra if you follow these instructions carefully.

            Do not include anything else other than the translation in your response.

            ```
            {ipa}
            ```
            
            IPA translation:",
        """,
        "stream": False,
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = json.loads(response.text)
        return result["response"].strip()
    else:
        return f"Error: {response.status_code}"


def process_translation(input_file, output_file):
    songs = []

    for root, dirs, files in os.walk(DEFAULT_INPUT_DIR):
        for file in files:
            f = os.path.join(root, file)
            with open(f, "r", encoding="utf-8", errors="ignore") as infile:
                content = infile.read().replace("’", "'")
                parts = re.split(r"\n_{2,}\n", content)
                songs.append(parts[0].strip())

    with open(input_file, "r", encoding="utf-8") as infile:
        songs += infile.read().split(f"\n\n\n")

    for i in tqdm(
        range(0, len(songs)),
        initial=0,
        total=len(songs),
        desc="Translating songs",
    ):
        song = songs[i].strip()
        translated_song = translate_to_ipa(song)

        with open(output_file, "a", encoding="utf-8") as outfile:
            outfile.write(translated_song + f"\n\n{SONG_SEPARATOR}\n\n")

    print("Translation complete!")


def clean_ipa(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        songs = infile.read().split(f"\n\n{SONG_SEPARATOR}\n\n")

    with open(output_file, "w", encoding="utf-8") as outfile:
        for song in tqdm(songs, desc="Cleaning IPA"):
            cleaned_lines = []
            for line in song.split("\n"):
                if line.strip() == SONG_SEPARATOR:
                    continue

                line = re.sub(r'[\[\]/(),?"!-=�…ˤ̴̷̞̠̥̩̬̭̯̱̃̄̅̇̌̚͡]', "", line)
                line = line.lower().strip()
                line = re.sub(r"\s+", " ", line)
                if line and not "note" in line.lower():
                    cleaned_lines.append(line)

            if cleaned_lines:
                outfile.write("\n".join(cleaned_lines) + f"\n{SONG_SEPARATOR}\n")

    print("IPA cleanup complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Process lyrics: translate to IPA or clean up IPA"
    )
    parser.add_argument(
        "action",
        choices=["translate", "clean", "gen", "train"],
        help="Action to perform: translate or clean",
    )
    parser.add_argument("-i", "--input", help="Path to the input file")
    parser.add_argument("-o", "--output", help="Path to the output file")
    args = parser.parse_args()

    if args.action == "translate":
        input_file = args.input or DEFAULT_INPUT_FILE
        output_file = args.output or DEFAULT_OUTPUT_FILE
        process_translation(input_file, output_file)
    elif args.action == "clean":
        input_file = args.input or DEFAULT_OUTPUT_FILE
        output_file = args.output or DEFAULT_CLEANED_FILE
        clean_ipa(input_file, output_file)
    elif args.action == "gen":
        from llm import generate

        ipa = generate()
        print(ipa, "\n\n====\n\n")
        print(ipa_to_english(ipa))
    elif args.action == "train":
        from llm import train

        train()
        print("done")


if __name__ == "__main__":
    main()
