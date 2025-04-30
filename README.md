# AI Criticiser

*Because your videos aren't quite embarrassing enough on their own.*

## What is this nonsense?

AI Criticiser is a delightfully brutal tool that takes your perfectly innocent videos and transforms them into a montage of shame. It extracts those moments most worthy of ridicule, freezes them for dramatic effect, and overlays them with devastatingly understated criticism narrated in a voice that sounds suspiciously like it's judging your life choices.

## How it works (or doesn't)

1. Place your soon-to-be-regretted video in the same directory as this code and name it `source.mp4`
2. Run the script and wait for the magic of artificial judgment
3. Receive your `criticism_montage.mp4` and question your career choices

```bash
python main.py
```

The program will then proceed to:
1. Extract audio from your video with all the grace of a toddler removing wallpaper
2. Transcribe said audio, capturing every "um" and mispronunciation for posterity
3. Analyze the transcription for moments worthy of criticism (there will be many)
4. Generate voice narration that sounds like Stephen Fry after he's read your diary
5. Create a montage that will ensure you never show your face at the office Christmas party again

## Requirements

* Python 3.x (because we're not complete savages)
* FFmpeg (for video manipulation that's only slightly more sophisticated than stop-motion)
* OpenAI API key (to fund the AI that's judging you)
* Gemini API key (because one AI judging you wasn't quite enough)
* A thick skin and sense of humour (not included in `requirements.txt`)

## Environment Variables

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Or don't, and enjoy the cascade of authentication errors. Your choice, really.

## Dependencies

```
pip install -r requirements.txt
```

Though if we're being honest, installing the dependencies is the least of your problems right now.

## Technical Details

For those who insist on understanding how they're being mocked:

* Audio extraction and video editing: FFmpeg (because reinventing the wheel would be too kind)
* Transcription: OpenAI's Whisper model (it hears everything, even what you wish it wouldn't)
* Criticism generation: Google's Gemini model (trained on British comedy panel shows, presumably)
* Voice narration: OpenAI's TTS model (voice: "onyx" - selected for maximum condescension)

## Cache Management

The program caches transcriptions and criticism analyses to avoid unnecessary API calls. Not because we care about your API budget, mind you, but because even AI has better things to do than repeatedly analyze your content.

## Error Handling

Errors are handled with all the grace and elegance of a giraffe on roller skates. The program will attempt to recover from failures, but no promises. Much like your video content.

## Disclaimer

This tool was created for entertainment purposes. If you find yourself deeply offended by the criticism, perhaps consider that the AI might have a point. Just saying.

## License

Feel free to use, modify, and distribute this code as you see fit. Though if you're willingly sharing a tool that generates criticism of yourself, perhaps reconsider your life choices.

*"The only thing worse than being talked about is not being talked about, unless it's by this particular AI."* - Oscar Wilde, probably
