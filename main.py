import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import hashlib
import time
import random

# Load environment variables
load_dotenv()
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Global settings
MAX_RETRIES = 3  # Number of retries for failed operations
CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)


class ProgressBar:
    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        self._print_progress()

    def update(self, increment: int = 1, additional_info: str = ""):
        self.current += increment
        self._print_progress(additional_info)

    def _print_progress(self, additional_info: str = ""):
        progress = int(50 * self.current / self.total)
        percentage = (self.current / self.total) * 100
        print(
            f"\r{self.description}[{'=' * progress}{'-' * (50 - progress)}] {self.current}/{self.total} ({percentage:.1f}%){additional_info}",
            end="")
        if self.current >= self.total:
            print()  # New line when complete


class CacheManager:
    @staticmethod
    def get_cache_path(data: str, prefix: str = '') -> Path:
        hash_str = hashlib.md5(data.encode()).hexdigest()
        return CACHE_DIR / f"{prefix}_{hash_str}.json"

    @staticmethod
    def cleanup_old_cache():
        try:
            current_time = time.time()
            for cache_file in CACHE_DIR.glob('*.json'):
                if current_time - os.path.getmtime(cache_file) > 86400:  # 24 hours
                    try:
                        os.remove(cache_file)
                    except Exception:
                        pass
        except Exception:
            pass

    @staticmethod
    def read_cache(cache_path: Path) -> Optional[Any]:
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'r') as f:
                content = f.read()
                if len(content.strip()) < 3:
                    os.remove(cache_path)
                    return None
                return json.loads(content)
        except Exception:
            try:
                os.remove(cache_path)
            except Exception:
                pass
            return None

    @staticmethod
    def write_cache(cache_path: Path, data: Any):
        try:
            with open(cache_path, 'w') as f:
                f.write(json.dumps(data))
        except Exception:
            pass


class FFmpegHelper:
    @staticmethod
    def run_command(cmd: List[str]) -> Tuple[bool, str]:
        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            return process.returncode == 0, process.stderr
        except Exception as e:
            return False, str(e)

    @staticmethod
    def get_duration(file_path: Path) -> Optional[float]:
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(file_path)
            ]
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode == 0 and process.stdout.strip():
                return float(process.stdout.strip())
        except Exception:
            pass
        return None


class FileManager:
    @staticmethod
    def cleanup_files(paths: List[Path]):
        for path in paths:
            try:
                if path.is_file():
                    os.remove(path)
                elif path.is_dir():
                    for file in path.glob('*'):
                        try:
                            os.remove(file)
                        except Exception:
                            pass
                    try:
                        path.rmdir()
                    except Exception:
                        pass
            except Exception:
                pass


def extract_audio(video_path: Path) -> Optional[Path]:
    print("\nExtracting audio...")
    try:
        duration = FFmpegHelper.get_duration(video_path)
        if not duration:
            return None

        audio_path = video_path.with_suffix('.mp3')
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vn', '-acodec', 'libmp3lame',
            '-ac', '1', '-ar', '16000',
            '-q:a', '5',
            '-y', str(audio_path)
        ]

        success, error_msg = FFmpegHelper.run_command(cmd)
        if not success:
            print(f"Error extracting audio: {error_msg}")
            return None

        return audio_path if audio_path.exists() else None
    except Exception as e:
        print(f"Error during audio extraction: {e}")
        return None


def transcribe_audio(audio_path: Path) -> List[Dict]:
    if not audio_path.exists():
        return []

    # Check cache for transcription
    cache_key = f"transcription_{audio_path.stem}_{os.path.getsize(audio_path)}_{os.path.getmtime(audio_path)}"
    cache_path = CacheManager.get_cache_path(cache_key, 'transcription')
    cached_data = CacheManager.read_cache(cache_path)
    if cached_data:
        print("\nUsing cached transcription")
        return cached_data

    print("\nTranscribing audio...")
    for attempt in range(MAX_RETRIES):
        try:
            with open(audio_path, "rb") as audio_file:
                response = openai_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )

            words = []
            if isinstance(response, dict):
                words = (response.get('words', []) or
                         response.get('segments', []) or
                         response.get('word_segments', []))
            elif hasattr(response, 'words'):
                words = response.words
            elif hasattr(response, 'segments'):
                words = response.segments

            result = []
            for word in words:
                try:
                    if isinstance(word, dict):
                        start = float(word.get('start', word.get('start_time', 0)))
                        end = float(word.get('end', word.get('end_time', 0)))
                        text = word.get('word', word.get('text', ''))
                    else:
                        start = float(getattr(word, 'start', getattr(word, 'start_time', 0)))
                        end = float(getattr(word, 'end', getattr(word, 'end_time', 0)))
                        text = getattr(word, 'word', getattr(word, 'text', ''))

                    result.append({
                        'text': text,
                        'start': start,
                        'end': end
                    })
                except Exception:
                    continue

            # Cache successful transcription
            CacheManager.write_cache(cache_path, result)
            return result

        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                retry_delay = 5 * (attempt + 1)
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("All retries failed")
    return []


def analyze_criticism(text: str) -> List[Dict]:
    # Check cache
    cache_path = CacheManager.get_cache_path(text, 'criticism')
    cached_data = CacheManager.read_cache(cache_path)
    if cached_data:
        print("\nUsing cached criticism analysis")
        return cached_data

    print("\nAnalyzing for criticism...")
    prompt = f"""You are a brutal criticism analysis system. Analyze this text and find moments worthy of harsh criticism. Make your response humorously understated, but be absolutely ruthless and devastating in your criticism:

Text: {text}

Each criticism moment must be different and you must respond with at least 20 criticism moments.
Return your analysis as a JSON object with this exact format:
{{
    "criticism_moments": [
        {{
            "text": "<exact word-for-word quote of the part to criticize>",
            "reason": "<brutal, lengthy, devastating criticism of why this moment is terrible>",
            "score": <number 0-100 indicating how worthy of criticism this moment is>
        }}
    ]
}}

Consider elements like:
- Poor choices or decisions
- Lack of skill or competence
- Inconsistencies or contradictions
- Flawed logic or reasoning
- Cringe-worthy moments
- Missed opportunities
- Obvious mistakes

Be as harsh and brutal as possible in your criticism. Make the creator question their life choices.
Ensure the "text" field matches words EXACTLY as they appear in the original text."""

    # Add retry logic with exponential backoff for rate limits
    max_retries = 5
    base_delay = 1

    for retry in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            break
        except Exception as e:
            if retry == max_retries - 1:
                raise e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                print(f"\nRate limited, retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                continue
            raise e

    # Extract and validate JSON from response
    try:
        text = response.text.strip()
        # Try to find valid JSON in the response
        json_str = ""
        depth = 0
        start_idx = text.find('{')
        
        if start_idx >= 0:
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = text[start_idx:i+1]
                        break
        
        if not json_str:
            print(f"\nNo valid JSON found in response: {text}")
            return []

        # Parse and validate JSON structure
        analysis = json.loads(json_str)
        if not isinstance(analysis, dict):
            print("\nInvalid JSON structure: not an object")
            return []
            
        criticism_moments = analysis.get('criticism_moments', [])
        if not isinstance(criticism_moments, list):
            print("\nInvalid criticism_moments: not an array")
            return []
        
        # Take top 30 by score first
        criticism_moments.sort(key=lambda x: x.get('score', 0), reverse=True)
        result = criticism_moments[:30]
        
        # Then sort by timestamp after matching with transcription
        
        # Cache the result
        CacheManager.write_cache(cache_path, result)
        return result
    except Exception as e:
        print(f"Error processing analysis: {e}")
        return []


def generate_voice_narration(text: str, output_path: Path) -> bool:
    try:
        print(f"\nGenerating voice narration for: {text}")
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text
        )
        
        # Save the audio file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error generating voice narration: {e}")
        return False


def create_montage(video_path: Path, criticism_moments: List[Dict], output_path: Path) -> bool:
    if not criticism_moments:
        print("\nNo criticism moments to create montage from")
        return False

    temp_dir = Path('temp_clips')
    temp_dir.mkdir(exist_ok=True)
    narration_dir = Path('temp_narration')
    narration_dir.mkdir(exist_ok=True)
    progress = ProgressBar(len(criticism_moments) * 2, "Creating clips: ")

    clip_pairs = []
    for idx, moment in enumerate(criticism_moments):
        try:
            # Use precise timestamps from transcription
            start_time = max(0, moment.get('start_time', 0))
            end_time = moment.get('end_time', 0)
            if not end_time or end_time <= start_time:
                print(f"Invalid timestamps for clip {idx}")
                progress.update(2)
                continue
            duration = end_time - start_time
            
            clip_path = temp_dir / f"clip_{idx:03d}.mp4"
            print(f"\nCreating clip {idx} from {start_time:.2f}s")
            
            # Generate narration if needed
            narration_path = narration_dir / f"narration_{idx:03d}.mp3"
            reason = moment.get('reason', '')
            if reason and not narration_path.exists():
                score = moment.get('score', 0)
                narration_text = f"This has a criticism-worthiness score of {score}. {reason}"
                success = generate_voice_narration(narration_text, narration_path)
                if not success:
                    print(f"Failed to generate narration for clip {idx}")
                    progress.update(2)
                    continue

            # Get durations
            narration_duration = 0
            if narration_path.exists():
                narration_duration = FFmpegHelper.get_duration(narration_path) or 0
            
            # Create clip that freezes immediately with narration
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', str(video_path)
            ]
            
            if narration_path.exists():
                cmd.extend(['-i', str(narration_path)])
                
                # Complex filter to play clip then freeze frame with narration
                filter_complex = [
                    # Split original video into moving part and frame to freeze
                    '[0:v]split[moving][to_freeze]',
                    # Get last frame and extend it for narration duration
                    f'[to_freeze]select=1[frozen_frame]',
                    f'[frozen_frame]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={narration_duration}[frozen_vid]',
                    # Process original audio and narration
                    '[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[orig_audio]',
                    '[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume=0.7[narration]',
                    # Concatenate video (moving + frozen) and audio (original + narration)
                    '[moving][frozen_vid]concat=n=2:v=1:a=0[outv]',
                    '[orig_audio][narration]concat=n=2:v=0:a=1[outa]'
                ]
                
                cmd.extend([
                    '-filter_complex', ';'.join(filter_complex),
                    '-map', '[outv]',
                    '-map', '[outa]'
                ])
            else:
                cmd.extend(['-map', '0:v', '-map', '0:a'])
            
            cmd.extend([
                '-c:v', 'libx264', '-preset', 'veryfast',
                '-c:a', 'aac', '-b:a', '192k',
                '-ar', '48000', '-ac', '2',
                str(clip_path),
                '-y'
            ])
            
            success, error = FFmpegHelper.run_command(cmd)
            if not success or not clip_path.exists():
                print(f"Error creating clip {idx}: {error}")
                progress.update(2)
                continue

            clip_size = os.path.getsize(clip_path)
            if clip_size == 0:
                print(f"Clip file {clip_path} is empty")
                os.remove(clip_path)
                progress.update(2)
                continue

            print(f"Successfully created clip {idx}")
            progress.update(2)
            clip_pairs.append((clip_path, None))

        except Exception as e:
            print(f"Exception creating clip pair {idx}: {str(e)}")
            progress.update(2)

    if not clip_pairs:
        print("\nNo valid clips were created!")
        return False

    print(f"\nSuccessfully created {len(clip_pairs)} clip pairs")

    try:
        print("\nMerging clips into final montage...")
        concat_file = temp_dir / 'concat.txt'

        # Write concat file with absolute paths
        with open(concat_file, 'w') as f:
            for clip_path, _ in clip_pairs:
                f.write(f"file '{clip_path.absolute()}'\n")

        print(f"Created concat file at {concat_file}")
        print("Concat file contents:")
        with open(concat_file, 'r') as f:
            print(f.read())

        cmd = [
            'ffmpeg', '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k',
            '-ar', '48000', '-ac', '2',
            str(output_path),
            '-y'
        ]
        success, error = FFmpegHelper.run_command(cmd)

        if not success:
            print(f"\nError merging clips: {error}")
        else:
            print("\nMontage creation complete!")

        return success
    finally:
        # Cleanup after concat is complete
        FileManager.cleanup_files([temp_dir])
        # Clean up any temporary frame images
        for frame_file in temp_dir.glob('frame_*.jpg'):
            try:
                os.remove(frame_file)
            except Exception:
                pass


def main():
    video_path = Path('source.mp4')
    print(f"\nChecking for source video at {video_path}...")
    if not video_path.exists():
        print("Error: source.mp4 not found!")
        return

    output_path = Path('criticism_montage.mp4')
    print(f"Checking for existing montage at {output_path}...")
    if output_path.exists():
        print("Criticism montage already exists! Skipping all processing.")
        return

    try:
        print("\nStarting criticism montage creation...")

        # Check for existing audio file
        audio_path = video_path.with_suffix('.mp3')
        print(f"Checking for existing audio at {audio_path}...")
        if not audio_path.exists():
            print("No existing audio found. Extracting audio...")
            audio_path = extract_audio(video_path)
            if not audio_path:
                print("Error extracting audio!")
                return
        else:
            print("Found existing audio file, skipping extraction...")

        # Transcribe the entire audio file
        print("\nTranscribing audio...")
        words = transcribe_audio(audio_path)
        if not words:
            print("Error transcribing audio!")
            return

        # Get full text for analysis
        full_text = ' '.join(word['text'] for word in words)
        
        # Analyze the full text for criticism
        print("\nAnalyzing for criticism...")
        criticism_moments = analyze_criticism(full_text)
        
        # Match criticism moments with timestamps
        for moment in criticism_moments:
            text = moment['text']
            words_in_moment = text.split()
            
            # Find the timestamp for this text in the transcription
            text = text.lower().strip()
            best_match = None
            best_match_score = 0
            
            for i in range(len(words) - len(words_in_moment) + 1):
                transcribed_words = words[i:i + len(words_in_moment)]
                transcribed_text = ' '.join(w['text'] for w in transcribed_words).lower().strip()
                
                # Calculate similarity score
                words_matched = sum(1 for w1, w2 in zip(text.split(), transcribed_text.split()) if w1 == w2)
                total_words = max(len(text.split()), len(transcribed_text.split()))
                score = words_matched / total_words if total_words > 0 else 0
                
                if score > best_match_score:
                    best_match_score = score
                    best_match = transcribed_words
            
            # Use the best match if it's good enough
            if best_match_score > 0.8:  # At least 80% word match
                moment['start_time'] = best_match[0]['start']
                moment['end_time'] = best_match[-1]['end']
            else:
                print(f"\nNo good match found for text: {text}")
                print(f"Best match ({best_match_score*100:.1f}%): {' '.join(w['text'] for w in best_match) if best_match else 'None'}")

        if criticism_moments:
            print(f"\nFound {len(criticism_moments)} moments worthy of criticism")
            
            # Sort moments chronologically by timestamp
            criticism_moments.sort(key=lambda x: x.get('start_time', 0))
            
            # Create the montage
            success = create_montage(
                video_path,
                criticism_moments,
                output_path
            )

            if success:
                print("Successfully created criticism_montage.mp4!")
            else:
                print("Error creating montage!")
        else:
            print("No moments worthy of criticism found!")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Only cleanup temporary files, not the audio file since we want to reuse it
        FileManager.cleanup_files([
            Path('temp_clips'),
            Path('temp_narration')
        ])


if __name__ == "__main__":
    main()
