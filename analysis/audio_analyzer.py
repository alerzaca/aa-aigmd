import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf
import librosa
from mutagen.mp3 import MP3
from mutagen import File as MutagenFile

## Mapa rozmiaru - ile bitów zajmuje zapis jednej próbki.
## PCM - nieskompresowane próbki całkowite: S/U oznacza zapis ze znakiem/bez, a liczba to bity na próbkę.
## FLOAT/DOUBLE - zapisują ułamki.
## ULAW/ALAW - nieliniowa kompresuja amplitudy (8 bitów na próbkę).
SUBTYPE_BIT_DEPTH = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT":  32,
    "DOUBLE": 64,
    "ULAW":   8,
    "ALAW":   8,
}

## Stereo - dwa kanały lewy/prawy tworzące wspólny obraz dźwięku, dynamiczna pula bitrate'u.
## Dual channel - dwie niezależne ścieżki mono, sztywno przydzielona pula bitrate'u.
## Joint stereo - dwa kanały lewy/prawy oraz środkowy mid: wspólna informacja jest zapisywana tylko raz, co oszczędza miejsce.
MP3_CHANNEL_MODES = {0: "stereo", 1: "joint_stereo", 2: "dual_channel", 3: "mono"}

## Tagi parsowane przez Mutagen - bibliotekę do odczytu metadanych audio.
EASY_TAG_KEYS = [
    ("title",       "title"),
    ("artist",      "artist"),
    ("album",       "album"),
    ("date",        "year"),
    ("tracknumber", "track"),
    ("genre",       "genre"),
    ("comment",     "comment"),
    ("composer",    "composer"),
    ("albumartist", "album_artist"),
    ("bpm",         "bpm"),
]


def _db(linear: float) -> float:
    ## Konwersja amplitudy do dBFS (Decibels Relative to Full Scale, takie jak w DAW, 0 to maksymalna wartość, powyżej tego jest clipping)
    ## abs() chroni przed wartościami ujemnymi fali dźwiękowej
    ## max(..., 1e-10) chroni przed logarytmem z zera
    return 20.0 * math.log10(max(abs(linear), 1e-10))


def _read_tags(path: Path) -> dict:
    ## Odczytanie obsługiwanych tagów audio.
    audio = MutagenFile(str(path), easy=True)
    if audio is None:
        return {}
    tags = {}
    for src_key, dst_key in EASY_TAG_KEYS:
        val = audio.get(src_key)
        if val:
            tags[dst_key] = val[0] if len(val) == 1 else list(val)
    return tags


def _signal_metrics(data: np.ndarray) -> dict:
    ## Oblicz poziomy sygnału dla tablicy próbek w układzie (ramki, kanały)
    num_channels = data.shape[1]

    squared_data = data ** 2
    absolute_data = np.abs(data)
    rms_all = float(np.sqrt(np.mean(squared_data)))
    peak_all = float(np.max(absolute_data))

    ## Statystyki per kanał są liczone tylko dla nagrań wielokanałowych.
    if num_channels > 1:
        rms_channels = np.sqrt(np.mean(squared_data, axis=0))
        peak_channels = np.max(absolute_data, axis=0)
        dc_channels = np.mean(data, axis=0)
        per_ch = [
            {
                "channel": ch,
                "rms_dbfs": round(_db(rms), 2),
                "peak_dbfs": round(_db(peak), 2),
                "dc_offset": round(float(dc), 6),
                "crest_factor_db": round(_db(peak / rms), 2) if rms > 0 else None,
            }
            for ch, (rms, peak, dc) in enumerate(zip(rms_channels, peak_channels, dc_channels), start=1)
        ]
    else:
        per_ch = []

    return {
        "rms_dbfs":        round(_db(rms_all), 2),
        "peak_dbfs":       round(_db(peak_all), 2),
        "headroom_db":     round(-_db(peak_all), 2),
        "per_channel":     per_ch,
    }


def _analyze_wav(path: Path) -> dict:
    ## Analiza WAV: Odczytanie właściwości i pomiar poziomów zdekodowanych próbek.
    with sf.SoundFile(str(path)) as audio:
        data = audio.read(always_2d=True, dtype="float64")
        file_format, subtype, sample_rate = audio.format, audio.subtype, audio.samplerate

    metrics = _signal_metrics(data)
    duration = data.shape[0] / sample_rate

    return {
        "format":          file_format,
        "subtype":         subtype,       ## sposób zapisu próbek, np. 16-bitowe PCM lub zmiennoprzecinkowe FLOAT
        "bit_depth":       SUBTYPE_BIT_DEPTH.get(subtype),
        "channels":        data.shape[1],
        "channel_layout":  {1: "mono", 2: "stereo"}.get(data.shape[1], f"{data.shape[1]}ch"),
        "sample_rate_hz":  sample_rate,
        "total_frames":    data.shape[0],
        "duration_seconds":round(duration, 6),
        "duration_hms":    _seconds_to_hms(duration),
        **metrics,
    }


def _analyze_mp3(path: Path) -> dict:
    ## Analiza MP3: odczyt metadanych i dekodowanie próbek do pomiaru poziomów.
    audio = MP3(str(path))
    info  = audio.info

    mode_id   = info.mode   ## 0=stereo, 1=joint stereo, 2=dual channel, 3=mono
    samples, sample_rate = librosa.load(str(path), sr=None, mono=False, dtype=np.float64)

    ## librosa zwraca próbki w układzie (kanały, ramki); normalizuj do (ramki, kanały).
    data = samples[:, np.newaxis] if samples.ndim == 1 else samples.T
    metrics = _signal_metrics(data)
    duration = data.shape[0] / sample_rate

    return {
        "format":           "MP3",
        "subtype":          "lossy",
        "bit_depth":        None,              ## kompresja MP3 nie przechowuje próbek o jednej stałej głębi bitowej
        "bitrate_kbps":     round(info.bitrate / 1000),
        "bitrate_mode":     _mp3_bitrate_mode(audio),
        "channels":         data.shape[1],
        "channel_layout":   MP3_CHANNEL_MODES.get(mode_id, "unknown"),
        "sample_rate_hz":   sample_rate,
        "total_frames":     data.shape[0],
        "duration_seconds": round(duration, 6),
        "duration_hms":     _seconds_to_hms(duration),
        "encoder":          getattr(info, "encoder_info", None) or None,
        **metrics,
    }


def _mp3_bitrate_mode(audio: MP3) -> str:
    ## Rozpoznaj CBR, VBR lub ABR z nagłówków MP3 odczytanych przez Mutagen
    bm = getattr(audio.info, "bitrate_mode", None)
    if bm is not None:
        return {0: "unknown", 1: "CBR", 2: "VBR", 3: "ABR"}.get(int(bm), "unknown")
    return "unknown"


def _seconds_to_hms(seconds: float) -> str:
    ## Zapisz czas z dokładnością do milisekund; dodaj godziny, gdy są niezerowe.
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def _file_meta(path: Path) -> dict:
    ## Zbierz systemowe informacje o pliku potrzebne do identyfikacji go w raporcie.
    stat = path.stat()
    size_bytes = stat.st_size
    return {
        "name":           path.name,
        "stem":           path.stem,
        "extension":      path.suffix.lower(),
        "path":           str(path.resolve()),
        "size_bytes":     size_bytes,
        "size_kb":        round(size_bytes / 1024, 2),
        "size_mb":        round(size_bytes / (1024 ** 2), 4),
        "modified":       datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


## Jedno źródło prawdy dla rozszerzeń obsługiwanych przez analizę i skanowanie katalogu.
ANALYZERS = {".wav": _analyze_wav, ".mp3": _analyze_mp3}


def analyze_file(path: Path) -> dict:
    ## Wybierz analizator na podstawie rozszerzenia.
    result = _file_meta(path)
    analyzer = ANALYZERS.get(path.suffix.lower())

    ## Błąd obsługi pojedynczego pliku trafia do jego wyniku, nie przerywa raportu katalogu.
    try:
        if analyzer is None:
            result["error"] = f"Unsupported extension: {path.suffix.lower()}"
            return result

        result.update(analyzer(path))
        tags = _read_tags(path)
        if tags:
            result["tags"] = tags

    ## Dołącz przyczynę błędu do wyniku zamiast odrzucać informacje o tym pliku.
    except Exception as exc:
        result["error"] = str(exc)

    return result


def collect_files(directory: Path) -> list[Path]:
    ## Zbierz obsługiwane pliki.
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in ANALYZERS
    )


def analyze_directory(directory: str, output_path: str | None = None) -> None:
    ## Sprawdź katalog, przeanalizuj jego pliki i wypisz lub zapisz zbiorczy raport JSON.
    dir_path = Path(directory)

    ## Odróżnij brak ścieżki od pliku podanego zamiast katalogu, by wyjaśnić błąd wejścia.
    if not dir_path.exists():
        sys.exit(f"Error: '{directory}' does not exist.")
    if not dir_path.is_dir():
        sys.exit(f"Error: '{directory}' is not a directory.")

    files = collect_files(dir_path)
    if not files:
        sys.exit(f"No WAV/MP3 files found in '{directory}'.")

    _log(f"Found {len(files)} audio file(s). Analysing…")

    results = []
    for i, fp in enumerate(files, 1):
        _log(f"  [{i:>3}/{len(files)}] {fp.name}")
        results.append(analyze_file(fp))

    ## Sumuj tylko udane odczyty czasu; wpisy błędów nie zawierają duration_seconds.
    total_duration = sum(
        result["duration_seconds"]
        for result in results
        if result.get("duration_seconds") is not None
    )

    summary = {
        "total_files":            len(results),
        "total_duration_seconds": round(total_duration, 3),
        "total_duration_hms":     _seconds_to_hms(total_duration),
        "files":                  results,
    }

    output_text = json.dumps(summary, indent=2, ensure_ascii=False)

    ## -o wybiera plik docelowy; bez tej opcji raport jest wypisywany w terminalu.
    if output_path:
        Path(output_path).write_text(output_text, encoding="utf-8")
        _log(f"\nReport saved to {output_path}")
    else:
        print(output_text)


def _log(msg: str) -> None:
    ## Wypisz postęp na stderr, pozostawiając stdout jako czyste wyjście raportu.
    print(msg, file=sys.stderr)


def main() -> None:
    ## Zdefiniuj argumenty wiersza poleceń i uruchom analizę wybranego katalogu.
    parser = argparse.ArgumentParser(description="Analyze WAV/MP3 files in a directory and output a JSON report.",)
    parser.add_argument("directory", help="Path to the directory containing audio files.")
    parser.add_argument("-o", "--output", metavar="FILE", help="Write report to FILE instead of stdout.")

    args = parser.parse_args()
    analyze_directory(directory=args.directory, output_path=args.output)


if __name__ == "__main__":
    main()